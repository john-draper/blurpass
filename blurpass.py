#!/usr/bin/env python
"""BlurPass - scan videos for gore/nudity with local models and blur flagged segments.

Usage examples:
    python blurpass.py scan "movie.mkv"                        # scan + report, no render
    python blurpass.py scan "movie.mkv" --strictness high
    python blurpass.py scan "movie.mkv" --categories gore      # one category only
    python blurpass.py process "movie.mkv"                     # scan + render movie.filtered.mkv
    python blurpass.py process D:\\Videos --recursive          # batch a whole library
    python blurpass.py process "movie.mkv" --reuse-scan --dry-run
    python blurpass.py selftest                                # synthetic end-to-end test
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import tomllib
from pathlib import Path

import detector
import renderer
import report
import sampling
from segments import FrameVerdict, Segment, build_segments, merge_segments

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config.toml"
REPORTS_DIR = PROJECT_DIR / "reports"
DEFAULT_STRICTNESS = "medium"


def load_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def clamp01(x: float) -> float:
    return max(0.05, min(0.95, x))


def apply_strictness(cfg: dict, level: str | None) -> dict:
    presets = cfg.get("presets", {})
    if level and level in presets:
        p = presets[level]
        for cat in cfg["categories"]["enabled"]:
            if cat in cfg:
                cfg[cat]["clip_threshold"] = clamp01(cfg[cat]["clip_threshold"] + p.get("clip_shift", 0.0))
                cfg[cat]["vlm_threshold"] = clamp01(cfg[cat]["vlm_threshold"] + p.get("vlm_shift", 0.0))
    return cfg


def collect_videos(inputs: list[str], recursive: bool) -> list[Path]:
    out: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            out.extend(
                f for f in sorted(it)
                if f.is_file()
                and f.suffix.lower() in sampling.VIDEO_EXTS
                and ".filtered." not in f.name
            )
        else:
            print(f"warning: skipping nonexistent path {p}")
    return out


class BlurPass:
    def __init__(self, cfg: dict, use_vlm: bool = True, low_vram: bool = False,
                 categories: list[str] | None = None):
        self.cfg = cfg
        self.categories = tuple(categories or cfg["categories"]["enabled"])
        unknown = [c for c in self.categories if c not in detector.CATEGORY_PROMPTS]
        if unknown:
            raise ValueError(f"unknown categories {unknown}; known: {list(detector.CATEGORY_PROMPTS)}")
        self.use_vlm = use_vlm
        if low_vram:
            cfg["clip"]["device"] = "cpu"
        print(f"[init] loading CLIP ({cfg['clip']['model']}/{cfg['clip']['pretrained']}) "
              f"for {', '.join(self.categories)}...")
        t0 = time.time()
        self.clip = detector.load_clip(cfg, self.categories)
        print(f"[init] CLIP ready on {self.clip.device} in {time.time() - t0:.1f}s")
        self.vlm = None
        if use_vlm and cfg["vlm"].get("backend", "none") != "none":
            backend = cfg["vlm"]["backend"]
            print(f"[init] loading VLM ({backend})... (first run downloads weights)")
            t0 = time.time()
            self.vlm = detector.load_vlm(cfg)
            print(f"[init] VLM ready on {self.vlm.device} in {time.time() - t0:.1f}s")
        elif not use_vlm:
            print("[init] VLM disabled (--no-vlm); relying on CLIP scores only")

    def scan(self, video: Path, keep_frames: bool = True) -> dict:
        """Full scan of one video across all enabled categories."""
        cfg = self.cfg
        t_start = time.time()
        print(f"[scan] {video.name}")

        info = sampling.probe(video)
        print(f"  duration {info.duration:.0f}s, {info.width}x{info.height}, "
              f"audio={'yes' if info.has_audio else 'no'}")

        report_dir = REPORTS_DIR / video.stem
        frames_dir = report_dir / "frames"
        # clear stale frames from any previous scan (counts differ between runs)
        shutil.rmtree(frames_dir, ignore_errors=True)
        frames, eff_fps = sampling.sample_frames(
            video, frames_dir,
            fps=cfg["sampling"]["fps"],
            thumb_size=cfg["sampling"]["thumb_size"],
            max_frames=cfg["sampling"]["max_frames"],
            duration=info.duration,
        )
        print(f"  sampled {len(frames)} frames at {eff_fps}/s")

        print(f"  CLIP triage on {len(frames)} frames ({', '.join(self.categories)})...")
        t0 = time.time()
        clip_scores = self.clip.score_multi([f.path for f in frames], list(self.categories))
        print(f"  CLIP done in {time.time() - t0:.1f}s")

        backend = cfg["vlm"].get("backend", "none") if self.vlm else "none"
        seg_cfg = cfg["segments"]
        all_segments: list[Segment] = []
        frame_records: dict[str, dict] = {f.path.name: {} for f in frames}
        cat_settings: dict[str, dict] = {}

        for cat in self.categories:
            cs = clip_scores[cat]
            cat_cfg = cfg[cat]
            clip_thr = cat_cfg["clip_threshold"]
            vlm_thr = cat_cfg["vlm_threshold"]
            cat_settings[cat] = {"clip_threshold": clip_thr, "vlm_threshold": vlm_thr}
            flagged_idx = [i for i, s in enumerate(cs) if s >= clip_thr]
            print(f"  [{cat}] {len(flagged_idx)} frames over CLIP threshold {clip_thr}")

            vlm_scores: dict[int, float] = {}
            if self.vlm and flagged_idx:
                cap = cfg["vlm"].get("max_frames", 4000)
                if len(flagged_idx) > cap:
                    # spend the budget on the most CLIP-suspicious frames, not the earliest
                    to_check = sorted(flagged_idx, key=lambda i: cs[i], reverse=True)[:cap]
                    print(f"  [{cat}] WARNING: {len(flagged_idx)} frames flagged but VLM cap is {cap}; "
                          f"checking the {cap} most suspicious, treating the rest as clean "
                          f"(raise vlm.max_frames or the clip threshold)")
                else:
                    to_check = flagged_idx
                print(f"  [{cat}] VLM confirming {len(to_check)} frames (threshold {vlm_thr}), "
                      f"3-frame context each...")
                t0 = time.time()
                n_total = len(frames)
                groups = [
                    [frames[j].path for j in range(max(0, i - 1), min(n_total, i + 2))]
                    for i in to_check
                ]
                probs = self.vlm.confirm(groups, category=cat)
                vlm_scores = dict(zip(to_check, probs))
                done = time.time() - t0
                print(f"  [{cat}] VLM done in {done:.0f}s ({len(to_check) / max(done, 0.001):.2f} frames/s)")
                confirmed = sum(1 for p in probs if p >= vlm_thr)
                print(f"  [{cat}] VLM confirmed {confirmed}/{len(to_check)} flagged frames")

            for i, f in enumerate(frames):
                frame_records[f.path.name][cat] = {
                    "clip": round(cs[i], 4),
                    "vlm": None if i not in vlm_scores else round(vlm_scores[i], 4),
                }

            verdicts = [
                FrameVerdict(
                    timestamp=frames[i].timestamp,
                    clip_score=cs[i],
                    vlm_score=vlm_scores.get(i),
                )
                for i in range(len(frames))
            ]
            cat_segs = build_segments(
                verdicts,
                step=1.0 / eff_fps,
                duration=info.duration,
                clip_threshold=clip_thr,
                vlm_threshold=vlm_thr,
                merge_gap=seg_cfg["merge_gap"],
                pad=seg_cfg["pad"],
                min_duration=seg_cfg["min_duration"],
                vlm_active=self.vlm is not None,
            )
            for s in cat_segs:
                s.category = cat
            all_segments.extend(cat_segs)
            blur = sum(s.duration for s in cat_segs)
            print(f"  [{cat}] -> {len(cat_segs)} segment(s), {blur:.0f}s total blur")

        all_segments.sort(key=lambda s: s.start)

        frame_dicts = [
            {
                "frame": f.path.name,
                "timestamp": round(f.timestamp, 3),
                **{cat: {
                    "clip": frame_records[f.path.name][cat]["clip"],
                    "vlm": frame_records[f.path.name][cat]["vlm"],
                    "flagged": (
                        frame_records[f.path.name][cat]["vlm"] is not None
                        and frame_records[f.path.name][cat]["vlm"] >= cat_settings[cat]["vlm_threshold"]
                    ) or (
                        frame_records[f.path.name][cat]["vlm"] is None
                        and self.vlm is None
                        and frame_records[f.path.name][cat]["clip"] >= cat_settings[cat]["clip_threshold"]
                    ),
                } for cat in self.categories},
            }
            for f in frames
        ]
        seg_dicts = [
            {
                "category": s.category,
                "start": s.start,
                "end": s.end,
                "duration": round(s.duration, 2),
                "peak_clip": round(s.peak_clip, 4),
                "peak_vlm": None if s.peak_vlm is None else round(s.peak_vlm, 4),
                "frames": [
                    fd for fd in frame_dicts
                    if fd[s.category]["flagged"]
                    and s.start - seg_cfg["pad"] <= fd["timestamp"] < s.end
                ],
            }
            for s in all_segments
        ]
        settings = {
            "categories": list(self.categories),
            "sampling_fps": eff_fps,
            "clip_model": f"{cfg['clip']['model']}/{cfg['clip']['pretrained']}",
            "vlm_backend": backend,
            "thresholds": cat_settings,
            "merge_gap": seg_cfg["merge_gap"],
            "pad": seg_cfg["pad"],
            "min_duration": seg_cfg["min_duration"],
        }

        report.write_json(
            report_dir / "report.json",
            video=str(video),
            info={"duration": info.duration, "width": info.width, "height": info.height,
                  "has_audio": info.has_audio},
            frames=frame_dicts, segments=seg_dicts, settings=settings,
        )
        report.write_html(
            report_dir / "report.html",
            video=str(video),
            info={"duration": info.duration},
            frames=frame_dicts, segments=seg_dicts, settings=settings,
            max_thumbs_per_segment=cfg["report"]["max_thumbs_per_segment"],
            keep_frames_dir=frames_dir,
        )
        print(f"  report: {report_dir / 'report.html'}")

        if not keep_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)

        return {
            "video": video,
            "info": info,
            "report_dir": report_dir,
            "settings": settings,
            "segments": all_segments,
            "elapsed": time.time() - t_start,
        }


def load_prior_scan(video: Path) -> dict | None:
    rj = REPORTS_DIR / video.stem / "report.json"
    if not rj.exists():
        return None
    import json
    with open(rj, "r", encoding="utf-8") as f:
        return json.load(f)


def process_one(bp: BlurPass, video: Path, args) -> bool:
    """Scan (or reuse) + render one video. Returns True on success."""
    cfg = bp.cfg
    prior = load_prior_scan(video) if args.reuse_scan else None

    if prior:
        print(f"[process] {video.name}: reusing prior scan from {REPORTS_DIR / video.stem}")
        segs = [Segment(s["start"], s["end"], s.get("category", "gore"))
                for s in prior["segments"]]
        info = sampling.VideoInfo(
            duration=prior["info"]["duration"], width=prior["info"]["width"],
            height=prior["info"]["height"], has_audio=prior["info"]["has_audio"],
        )
    else:
        result = bp.scan(video)
        segs = result["segments"]
        info = result["info"]

    merged = merge_segments(segs)
    out_dir = Path(args.out) if args.out else video.parent
    suffix = args.suffix if args.suffix.startswith(".") else f".{args.suffix}"
    out_path = out_dir / f"{video.stem}{suffix}{video.suffix}"

    if args.dry_run:
        print(f"[dry-run] would render {out_path} with {len(merged)} blur window(s):")
        for i, s in enumerate(merged, 1):
            print(f"  #{i}: {s.start:8.2f}s -> {s.end:8.2f}s  ({s.duration:.1f}s) [{s.category}]")
        return True

    if out_path.exists() and not args.overwrite:
        print(f"[process] output exists, skipping (use --overwrite): {out_path}")
        return True

    print(f"[render] {len(segs)} segment(s) -> {len(merged)} blur window(s) -> {out_path.name}")
    t0 = time.time()
    r_cfg = cfg["render"]
    encoder_used = renderer.render(
        video, merged, out_path,
        filter_name=r_cfg["filter"], sigma=r_cfg["sigma"],
        encoder_setting=r_cfg["encoder"], cq=r_cfg["cq"],
        has_audio=info.has_audio,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"[render] done in {time.time() - t0:.0f}s with {encoder_used} ({size_mb:.1f} MB)")
    return True


def cmd_selftest(args) -> int:
    """Generate synthetic videos and push them through the whole machine."""
    import subprocess

    test_dir = PROJECT_DIR / "test"
    test_dir.mkdir(exist_ok=True)
    clean = test_dir / "synth_clean.mp4"
    red = test_dir / "synth_red.mp4"

    print("[selftest] generating synthetic videos...")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=12",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(clean)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=0x8a0f0f:size=640x360:rate=24:duration=12",
         "-vf", "noise=alls=20:allf=t",
         "-c:v", "libx264", "-preset", "ultrafast", str(red)],
        check=True,
    )

    cfg = load_config()
    gf = BlurPass(cfg, use_vlm=not args.no_vlm)

    failures = 0
    for vid in (clean, red):
        print()
        result = gf.scan(vid)
        n_frames = len(list((REPORTS_DIR / vid.stem / "frames").glob("*.jpg"))) \
            if (REPORTS_DIR / vid.stem / "frames").exists() else -1
        ok_report = (REPORTS_DIR / vid.stem / "report.json").exists() \
            and (REPORTS_DIR / vid.stem / "report.html").exists()
        print(f"[selftest] {vid.name}: frames={n_frames}, segments={len(result['segments'])}, "
              f"report_files={'ok' if ok_report else 'MISSING'}, "
              f"scan_time={result['elapsed']:.1f}s")
        if n_frames < 10 or not ok_report:
            failures += 1

    # render pass with a forced segment exercising both categories + merge
    print()
    print("[selftest] render pass with forced overlapping windows (gore + nudity)...")
    out = test_dir / "synth_clean.filtered.mp4"
    enc = renderer.render(
        clean, [Segment(2.0, 5.0, "gore"), Segment(4.0, 8.0, "nudity")], out,
        filter_name=cfg["render"]["filter"], sigma=cfg["render"]["sigma"],
        encoder_setting=cfg["render"]["encoder"], cq=cfg["render"]["cq"],
        has_audio=True,
    )
    ok = out.exists() and out.stat().st_size > 0
    print(f"[selftest] render {'ok' if ok else 'FAILED'} (encoder: {enc}, {out.stat().st_size / 1e6:.1f} MB)")
    if not ok:
        failures += 1

    # segment merge unit check
    merged = merge_segments([Segment(2.0, 5.0, "gore"), Segment(4.0, 8.0, "nudity"),
                             Segment(9.0, 10.0, "gore")])
    expect = [(2.0, 8.0, "merged"), (9.0, 10.0, "gore")]
    got = [(s.start, s.end, s.category) for s in merged]
    print(f"[selftest] merge_segments: {got} {'ok' if got == expect else 'FAILED'}")
    if got != expect:
        failures += 1

    print()
    print("[selftest] PASS" if failures == 0 else f"[selftest] {failures} FAILURE(S)")
    return 0 if failures == 0 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="blurpass",
        description="BlurPass: scan videos for gore/nudity with local models; blur flagged segments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage examples:")[1] if __doc__ else None,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("inputs", nargs="+", help="video file(s) or folder(s)")
        p.add_argument("--strictness", choices=["low", "medium", "high"], default=DEFAULT_STRICTNESS,
                       help="threshold preset (default: medium)")
        p.add_argument("--no-vlm", action="store_true", help="skip VLM confirmation (CLIP only)")
        p.add_argument("--low-vram", action="store_true", help="run CLIP on CPU")
        p.add_argument("--fps", type=float, help="override sampling fps")
        p.add_argument("--categories", help="comma-separated categories (default: config enabled list)")
        p.add_argument("--recursive", action="store_true", help="recurse into folders")

    p_scan = sub.add_parser("scan", help="scan and report; no video is modified")
    common(p_scan)

    p_proc = sub.add_parser("process", help="scan and render a blurred copy")
    common(p_proc)
    p_proc.add_argument("--out", help="output directory (default: alongside input)")
    p_proc.add_argument("--suffix", default=".filtered", help="output suffix (default: .filtered)")
    p_proc.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")
    p_proc.add_argument("--reuse-scan", action="store_true",
                        help="reuse a previous scan's report instead of rescanning")
    p_proc.add_argument("--dry-run", action="store_true", help="show blur windows; do not render")

    p_test = sub.add_parser("selftest", help="synthetic end-to-end test")
    p_test.add_argument("--no-vlm", action="store_true")

    args = ap.parse_args(argv)

    if args.command == "selftest":
        return cmd_selftest(args)

    cfg = load_config()
    apply_strictness(cfg, args.strictness)
    if args.fps:
        cfg["sampling"]["fps"] = args.fps
    categories = args.categories.split(",") if getattr(args, "categories", None) else None

    videos = collect_videos(args.inputs, getattr(args, "recursive", False))
    if not videos:
        print("no videos found")
        return 1
    print(f"{len(videos)} video(s) queued")

    try:
        bp = BlurPass(cfg, use_vlm=not args.no_vlm, low_vram=args.low_vram,
                      categories=categories)
    except Exception as e:
        print(f"failed to load models: {e}")
        return 2

    failed = []
    for i, vid in enumerate(videos, 1):
        print(f"\n=== [{i}/{len(videos)}] {vid} ===")
        try:
            if args.command == "scan":
                bp.scan(vid, keep_frames=False)
            else:
                process_one(bp, vid, args)
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(vid)

    print()
    if failed:
        print(f"done with {len(failed)} failure(s):")
        for v in failed:
            print(f"  {v}")
        return 1
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
