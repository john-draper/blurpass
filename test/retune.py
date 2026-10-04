"""Retune-in-place: run the VLM over CLIP band frames the current gate skipped,
then rebuild segments at candidate threshold combos. No rescan needed.

Usage: retune.py <report_dir> [backend]
  e.g. python test/retune.py reports/MyMovie qwen25vl_7b
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import detector
import tomllib
from segments import FrameVerdict, build_segments


def mmss(t):
    m, s = divmod(t, 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{int(m):02d}:{s:04.1f}" if h else f"{int(m):02d}:{s:04.1f}"


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: retune.py <report_dir> [backend]")
    report_path = Path(sys.argv[1]) / "report.json"
    backend = sys.argv[2] if len(sys.argv) > 2 else "qwen25vl_7b"
    r = json.load(open(report_path))
    frames = r["frames"]
    st = r["settings"]
    fps = st["sampling_fps"]
    duration = r["info"]["duration"]
    cur_clip = st["clip_threshold"]

    band = [i for i, f in enumerate(frames)
            if f["vlm_score"] is None and f["clip_score"] >= cur_clip - 0.15]
    print(f"{len(band)} band frames (clip {cur_clip - 0.15:.2f}-{cur_clip}) never sent to the VLM")

    cfg = tomllib.load(open(Path(__file__).parent.parent / "config.toml", "rb"))
    cfg["vlm"]["backend"] = backend
    vlm = detector.load_vlm(cfg)

    frame_files = sorted((report_path.parent / "frames").glob("frame_*.jpg"))
    groups = [
        [frame_files[j] for j in range(max(0, i - 1), min(len(frame_files), i + 2))]
        for i in band
    ]
    probs = vlm.confirm(groups)
    for i, p in zip(band, probs):
        frames[i]["vlm_score"] = round(p, 4)

    verdicts = [FrameVerdict(f["timestamp"], f["clip_score"], f["vlm_score"]) for f in frames]
    seg_cfg = cfg["segments"]

    def build(clip_thr, vlm_thr):
        return build_segments(
            verdicts, step=1.0 / fps, duration=duration,
            clip_threshold=clip_thr, vlm_threshold=vlm_thr,
            merge_gap=seg_cfg["merge_gap"], pad=seg_cfg["pad"],
            min_duration=seg_cfg["min_duration"], vlm_active=True,
        )

    print("\ncandidate combos (clip gate x vlm threshold):")
    combos = sorted({(cur_clip, thr) for thr in (0.5, 0.6, 0.7)} |
                    {(cur_clip - 0.15, thr) for thr in (0.5, 0.6, 0.7)})
    for clip_thr, vlm_thr in combos:
        segs = build(clip_thr, vlm_thr)
        print(f"\n  clip>={clip_thr:.2f} vlm>={vlm_thr}: {len(segs)} segments, {sum(s.duration for s in segs):.0f}s")
        for i, s in enumerate(segs, 1):
            print(f"    #{i:2d}  {mmss(s.start)} -> {mmss(s.end)}  peak vlm {s.peak_vlm:.2f}")

    print("\ninspect the combos above, then either adjust config.toml and rescan, or hand-edit")
    print("report.json settings+segments and re-render with --reuse-scan --overwrite.")


if __name__ == "__main__":
    main()
