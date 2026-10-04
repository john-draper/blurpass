"""Measure a VLM judge against hand-labeled segments (no rescan needed).

Usage: eval_labeled.py <backend> <report_dir>
  e.g. python test/eval_labeled.py qwen25vl_7b reports/MyMovie

Reads the frames/ and report.json from a prior scan, asks the judge for a
yes/no probability across labeled time regions, and reports whether any
threshold separates the true positives from the false positives.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import detector
import tomllib


def frange(a, b, step):
    x = a
    while x <= b + 1e-9:
        yield x
        x += step


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "qwen25vl_7b"
    report_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if report_dir is None or not (report_dir / "report.json").exists():
        sys.exit("usage: eval_labeled.py <backend> <report_dir>  (a prior scan's reports/<video> dir)")

    labels_file = Path(__file__).parent / "ground_truth_example.json"
    gt = json.load(open(labels_file))
    video_name = Path(gt["video"]).stem

    # (label, t_start, t_end, expected) - "?" = unlabeled, shown for context
    regions = [
        (l["note"][:28], l["start"], l["end"],
         {"true_positive": "YES", "false_positive": "NO"}.get(l["verdict"], "?"))
        for l in gt["labels"]
    ]

    report = json.load(open(report_dir / "report.json"))
    fps = report["settings"]["sampling_fps"]
    frames = sorted((report_dir / "frames").glob("frame_*.jpg"))
    if Path(gt["video"]).stem != report_dir.name:
        print(f"note: ground truth is for '{gt['video']}', report dir is '{report_dir.name}' - "
              f"timestamps may not line up with a different scan")

    cfg = tomllib.load(open(Path(__file__).parent.parent / "config.toml", "rb"))
    cfg["vlm"]["backend"] = backend
    vlm = detector.load_vlm(cfg)
    print(f"\n=== {backend} on {len(regions)} labeled regions ===")

    results = []
    for label, t0, t1, expected in regions:
        idx = [round(t * fps) for t in frange(t0, t1, 0.5)]
        groups = [
            [frames[j] for j in range(max(0, i - 1), min(len(frames), i + 2))]
            for i in idx
        ]
        probs = vlm.confirm(groups)
        peak = max(probs)
        results.append((label, expected, peak))
        verdict = "OK " if (expected == "?" or (expected == "YES") == (peak >= 0.5)) else "MISS"
        print(f"  [{verdict}] {label:28s} expected {expected:3s}  peak P(yes)={peak:.2f}")

    tp = [p for _, e, p in results if e == "YES"]
    fp = [p for _, e, p in results if e == "NO"]
    print(f"\nTP peaks: {sorted(round(p, 2) for p in tp)}")
    print(f"FP peaks: {sorted(round(p, 2) for p in fp)}")
    best_acc, best_thr = 0, 0
    for thr in [i / 100 for i in range(20, 96)]:
        acc = sum((p >= thr) == (e == "YES") for _, e, p in results if e != "?") / max(len(tp) + len(fp), 1)
        if acc >= best_acc:
            best_acc, best_thr = acc, thr
    print(f"best threshold for this judge: {best_thr} (accuracy {best_acc:.0%})")


if __name__ == "__main__":
    main()
