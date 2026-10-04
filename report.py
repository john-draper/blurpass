"""JSON sidecar + HTML report for a scan."""
from __future__ import annotations

import html
import json
import shutil
import time
from pathlib import Path

from segments import Segment


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{int(m):02d}:{s:05.2f}" if h else f"{int(m):02d}:{s:05.2f}"


def write_json(
    out_path: Path,
    *,
    video: str,
    info: dict,
    frames: list[dict],
    segments: list[dict],
    settings: dict,
) -> None:
    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "video": video,
        "info": info,
        "settings": settings,
        "frames": frames,
        "segments": segments,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _thumb_grid(seg_frames: list[dict], thumbs_dir_rel: str, max_thumbs: int) -> str:
    shown = seg_frames[:max_thumbs]
    cells = []
    for fr in shown:
        name = Path(fr["frame"]).name
        clip = fr["clip_score"]
        vlm = fr.get("vlm_score")
        vlm_txt = f" | vlm {vlm:.2f}" if vlm is not None else ""
        cells.append(
            f'<figure><img loading="lazy" src="{thumbs_dir_rel}/{name}" alt="t={fr["timestamp"]:.1f}s">'
            f"<figcaption>{_fmt_ts(fr['timestamp'])} | clip {clip:.2f}{vlm_txt}</figcaption></figure>"
        )
    extra = len(seg_frames) - len(shown)
    if extra > 0:
        cells.append(f'<div class="more">+{extra} more frames</div>')
    return "".join(cells)


def write_html(
    out_path: Path,
    *,
    video: str,
    info: dict,
    frames: list[dict],
    segments: list[dict],
    settings: dict,
    max_thumbs_per_segment: int,
    keep_frames_dir: Path | None = None,
) -> None:
    flagged_count = sum(1 for f in frames if f.get("vlm_score") is not None or f.get("flagged", False))
    total_blur = sum(s["end"] - s["start"] for s in segments)

    seg_rows = []
    for i, s in enumerate(segments, 1):
        seg_frames = s.get("frames", [])
        thumbs = _thumb_grid(seg_frames, "thumbs", max_thumbs_per_segment)
        seg_rows.append(f"""
      <div class="segment">
        <div class="seghead">
          <span class="num">#{i}</span>
          <span class="time">{_fmt_ts(s['start'])} &rarr; {_fmt_ts(s['end'])}</span>
          <span class="dur">{s['end'] - s['start']:.1f}s</span>
          <span class="peak">peak clip {s.get('peak_clip', 0):.2f}{f" | peak vlm {s['peak_vlm']:.2f}" if s.get('peak_vlm') is not None else ''}</span>
        </div>
        <div class="thumbs">{thumbs}</div>
      </div>""")

    st = settings
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>BlurPass report - {html.escape(Path(video).name)}</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #14161a; color: #d7dae0;
         margin: 0; padding: 24px; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .sub {{ color: #7d8590; font-size: 13px; margin-bottom: 18px; }}
  .summary {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px; }}
  .card {{ background: #1d2026; border: 1px solid #2a2e36; border-radius: 8px;
           padding: 10px 14px; min-width: 110px; }}
  .card .k {{ color: #7d8590; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; }}
  .card .v {{ font-size: 18px; font-weight: 600; margin-top: 2px; }}
  .segment {{ background: #1d2026; border: 1px solid #2a2e36; border-radius: 8px;
              padding: 12px 14px; margin-bottom: 14px; }}
  .seghead {{ display: flex; gap: 14px; align-items: baseline; font-size: 14px; margin-bottom: 8px; }}
  .num {{ color: #f0a35e; font-weight: 700; }}
  .time {{ font-family: Consolas, monospace; font-size: 15px; }}
  .dur {{ color: #7d8590; }}
  .peak {{ color: #7d8590; margin-left: auto; font-size: 12px; }}
  .thumbs {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  figure {{ margin: 0; width: 150px; }}
  figure img {{ width: 150px; border-radius: 4px; display: block; border: 1px solid #2a2e36; }}
  figcaption {{ font-size: 11px; color: #8b929c; font-family: Consolas, monospace; margin-top: 3px; }}
  .more {{ align-self: center; color: #7d8590; font-size: 13px; }}
  .none {{ color: #6fbf73; font-size: 15px; }}
</style></head>
<body>
  <h1>{html.escape(Path(video).name)}</h1>
  <div class="sub">BlurPass report &middot; {info.get('duration', 0):.0f}s video &middot; scanned {time.strftime('%Y-%m-%d %H:%M')}</div>
  <div class="summary">
    <div class="card"><div class="k">Frames scanned</div><div class="v">{len(frames)}</div></div>
    <div class="card"><div class="k">Flagged frames</div><div class="v">{flagged_count}</div></div>
    <div class="card"><div class="k">Segments</div><div class="v">{len(segments)}</div></div>
    <div class="card"><div class="k">Total blur</div><div class="v">{total_blur:.0f}s</div></div>
    <div class="card"><div class="k">Sampling</div><div class="v">{st.get('sampling_fps', '?')}/s</div></div>
    <div class="card"><div class="k">Clip thr</div><div class="v">{st.get('clip_threshold', '?')}</div></div>
    <div class="card"><div class="k">VLM</div><div class="v">{st.get('vlm_backend', 'none')} @{st.get('vlm_threshold', '-')}</div></div>
  </div>
  {('<div class="none">No gore detected - nothing would be blurred.</div>' if not segments else ''.join(seg_rows))}
</body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    # thumbnails referenced by the report
    thumbs_dir = out_path.parent / "thumbs"
    thumbs_dir.mkdir(exist_ok=True)
    wanted = set()
    for s in segments:
        for fr in s.get("frames", [])[:max_thumbs_per_segment]:
            wanted.add(Path(fr["frame"]).name)
    src_dir = keep_frames_dir
    if src_dir and src_dir.is_dir():
        for name in wanted:
            src = src_dir / name
            if src.exists():
                shutil.copy2(src, thumbs_dir / name)
