"""JSON sidecar + HTML report for a scan."""
from __future__ import annotations

import html
import json
import shutil
import time
from pathlib import Path


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


def _thumb_grid(seg_frames: list[dict], category: str, max_thumbs: int) -> str:
    shown = seg_frames[:max_thumbs]
    cells = []
    for fr in shown:
        name = Path(fr["frame"]).name
        cat = fr.get(category, {})
        clip = cat.get("clip", 0.0)
        vlm = cat.get("vlm")
        vlm_txt = f" | vlm {vlm:.2f}" if vlm is not None else ""
        cells.append(
            f'<figure><img loading="lazy" src="thumbs/{name}" alt="t={fr["timestamp"]:.1f}s">'
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
    categories = list(settings.get("categories", ["gore"]))
    cat_segs = {
        cat: [s for s in segments if s.get("category", "gore") == cat]
        for cat in categories
    }
    total_blur = sum(s["end"] - s["start"] for s in segments)

    cards = "".join(
        f'<div class="card"><div class="k">{cat} segments</div>'
        f'<div class="v">{len(cat_segs[cat])}</div></div>'
        f'<div class="card"><div class="k">{cat} blur</div>'
        f'<div class="v">{sum(s["end"] - s["start"] for s in cat_segs[cat]):.0f}s</div></div>'
        for cat in categories
    )

    seg_blocks = []
    for i, s in enumerate(segments, 1):
        cat = s.get("category", "gore")
        seg_frames = s.get("frames", [])
        thumbs = _thumb_grid(seg_frames, cat, max_thumbs_per_segment)
        peak_vlm = s.get("peak_vlm")
        vlm_txt = f" | peak vlm {peak_vlm:.2f}" if peak_vlm is not None else ""
        seg_blocks.append(f"""
      <div class="segment">
        <div class="seghead">
          <span class="num">#{i}</span>
          <span class="cat cat-{cat}">{cat}</span>
          <span class="time">{_fmt_ts(s['start'])} &rarr; {_fmt_ts(s['end'])}</span>
          <span class="dur">{s['end'] - s['start']:.1f}s</span>
          <span class="peak">peak clip {s.get('peak_clip', 0):.2f}{vlm_txt}</span>
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
  .cat {{ font-size: 11px; text-transform: uppercase; letter-spacing: .5px; border-radius: 4px;
          padding: 2px 8px; }}
  .cat-gore {{ background: #4a1f1f; color: #e88f8f; }}
  .cat-nudity {{ background: #2e3a5c; color: #9fb4e8; }}
  .cat-merged {{ background: #3d3450; color: #c3a8e8; }}
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
    {cards}
    <div class="card"><div class="k">Total blur</div><div class="v">{total_blur:.0f}s</div></div>
    <div class="card"><div class="k">Sampling</div><div class="v">{st.get('sampling_fps', '?')}/s</div></div>
    <div class="card"><div class="k">VLM</div><div class="v">{st.get('vlm_backend', 'none')}</div></div>
  </div>
  {('<div class="none">Nothing flagged - no segments would be blurred.</div>' if not segments else ''.join(seg_blocks))}
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
