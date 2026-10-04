"""ffmpeg rendering: whole-frame blur during flagged windows only."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from segments import Segment


def filter_supports_timeline(name: str) -> bool:
    """True if ffmpeg's `<name>` filter carries the timeline (T) flag."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True
    )
    for line in proc.stdout.splitlines():
        # e.g. " ... gblur            V->V Apply Gaussian Blur filter."
        parts = line.split()
        if len(parts) >= 2 and parts[1] == name:
            return "T" in parts[0]
    return False


def check_nvenc() -> bool:
    """Probe whether h264_nvenc actually works on this machine."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=0.5",
                "-frames:v", "5", "-pix_fmt", "yuv420p",
                "-c:v", "h264_nvenc", "-f", "null", "-",
            ],
            check=True, capture_output=True, timeout=30,
        )
        return True
    except Exception:
        return False


def _enable(start: float, end: float) -> str:
    return f"enable='between(t,{start:.3f},{end:.3f})'"


def build_filtergraph(segments: list[Segment], filter_name: str, sigma: float) -> str:
    if not filter_supports_timeline(filter_name):
        raise RuntimeError(
            f"ffmpeg filter '{filter_name}' not found or lacks timeline support; "
            f"pick gblur/boxblur in config.toml [render].filter"
        )
    parts: list[str] = []
    for seg in segments:
        en = _enable(seg.start, seg.end)
        if filter_name == "gblur":
            parts.append(f"gblur=sigma={sigma:.0f}:{en}")
        elif filter_name == "boxblur":
            r = max(2, int(sigma / 1.5))
            parts.append(
                f"boxblur=luma_radius={r}:luma_power=2:"
                f"chroma_radius={max(1, r // 2)}:chroma_power=1:{en}"
            )
        elif filter_name == "pixelize":
            bw = max(4, int(sigma / 3))
            parts.append(f"pixelize=width={bw}:height={bw}:{en}")
        else:
            raise RuntimeError(f"unknown render filter: {filter_name!r}")
    return ",".join(parts)


def resolve_encoder(requested: str) -> str:
    if requested != "auto":
        return requested
    return "h264_nvenc" if check_nvenc() else "libx264"


def render(
    video: Path,
    segments: list[Segment],
    out_path: Path,
    filter_name: str,
    sigma: float,
    encoder_setting: str,
    cq: int,
    has_audio: bool,
) -> str:
    """Render a copy with blur applied during `segments`. Returns encoder used.

    With no flagged segments the file is stream-copied (no re-encode, instant).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not segments:
        shutil.copy2(video, out_path)
        return "copy"

    vf = build_filtergraph(segments, filter_name, sigma)
    encoder = resolve_encoder(encoder_setting)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(video)]
    cmd += ["-vf", vf, "-map", "0:v:0"]
    if has_audio:
        cmd += ["-map", "0:a?", "-c:a", "copy"]

    if encoder == "h264_nvenc":
        cmd += ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(cq), "-b:v", "0"]
    elif encoder == "libx264":
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(cq)]
    else:
        raise RuntimeError(f"unknown encoder: {encoder!r}")

    if out_path.suffix.lower() in (".mp4", ".m4v", ".mov"):
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(out_path))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg render failed: {proc.stderr.strip()[-1200:]}")
    return encoder
