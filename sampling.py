"""ffmpeg-based probing and frame sampling."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".ts", ".wmv", ".flv", ".mpg", ".mpeg"}


@dataclass
class VideoInfo:
    duration: float
    width: int
    height: int
    has_audio: bool


@dataclass
class SampledFrame:
    index: int          # 0-based, in time order
    timestamp: float    # seconds into the video
    path: Path


def probe(path: Path) -> VideoInfo:
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {proc.stderr.strip()[:500]}")
    data = json.loads(proc.stdout)

    duration = 0.0
    fmt_dur = data.get("format", {}).get("duration")
    if fmt_dur:
        try:
            duration = float(fmt_dur)
        except ValueError:
            duration = 0.0
    if duration <= 0:
        # some containers omit format duration; fall back to stream durations
        for s in data.get("streams", []):
            d = s.get("duration")
            if d:
                try:
                    duration = max(duration, float(d))
                except ValueError:
                    pass
    if duration <= 0:
        raise RuntimeError(f"Could not determine duration of {path}")

    video_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not video_streams:
        raise RuntimeError(f"No video stream in {path}")
    v = video_streams[0]
    # prefer displayed rotation-corrected dims when present
    width = v.get("width") or 0
    height = v.get("height") or 0
    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    return VideoInfo(duration=duration, width=width, height=height, has_audio=has_audio)


def sample_frames(
    video: Path,
    out_dir: Path,
    fps: float,
    thumb_size: int,
    max_frames: int,
    duration: float,
) -> tuple[list[SampledFrame], float]:
    """Extract ~fps frames/sec into out_dir as JPEGs.

    Returns (frames, effective_fps). effective_fps may be lower than requested
    to stay under max_frames on very long videos.
    """
    if duration * fps > max_frames:
        fps = round(max_frames / duration, 4)
        if fps < 0.05:
            fps = 0.05
        print(f"  [sampling] long video: lowering sample rate to {fps}/s to stay under {max_frames} frames")

    out_dir.mkdir(parents=True, exist_ok=True)
    vf = f"fps={fps},scale={thumb_size}:-2"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", str(video), "-vf", vf, "-q:v", "3",
        str(out_dir / "frame_%05d.jpg"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg sampling failed on {video}: {proc.stderr.strip()[-800:]}")

    files = sorted(out_dir.glob("frame_*.jpg"))
    step = 1.0 / fps
    frames = [SampledFrame(i, i * step, f) for i, f in enumerate(files)]
    return frames, fps
