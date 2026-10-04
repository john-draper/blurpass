"""Turn per-frame verdicts into blur time windows."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FrameVerdict:
    timestamp: float
    clip_score: float
    vlm_score: float | None = None

    @property
    def confirmed(self) -> bool | None:
        """None when no VLM looked at this frame."""
        return None if self.vlm_score is None else self.vlm_score


@dataclass
class Segment:
    start: float
    end: float
    category: str = "gore"
    frames: list[FrameVerdict] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def peak_clip(self) -> float:
        return max((f.clip_score for f in self.frames), default=0.0)

    @property
    def peak_vlm(self) -> float | None:
        vals = [f.vlm_score for f in self.frames if f.vlm_score is not None]
        return max(vals) if vals else None


def _runs_of(flagged_ts: list[float], step: float) -> list[tuple[float, float]]:
    """Group flagged timestamps into (start, end) runs.

    A frame at time t is treated as covering [t, t + step). Consecutive flagged
    frames within ~2 steps of each other are considered one run.
    """
    runs: list[tuple[float, float]] = []
    for t in flagged_ts:
        if runs and t - runs[-1][1] <= step * 1.9:
            runs[-1] = (runs[-1][0], t + step)
        else:
            runs.append((t, t + step))
    return runs


def build_segments(
    verdicts: list[FrameVerdict],
    *,
    step: float,                # 1 / sampling fps
    duration: float,
    clip_threshold: float,
    vlm_threshold: float,
    merge_gap: float,
    pad: float,
    min_duration: float,
    vlm_active: bool = False,   # True when a VLM ran: unchecked frames are NOT flagged
) -> list[Segment]:
    """A frame is confirmed when the VLM scored it >= vlm_threshold. With a VLM
    in the loop, frames the VLM never saw are treated as clean (the VLM budget
    goes to the most CLIP-suspicious frames first, so unseen means low-suspicion).
    Only in pure CLIP mode (--no-vlm) does the CLIP threshold decide."""
    flagged = [
        v for v in verdicts
        if (v.vlm_score >= vlm_threshold if v.vlm_score is not None
            else (not vlm_active and v.clip_score >= clip_threshold))
    ]
    if not flagged:
        return []

    runs = _runs_of([v.timestamp for v in flagged], step)

    # merge runs separated by less than merge_gap
    merged: list[tuple[float, float]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # pad, clamp to video bounds, drop short segments
    segments: list[Segment] = []
    for s, e in merged:
        s2 = max(0.0, s - pad)
        e2 = min(duration, e + pad)
        if e2 - s2 < min_duration:
            continue
        members = [v for v in flagged if s <= v.timestamp < e]
        segments.append(Segment(start=round(s2, 3), end=round(e2, 3), frames=members))
    return segments


def merge_segments(segments: list[Segment]) -> list[Segment]:
    """Union overlapping/touching windows across categories (blur is whole-frame,
    so two categories flagging the same moment need one window, not two stacked
    blurs). Output is sorted; category becomes 'merged' when windows overlap."""
    out: list[Segment] = []
    for s in sorted(segments, key=lambda x: x.start):
        if out and s.start <= out[-1].end:
            prev = out[-1]
            cat = prev.category if prev.category == s.category else "merged"
            out[-1] = Segment(prev.start, max(prev.end, s.end), cat,
                              prev.frames + s.frames)
        else:
            out.append(Segment(s.start, s.end, s.category, s.frames))
    return out
