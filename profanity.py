"""Audio profanity muting: faster-whisper transcription -> word match -> mute.

Video is stream-copied (never re-encoded); only the audio track is touched,
with volume=0 windows over each matched word (slightly padded so leading
consonants and trailing sibilance die too).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

# Default vocabulary: owner's curated map from the bayo subtitle projects,
# plus common inflections/heard-variants. Matched on normalized whole tokens.
DEFAULT_WORDS = {
    "hell", "hells", "hell's",
    "damn", "damns", "damned",
    "goddamn", "goddamned", "goddammit", "goddamnit", "god damn",
    "shit", "shits", "shitty", "shite",
    "fuck", "fucks", "fucked", "fucking", "fuckin", "fuckin'", "fucker", "fuckers",
    "bitch", "bitches", "bitchy",
    "bastard", "bastards", "bastardized",
    "ass", "asses", "asshole", "assholes", "assholes'",
    "badass", "badasses",
    "piss", "pisses", "pissed", "pissing",
    "crap", "craps", "crappy",
    "bloody",
    "dumbass", "dumbasses", "dickhead", "dickheads",
}


def _normalize(token: str) -> str:
    return token.strip().strip(".,!?\"'()[]…—-").lower().rstrip("'")


def load_words(cfg: dict) -> set[str]:
    p = cfg.get("profanity", {}).get("words_file")
    if p:
        data = json.load(open(p, encoding="utf-8"))
        words = set(data.get("map", data if isinstance(data, list) else []))
        print(f"[profanity] loaded {len(words)} words from {p}")
        return words
    return set(DEFAULT_WORDS)


def _prep_gpu_dlls() -> None:
    """CTranslate2 needs cuDNN DLLs; torch's bundled copies usually satisfy it."""
    try:
        import torch
        dll_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(dll_dir):
            os.add_dll_directory(dll_dir)
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def extract_wav(video: Path, out_wav: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(out_wav)],
        check=True, capture_output=True,
    )


def _resolve_model(model_size: str) -> str:
    """faster-whisper 1.2.1's downloader is incompatible with huggingface_hub
    1.x; fetch the model ourselves and hand it a local path."""
    p = Path(model_size)
    if p.is_dir():
        return str(p)
    from huggingface_hub import snapshot_download
    repo = f"Systran/faster-whisper-{model_size}"
    local = snapshot_download(repo)
    print(f"[profanity] model cached at {local}")
    return local


def _load_wav_f32(wav: Path):
    """16-bit mono PCM wav -> float32 numpy array. Avoids PyAV entirely:
    faster-whisper 1.2.1's av.open() kwargs don't match PyAV 19's API."""
    import wave
    import numpy as np
    with wave.open(str(wav), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, "expected 16-bit mono wav"
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0


def transcribe(wav: Path, model_size: str, device: str) -> tuple[list[tuple[str, float, float]], str]:
    """Returns ([(token, start_s, end_s)], language)."""
    from faster_whisper import WhisperModel

    model_path = _resolve_model(model_size)
    audio = _load_wav_f32(wav)
    compute = None
    if device == "auto":
        _prep_gpu_dlls()
        try:
            model = WhisperModel(model_path, device="cuda", compute_type="float16")
            compute = "cuda/float16"
        except Exception as e:
            print(f"[profanity] GPU transcription unavailable ({str(e)[:120]}); using CPU")
            model = WhisperModel(model_path, device="cpu", compute_type="int8")
            compute = "cpu/int8"
    else:
        model = WhisperModel(model_path, device=device,
                             compute_type="float16" if device == "cuda" else "int8")
        compute = device
    print(f"[profanity] whisper {model_size} on {compute}, transcribing...")

    segments, info = model.transcribe(audio, word_timestamps=True, vad_filter=True)
    words: list[tuple[str, float, float]] = []
    for seg in segments:
        for w in seg.words or []:
            words.append((w.word.strip(), float(w.start), float(w.end)))
    print(f"[profanity] language={info.language}, {len(words)} words transcribed")
    return words, info.language


def profanity_windows(
    words: list[tuple[str, float, float]],
    wordlist: set[str],
    *,
    pad_in: float,
    pad_out: float,
    merge_gap: float,
    duration: float,
) -> tuple[list[tuple[float, float]], list[tuple[str, float, float]]]:
    hits = [(tok, s, e) for tok, s, e in words if _normalize(tok) in wordlist]
    windows: list[tuple[float, float]] = []
    for _tok, s, e in hits:
        s2, e2 = max(0.0, s - pad_in), min(duration, e + pad_out)
        if windows and s2 - windows[-1][1] <= merge_gap:
            windows[-1] = (windows[-1][0], e2)
        else:
            windows.append((s2, e2))
    return windows, hits


def render_mute(video: Path, windows: list[tuple[float, float]], out_path: Path) -> str:
    """Mute the given audio windows; video stream is copied untouched."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not windows:
        import shutil
        shutil.copy2(video, out_path)
        return "copy"

    parts = [
        f"volume=volume=0:enable='between(t,{s:.3f},{e:.3f})'"
        for s, e in windows
    ]
    af = ",".join(parts)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(video),
        "-map", "0:v:0", "-map", "0:a?",
        "-af", af,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
    ]
    if out_path.suffix.lower() in (".mp4", ".m4v", ".mov"):
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(out_path))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg mute render failed: {proc.stderr.strip()[-800:]}")
    return "muted"


def audio_duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=duration", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0
