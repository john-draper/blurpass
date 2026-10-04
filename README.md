# BlurPass

Scan videos for gore or nudity with local models, then render a copy with those
segments blurred. Everything runs on your machine - no cloud calls, no video
leaves the box.

Developed and calibrated on an RTX 5090 (32 GB) + ffmpeg with NVENC; the VLM
judge needs roughly 16 GB of VRAM, and everything scales down (CLIP triage alone
runs on CPU if you accept more false positives).

## How it works

```
video ──ffmpeg──> 2 fps frames ──> CLIP triage (fast, blunt, per category)
                                        │ frames that might match
                                        ▼
                                  VLM confirm (Qwen2.5-VL-7B, judgment,
                                               3-frame context strip)
                                        │ confirmed timestamps
                                        ▼
                              segment builder (merge/pad/smooth)
                                        │
                                        ▼
                  ffmpeg render: gblur enabled only between t1..t2, NVENC, audio copied
```

- **Stage A - CLIP zero-shot** (`ViT-B-32`, hundreds of fps): scores each sampled
  frame against a per-category prompt ensemble. Tuned for recall - it would
  rather send a false alarm to stage B than miss something. Safe prompts
  deliberately include dark cinematic moods; without them, dimly-lit thrillers
  false-flag 80%+ of frames (lesson learned the hard way).
- **Stage B - VLM** (`Qwen2.5-VL-7B-Instruct` default): judges only the
  CLIP-flagged frames, as a 3-frame strip (context beats single frames - a
  red-lit dancer and a real injury look alike alone). A 7B-class judge is the
  practical floor: on hand-labeled movie frames it separated gore (>=0.82)
  from red-light false positives (<=0.22) perfectly, where a 3B could not (an
  FP outranked a TP - unfixable by thresholds).
- **Segments**: confirmed frames become time windows - nearby flags merge,
  windows get +/-1.5s padding (covers the delay between content appearing and
  the next sample), sub-second blips are dropped.
- **Render**: whole-frame gaussian blur during flagged windows only
  (`gblur=...:enable='between(t,a,b)'`), `h264_nvenc` when available, audio
  untouched. Clean videos are stream-copied (no re-encode).

## Setup

Requirements: Python 3.11+, ffmpeg + ffprobe on PATH (NVENC optional but fast),
an NVIDIA GPU with ~16 GB VRAM for the default judge.

```
python -m venv venv
venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
venv\Scripts\pip install open-clip-torch transformers accelerate pillow qwen-vl-utils "huggingface_hub[cli]"
```

(Adjust the torch index for your CUDA version; on CPU-only machines use the
default PyPI torch and expect a slower, CLIP-only experience.)

Model weights auto-download on first run to the HF cache (~16 GB for the 7B
VLM, ~600 MB for CLIP).

## Usage

```
python blurpass.py scan "movie.mkv"                        # scan + HTML report only
python blurpass.py scan "movie.mkv" --no-vlm               # CLIP-only quick pass
python blurpass.py process "movie.mkv"                     # -> movie.filtered.mkv
python blurpass.py process D:\Videos --recursive           # whole library
python blurpass.py process "movie.mkv" --reuse-scan --dry-run
python blurpass.py selftest                                # synthetic end-to-end test
```

- `--strictness low|medium|high` - preset thresholds (lower = more sensitive =
  more blur). Defaults live in `config.toml` under `[presets]`.
- `--fps 3` - sample more frames per second (finer time resolution, slower).
- `--out DIR`, `--suffix .filtered`, `--overwrite`, `--recursive`.
- `--low-vram` - run CLIP on CPU (the 7B VLM still needs its GPU memory).

Reports land in `reports\<video name>\report.html` with a thumbnail grid of
every flagged frame and its CLIP/VLM scores - use it to sanity-check and tune
thresholds before trusting a batch run.

## Categories

`gore` and `nudity` are enabled by default; control per run or in
`config.toml [categories]`:

```
python blurpass.py process "movie.mkv" --categories gore     # gore only
python blurpass.py process "movie.mkv" --categories nudity   # nudity only
```

Each category has its own CLIP prompt ensemble, VLM question, and thresholds
(`[gore]` / `[nudity]` in config.toml). Blur windows from overlapping
categories merge into one window at render time. Adding a new category is a
prompt list + question entry in `detector.py` plus a config section.

**Nudity rule:** by default it blurs nudity *and* suggestive undress -
underwear and partially-undressed scenes count, animated content counts. On a
film with no real nudity but a bathroom/underwear sequence, those windows
score 0.86-0.97 while clothed-scene noise stays below 0.85, which is where
the default threshold sits. If you want only actual nudity, raise the
threshold (underwear scenes may still catch - edit the question in
`detector.py CATEGORY_QUESTIONS` to re-add exclusions).

## Calibrating on your own content

The refinement loop that works: scan a movie, watch the result, label the
segments (true/false positive) in a `test/ground_truth_*.json` file, then
measure a judge against your labels before re-rendering:

```
python test\eval_labeled.py qwen25vl_7b <reports\<video name>>
```

`test/ground_truth_example.json` shows the format and records a real
calibration session's findings, including the score collisions that define the
practical frontier (one film's medium-shot gore and another scene's benign
close-up both scored 0.68 - no threshold separates them).

## Swapping judges

- `qwen25vl_7b` (default) - best calibration observed.
- `qwen25vl_3b` - faster, fits ~8 GB, but expect false positives that outrank
  true positives.
- `shieldgemma2_4b` - Google's purpose-built moderation VLM; gated repo
  (accept license at huggingface.co/google/shieldgemma-2-4b-it, then
  `hf auth login`). Single-frame policy scoring, ~9 GB VRAM.
- `none` - CLIP thresholds decide directly (fast, blunt).

Set in `config.toml` `[vlm] backend`.

## Tuning notes

- **Too much blur** (horror ambience, fake blood, red-lit drama): raise
  thresholds via `--strictness low`, or tune `[presets]`.
- **Missed content**: lower thresholds, and/or raise `--fps`.
- **Content visible before blur starts**: raise `[segments] pad` (default 1.5s)
  or `[sampling] fps`.
- **Blur style**: `[render] filter` = `gblur` | `boxblur` | `pixelize`.
- Very long videos: `[sampling] max_frames` auto-lowers fps past 20000 frames.
- Subtle content (small traces in dark frames) sits near the judge's
  perception limit at 512px thumbs; raising `[sampling] thumb_size` to 768
  helps but slows everything.

## Known limits / v2 ideas

- **Region-only blur** (blur just the offending part, keep the rest visible):
  no off-the-shelf gore bounding-box model exists (NudeNet covers nudity
  classes only; verified). Would require training a small YOLO + per-frame
  tracking. Natural v2.
- Whole-frame blur during dialog in a flagged scene hides faces/subtitles too.
- Subtitle streams are dropped on render (only first video + audio mapped).
- Audio (screams, etc.) is not analyzed.
- Judge score collisions (a benign scene scoring identically to a graphic one)
  are a model-resolution limit, not a threshold problem.

## License

MIT - see LICENSE.
