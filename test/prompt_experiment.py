"""Calibrate the CLIP prompt ensemble against a folder of sampled frames.

Usage: prompt_experiment.py <frames_dir> [positive_control_dir]

Prints how often each ensemble scores frames over various thresholds, so you
can compare prompt sets before committing to a scan. Include a folder of
known-matching frames (e.g. staged/horror-makeup photos for the gore
category) as the positive control: it must stay at ~1.0 while the movie
sample's flag rate drops.
"""
import sys
from pathlib import Path

import torch
from PIL import Image
import open_clip

GORE_OLD = [
    "graphic violence and gore",
    "blood splatter and open wounds",
    "a mutilated or dismembered dead body",
    "a gruesome bloody injury",
    "a gory horror scene with excessive blood",
    "severe injury with visible blood and flesh",
]
SAFE_OLD = [
    "a normal everyday scene",
    "people talking in a room",
    "a landscape or nature scenery",
    "a building or a street",
    "food, objects, or products",
    "people playing sports or a video game",
]
GORE_NEW = [
    "graphic violence with visible blood and gore",
    "a person with an open bleeding wound",
    "a mutilated or dismembered dead body",
    "blood splattered on a body or surface",
    "a gruesome bloody injury with exposed flesh",
    "a corpse covered in blood",
]
SAFE_NEW = SAFE_OLD + [
    "a dark moody scene from a drama film",
    "a suspenseful scene with a shadowy figure",
    "a dimly lit room at night",
    "a cold blue-toned cinematic scene",
    "a close-up of a person's face in dim light",
    "an atmospheric scene without any blood",
]


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: prompt_experiment.py <frames_dir> [positive_control_dir]")
    frames = sorted(Path(sys.argv[1]).glob("*.jpg"))[::8]
    control = sorted(Path(sys.argv[2]).glob("*.jpg")) if len(sys.argv) > 2 else []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    model.eval()
    if device == "cuda":
        model.half()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    logit_scale = model.logit_scale.exp().item()

    def embed_class(prompts):
        toks = tokenizer(prompts).to(device)
        with torch.no_grad():
            f = model.encode_text(toks).float()
            f = f / f.norm(dim=-1, keepdim=True)
        e = f.mean(dim=0)
        return e / e.norm()

    def score(paths, gore_p, safe_p, batch=128):
        text = torch.stack([embed_class(gore_p), embed_class(safe_p)])
        text = (text.half() if device == "cuda" else text).to(device)
        out = []
        for i in range(0, len(paths), batch):
            imgs = []
            for p in paths[i:i + batch]:
                with Image.open(p) as im:
                    imgs.append(preprocess(im.convert("RGB")))
            x = torch.stack(imgs)
            x = x.half().to(device) if device == "cuda" else x.to(device)
            with torch.no_grad():
                f = model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
                out += torch.softmax(logit_scale * (f.float() @ text.float().T), -1)[:, 0].cpu().tolist()
        return out

    def stats(name, scores):
        n = len(scores)
        over = {t: sum(1 for s in scores if s >= t) for t in (0.45, 0.8, 0.9, 0.95)}
        print(f"{name:28s} n={n:4d}  " + "  ".join(f">={t}: {c:4d} ({c/n*100:4.1f}%)" for t, c in over.items()))

    print("== target frames (should mostly NOT match) ==")
    stats("baseline ensemble", score(frames, GORE_OLD, SAFE_OLD))
    stats("cinematic-safe ensemble", score(frames, GORE_NEW, SAFE_NEW))
    if control:
        print("\n== positive control (should stay ~100%) ==")
        stats("baseline ensemble", score(control, GORE_OLD, SAFE_OLD))
        stats("cinematic-safe ensemble", score(control, GORE_NEW, SAFE_NEW))


if __name__ == "__main__":
    main()
