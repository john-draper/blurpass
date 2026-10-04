"""Two-stage detection: CLIP zero-shot triage + optional VLM confirmation."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from PIL import Image

# Prompt ensemble for zero-shot triage. Gore prompts describe CONTENT (blood,
# wounds, bodies); safe prompts deliberately include dark/cinematic moods so a
# moody thriller frame has somewhere to land besides "gore" (with only
# bright/everyday safe prompts, dimly-lit dramas false-flag en masse).
GORE_PROMPTS = [
    "graphic violence with visible blood and gore",
    "a person with an open bleeding wound",
    "a mutilated or dismembered dead body",
    "blood splattered on a body or surface",
    "a gruesome bloody injury with exposed flesh",
    "a corpse covered in blood",
]
SAFE_PROMPTS = [
    "a normal everyday scene",
    "people talking in a room",
    "a landscape or nature scenery",
    "a building or a street",
    "food, objects, or products",
    "people playing sports or a video game",
    "a dark moody scene from a drama film",
    "a suspenseful scene with a shadowy figure",
    "a dimly lit room at night",
    "a cold blue-toned cinematic scene",
    "a close-up of a person's face in dim light",
    "an atmospheric scene without any blood",
]

def vlm_question(n_frames: int) -> str:
    plural = "frames" if n_frames > 1 else "frame"
    return (
        f"You are given {n_frames} consecutive {plural} from a video, in chronological order. "
        "Judge the scene they show. Does the video contain graphic gore: visible blood, "
        "open wounds, dead or mutilated bodies, or a hanging? "
        "Red or dramatic lighting, suspense, dancing, sleeping, and emotional distress "
        "do NOT count. Answer yes only if actual graphic gore is visible. "
        "Answer with exactly one word: yes or no."
    )


# Kept for backward reference / docs.
VLM_QUESTION = vlm_question(1)

# ShieldGemma-2 "Violence & Gore" policy, verbatim from the model card.
SHIELDGEMMA_VG_POLICY = (
    "The image shall not contain shocking, sensational, or gratuitous violence, "
    "such as excessive blood, severe injuries, or dismembered bodies."
)


def pick_device(requested: str, min_free_gb: float) -> str:
    if requested not in ("auto",):
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    free, _total = torch.cuda.mem_get_info()
    return "cuda" if free / 1e9 >= min_free_gb else "cpu"


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


class ClipTriage:
    """Stage A: zero-shot gore/safe classification, hundreds of fps on GPU."""

    def __init__(self, model: str, pretrained: str, device: str = "auto"):
        import open_clip

        self.device = pick_device(device, min_free_gb=1.5)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.use_half = self.device == "cuda"
        if self.use_half:
            self.model.half()

        tokenizer = open_clip.get_tokenizer(model)

        def class_embedding(prompts: list[str]) -> torch.Tensor:
            toks = tokenizer(prompts).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(toks).float()
                feats = feats / feats.norm(dim=-1, keepdim=True)
            emb = feats.mean(dim=0)
            return emb / emb.norm()

        gore = class_embedding(GORE_PROMPTS)
        safe = class_embedding(SAFE_PROMPTS)
        text = torch.stack([gore, safe]).to(self.device)
        self.text_features = text.half() if self.use_half else text
        self.logit_scale = self.model.logit_scale.exp().item()

    @torch.no_grad()
    def score(self, paths: list[Path], batch_size: int = 64) -> list[float]:
        """P(gore) in [0,1] for each image path, in order."""
        scores: list[float] = []
        for chunk in _chunks(paths, batch_size):
            imgs = []
            for p in chunk:
                with Image.open(p) as im:
                    imgs.append(self.preprocess(im.convert("RGB")))
            batch = torch.stack(imgs).to(self.device)
            if self.use_half:
                batch = batch.half()
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = self.logit_scale * (feats.float() @ self.text_features.float().T)
            probs = torch.softmax(logits, dim=-1)[:, 0]
            scores.extend(probs.cpu().tolist())
        return scores


class _VlmBase:
    """Shared yes/no-with-probability machinery for VLM backends."""

    # subclasses set: model_id, answer_yes_words, answer_no_words

    def _prob_from_first_token(self, scores) -> float:
        """P(yes) from the first generated token, renormalized over yes/no variants."""
        logits = scores[0][0].float()
        yes_ids = [self.tokenizer.encode(w, add_special_tokens=False)[0] for w in self.yes_words]
        no_ids = [self.tokenizer.encode(w, add_special_tokens=False)[0] for w in self.no_words]
        yes_lp = torch.logsumexp(logits[yes_ids], dim=0)
        no_lp = torch.logsumexp(logits[no_ids], dim=0)
        p_yes = torch.sigmoid(yes_lp - no_lp).item()
        return p_yes


class Qwen25Vl(_VlmBase):
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    yes_words = ["yes", "Yes", "YES"]
    no_words = ["no", "No", "NO"]

    def __init__(self, device: str = "auto", dtype: str = "bfloat16", model_id: str | None = None):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if model_id:
            self.model_id = model_id
        self.device = pick_device(device, min_free_gb=8.0)
        if self.device == "cpu":
            print("  [vlm] WARNING: running the VLM on CPU will be very slow; consider --no-vlm")
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch_dtype, device_map=self.device
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            size={"shortest_edge": 256 * 28 * 28, "longest_edge": 1280 * 28 * 28},
        )
        self.tokenizer = self.processor.tokenizer

    @torch.no_grad()
    def confirm(self, frame_groups: list[list[Path]]) -> list[float]:
        """Each group is a short run of consecutive frames (center = judged frame)."""
        from qwen_vl_utils import process_vision_info

        out: list[float] = []
        for group in frame_groups:
            content: list[dict] = [
                {"type": "image", "image": str(p)} for p in group
            ]
            content.append({"type": "text", "text": vlm_question(len(group))})
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(self.device)
            gen = self.model.generate(
                **inputs, max_new_tokens=1, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
            )
            out.append(self._prob_from_first_token(gen.scores))
        return out


class ShieldGemma2(_VlmBase):
    """Experimental: requires one-time acceptance of the gated repo on huggingface.co."""

    model_id = "google/shieldgemma-2-4b-it"
    yes_words = ["Yes", "yes"]
    no_words = ["No", "no"]

    def __init__(self, device: str = "auto", dtype: str = "bfloat16", model_id: str | None = None):
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        if model_id:
            self.model_id = model_id
        self.device = pick_device(device, min_free_gb=9.0)
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch_dtype, device_map=self.device
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = self.processor.tokenizer

    @torch.no_grad()
    def confirm(self, frame_groups: list[list[Path]]) -> list[float]:
        """Policy model: judges the center frame of each group."""
        out: list[float] = []
        for group in frame_groups:
            center = group[len(group) // 2]
            prompt = (
                "<start_of_turn>user\n"
                f"{SHIELDGEMMA_VG_POLICY}\n"
                "Does the content in the image violate the above rule?"
                "<end_of_turn>\n"
                "<start_of_turn>model\n"
            )
            from PIL import Image as PILImage

            with PILImage.open(center) as im:
                img = im.convert("RGB")
            inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(self.device)
            gen = self.model.generate(
                **inputs, max_new_tokens=1, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
            )
            out.append(self._prob_from_first_token(gen.scores))
        return out


def load_clip(cfg: dict) -> ClipTriage:
    c = cfg["clip"]
    return ClipTriage(model=c["model"], pretrained=c["pretrained"], device=c["device"])


def load_vlm(cfg: dict) -> _VlmBase | None:
    v = cfg["vlm"]
    backend = v.get("backend", "qwen25vl_3b")
    if backend in ("none", ""):
        return None
    if backend == "qwen25vl_3b":
        return Qwen25Vl(device=v["device"], dtype=v.get("dtype", "bfloat16"))
    if backend == "qwen25vl_7b":
        return Qwen25Vl(
            device=v["device"], dtype=v.get("dtype", "bfloat16"),
            model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        )
    if backend == "shieldgemma2_4b":
        return ShieldGemma2(device=v["device"], dtype=v.get("dtype", "bfloat16"))
    raise ValueError(
        f"unknown vlm backend: {backend!r} "
        f"(expected qwen25vl_3b | qwen25vl_7b | shieldgemma2_4b | none)"
    )
