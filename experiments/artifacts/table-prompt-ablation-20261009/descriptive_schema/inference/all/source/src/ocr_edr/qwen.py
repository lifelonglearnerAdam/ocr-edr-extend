"""Lazy, locally cached Qwen2-VL inference for the diagnostic pilot."""

from __future__ import annotations

import time
from pathlib import Path

from .formula_pilot import make_prompt, sha256_file
from .loop import digest

EVIDENCE_MODES = (
    "source_only",
    "source_first",
    "source_last",
    "labeled_source_first",
    "labeled_source_last",
    "duplicate_source",
)


def evidence_message(source: Path, prediction: str, rendered: Path | None, mode: str):
    """Construct image roles and their processor order together, without labels from GT."""
    if mode not in EVIDENCE_MODES:
        raise ValueError("Unknown evidence mode")
    if mode not in {"source_only", "duplicate_source"} and rendered is None:
        raise ValueError("Candidate evidence mode requires its rendered image")
    if mode == "source_only":
        paths, prompt = [source], make_prompt(prediction, False)
    elif mode == "duplicate_source":
        paths = [source, source]
        prompt = make_prompt(prediction, False).replace(
            "Image 1 is the source formula.",
            "Image 1 is the source formula. Image 2 is an identical copy of the source formula.",
        )
    else:
        source_last = mode in {"source_last", "labeled_source_last"}
        paths = [rendered, source] if source_last else [source, rendered]
        prompt = make_prompt(prediction, True)
        if source_last:
            prompt = prompt.replace(
                "Image 1 is the source formula. Image 2 is a rendering of the current OCR prediction.",
                "Image 1 is a rendering of the current OCR prediction. Image 2 is the source formula.",
            )
    content = []
    if mode.startswith("labeled_"):
        roles = ["SOURCE FORMULA", "CURRENT OCR RENDERING"]
        if mode == "labeled_source_last":
            roles.reverse()
        for role in roles:
            content.extend([{"type": "text", "text": role + ":"}, {"type": "image"}])
    else:
        content.extend({"type": "image"} for _ in paths)
    content.append({"type": "text", "text": prompt})
    return paths, [{"role": "user", "content": content}], prompt


class QwenFormulaProposer:
    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "cuda:0",
        max_new_tokens: int = 192,
        min_pixels: int = 128 * 28 * 28,
        max_pixels: int = 256 * 28 * 28,
        adapter_path: Path | None = None,
        precision_profile: str = "bf16_lora",
    ):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        if min_pixels <= 0 or max_pixels < min_pixels:
            raise ValueError("Positive ordered image pixel limits required")
        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.precision_metadata = {}
        self.processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=True,
            use_fast=False,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        if precision_profile == "bf16_lora":
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                str(model_path),
                local_files_only=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).to(device)
        elif precision_profile == "nf4_lora_8gb" and device == "cuda:0":
            from .inference_precision import load_nf4_inference_base

            self.model, self.precision_metadata = load_nf4_inference_base(model_path)
        else:
            raise ValueError("Unknown inference precision profile or incompatible device")
        if adapter_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model, str(adapter_path), is_trainable=False, local_files_only=True
            )
        self.model.eval()

    def propose(
        self,
        source: Path,
        prediction: str,
        rendered: Path | None,
        *,
        evidence_mode: str | None = None,
    ) -> dict:
        mode = evidence_mode or ("source_first" if rendered is not None else "source_only")
        paths, messages, prompt = evidence_message(source, prediction, rendered, mode)
        return self.generate(paths, messages, prompt)

    def generate(self, image_paths: list[Path], messages: list[dict], prompt: str) -> dict:
        from PIL import Image

        image_count = sum(
            block["type"] == "image" for message in messages for block in message["content"]
        )
        if image_count != len(image_paths) or not image_paths:
            raise ValueError("Message image markers must match the ordered image inputs")
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False, use_cache=True
            )
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        input_tokens = int(inputs.input_ids.shape[1])
        generated = output[:, input_tokens:]
        raw = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return {
            "raw_output": raw,
            "input_tokens": input_tokens,
            "output_tokens": int(generated.shape[1]),
            "generation_seconds": elapsed,
            "prompt": prompt,
            "messages": messages,
            "ordered_image_sha256": [sha256_file(path) for path in image_paths],
            "chat_template_sha256": digest(text),
            "image_grid_thw": inputs.image_grid_thw.detach().cpu().tolist(),
        }
