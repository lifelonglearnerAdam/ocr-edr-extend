"""Lazy, locally cached Qwen2-VL inference for the diagnostic pilot."""

from __future__ import annotations

import time
from pathlib import Path

from .formula_pilot import make_prompt


class QwenFormulaProposer:
    def __init__(self, model_path: Path, *, device: str = "cuda:0", max_new_tokens: int = 192):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=True,
            use_fast=False,
            min_pixels=128 * 28 * 28,
            max_pixels=256 * 28 * 28,
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(model_path),
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device)
        self.model.eval()

    def propose(self, source: Path, prediction: str, rendered: Path | None) -> dict:
        from PIL import Image

        image_paths = [source] + ([rendered] if rendered is not None else [])
        prompt = make_prompt(prediction, rendered is not None)
        messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image"} for _ in image_paths] + [{"type": "text", "text": prompt}]
                ),
            }
        ]
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
        }
