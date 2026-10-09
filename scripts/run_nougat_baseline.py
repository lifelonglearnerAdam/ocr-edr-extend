#!/usr/bin/env python3
"""Generate native Nougat-LaTeX predictions from strict image-only source inputs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.formula_pilot import sha256_file

MODEL_REVISION = "6c6f2afe62ae51d57d0621ea9f2e11e02fb8385a"
PROCESSOR_REVISION = "d735d3a31bfd0cd48a020e01c5233a9154c6d4c2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--processor-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.model_path.resolve().name != MODEL_REVISION:
        parser.error("Use the pinned model snapshot")
    source_info = json.loads((args.processor_repo / "source.json").read_text())
    if source_info["revision"] != PROCESSOR_REVISION:
        parser.error("Use the pinned author image processor")
    input_path = args.inputs.resolve()
    cases = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        cases = cases[: args.limit]
    for case in cases:
        if set(case) != {"family_id", "source_image", "source_sha256"}:
            raise ValueError("Recognizer inputs must contain images and generic IDs only")
        if sha256_file(input_path.parent / case["source_image"]) != case["source_sha256"]:
            raise ValueError("Source hash mismatch")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": "Norm/nougat-latex-base",
        "model_revision": MODEL_REVISION,
        "processor_revision": PROCESSOR_REVISION,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "input_sha256": sha256_file(input_path),
        "cases": len(cases),
        "reference_access": "none",
        "processor_hashes": {
            str(p.relative_to(args.processor_repo)): sha256_file(p)
            for p in args.processor_repo.glob("nougat_latex/*.py")
        },
        "versions": {
            n: importlib.metadata.version(n) for n in ["torch", "transformers", "pillow", "numpy"]
        },
        "device": "cpu",
        "dtype": "bfloat16",
        "cpu_threads": 8,
        "num_beams": 5,
        "seed": 20261003,
        "status": "initializing",
    }
    (root / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    import torch
    from PIL import Image
    from transformers import VisionEncoderDecoderModel
    from transformers.models.nougat import NougatTokenizerFast

    sys.path.insert(0, str(args.processor_repo.resolve()))
    from nougat_latex import NougatLaTexProcessor

    torch.set_num_threads(metadata["cpu_threads"])
    torch.manual_seed(metadata["seed"])
    model = VisionEncoderDecoderModel.from_pretrained(
        str(args.model_path), local_files_only=True, dtype=torch.bfloat16
    ).eval()
    tokenizer = NougatTokenizerFast.from_pretrained(str(args.model_path), local_files_only=True)
    processor = NougatLaTexProcessor.from_pretrained(str(args.model_path), local_files_only=True)
    metadata["max_length"] = model.decoder.config.max_length
    count = 0
    try:
        with (root / "predictions.jsonl").open("w") as output:
            for case in cases:
                with Image.open(input_path.parent / case["source_image"]) as image:
                    pixels = processor(image.convert("RGB"), return_tensors="pt").pixel_values
                decoder_input = tokenizer(
                    tokenizer.bos_token, add_special_tokens=False, return_tensors="pt"
                ).input_ids
                started = time.perf_counter()
                with torch.inference_mode():
                    result = model.generate(
                        pixels.to(dtype=torch.bfloat16),
                        decoder_input_ids=decoder_input,
                        max_length=metadata["max_length"],
                        early_stopping=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=metadata["num_beams"],
                        bad_words_ids=[[tokenizer.unk_token_id]],
                        return_dict_in_generate=True,
                    )
                elapsed = time.perf_counter() - started
                raw = tokenizer.batch_decode(result.sequences)[0]
                text = raw
                for token in [tokenizer.eos_token, tokenizer.pad_token, tokenizer.bos_token]:
                    text = text.replace(token, "")
                row = {
                    **case,
                    "prediction": text.strip(),
                    "raw_output": raw,
                    "generation_seconds": elapsed,
                    "output_tokens": int(result.sequences.shape[1]),
                    "pixel_shape": list(pixels.shape),
                    "hit_length_cap": int(result.sequences.shape[1]) >= metadata["max_length"],
                }
                output.write(json.dumps(row) + "\n")
                output.flush()
                count += 1
                print(f"{case['family_id']}: recognized in {elapsed:.2f}s", flush=True)
        metadata["status"] = "completed"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        raise
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        metadata["completed_sources"] = count
        (root / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
