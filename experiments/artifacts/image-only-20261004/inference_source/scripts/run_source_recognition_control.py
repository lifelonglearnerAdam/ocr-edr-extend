#!/usr/bin/env python3
"""Fixed image-only Qwen recognition; native predictions and labels are not read."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.formula_pilot import extract_formula, sha256_file
from ocr_edr.qwen import QwenFormulaProposer

PROMPT = (
    "Transcribe the mathematical expression in the image into LaTeX. Preserve every visible "
    "symbol, subscript, superscript, fraction and delimiter. Do not solve or simplify it. "
    "Return only the expression inside <latex>...</latex>. Do not describe the image or return "
    "bounding-box coordinates."
)


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory")
    inputs = list(map(json.loads, args.inputs.read_text().splitlines()))
    allowed = {"family_id", "source_image", "source_sha256"}
    seen = set()
    for row in inputs:
        if set(row) != allowed or row["family_id"] in seen:
            raise ValueError("Unexpected fields or duplicate source; labels/candidates forbidden")
        seen.add(row["family_id"])
        if sha256_file(args.inputs.parent / row["source_image"]) != row["source_sha256"]:
            raise ValueError("Source image hash mismatch")
    torch.set_num_threads(8)
    torch.manual_seed(20261004)
    args.output.mkdir(parents=True)
    metadata = {
        "study": "image-only recognition framing/resolution control",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": "Qwen/Qwen2-VL-2B-Instruct",
        "model_revision": args.model.name,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "source_hashes": {
            path: sha256_file(Path(path))
            for path in ["scripts/run_source_recognition_control.py", "src/ocr_edr/qwen.py"]
        },
        "input_sha256": sha256_file(args.inputs),
        "reference_access": "none",
        "candidate_access": "none",
        "cases": len(inputs),
        "device": "cpu",
        "dtype": "bfloat16",
        "cpu_threads": 8,
        "max_new_tokens": 128,
        "seed": 20261004,
        "status": "running",
        "calls": 0,
        "conditions": {"image_only_low": [100352, 200704], "image_only_high": [100352, 802816]},
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        with (args.output / "predictions.jsonl").open("w") as sink:
            for arm, (_, max_pixels) in metadata["conditions"].items():
                proposer = QwenFormulaProposer(
                    args.model, device="cpu", max_new_tokens=128, max_pixels=max_pixels
                )
                for row in inputs:
                    source = args.inputs.parent / row["source_image"]
                    messages = [
                        {
                            "role": "user",
                            "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
                        }
                    ]
                    call = proposer.generate([source], messages, PROMPT)
                    candidate, extraction = extract_formula(call["raw_output"])
                    call.update({"candidate": candidate, "extraction": extraction})
                    sink.write(json.dumps({**row, "arm": arm, "call": call}) + "\n")
                    sink.flush()
                    metadata["calls"] += 1
                    print(f"{arm} {row['family_id']}: {metadata['calls']} calls", flush=True)
                del proposer
        metadata["status"] = "completed"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        raise
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
