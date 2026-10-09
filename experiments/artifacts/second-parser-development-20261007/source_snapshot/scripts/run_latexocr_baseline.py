#!/usr/bin/env python3
"""One fixed author LaTeX-OCR prediction per reference-free source image."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import subprocess
import sys
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.sft import sha256

REVISION = "5c1ac929bd19a7ecf86d5fb8d94771c8969fcb80"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--author-source", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.author_source.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        revision != REVISION
        or subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True
        ).strip()
    ):
        raise ValueError("Use the clean pinned author source")
    records = [json.loads(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    if len(records) != 32 or len({r["family_id"] for r in records}) != 32:
        raise ValueError("All 32 fixed development sources are required")
    for row in records:
        if set(row) != {"family_id", "source_image", "source_sha256"}:
            raise ValueError("Recognizer inputs may contain only source-image identity")
        if sha256(args.inputs.parent / row["source_image"]) != row["source_sha256"]:
            raise ValueError("Source-image hash mismatch")
    weight_receipt = json.loads((args.checkpoints / "receipt.json").read_text())
    if weight_receipt["tag"] != "v0.0.1":
        raise ValueError("Unexpected author checkpoint tag")
    for asset in weight_receipt["assets"]:
        p = args.checkpoints / asset["name"]
        if p.stat().st_size != asset["size"] or sha256(p) != asset["sha256"]:
            raise ValueError("Fixed author weight identity mismatch")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    receipt = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "initializing",
        "parser": "lukas-blecher/LaTeX-OCR",
        "author_source_revision": revision,
        "checkpoint": weight_receipt,
        "input_sha256": sha256(args.inputs),
        "reference_access": "none",
        "cases": len(records),
        "device": "cpu",
        "dtype": "fp32",
        "temperature": 0.2,
        "author_image_resizer": True,
        "clipboard_disabled": True,
        "seed_rule": "int(first 8 hex digits of SHA256(20261007:family_id))",
        "selection_uses_model_outputs": False,
        "retries_per_image": 0,
        "source_sha256": {
            str(p.relative_to(source)): sha256(p)
            for p in sorted((source / "pix2tex").rglob("*.py"))
        },
        "driver_sha256": sha256(Path(__file__)),
        "versions": {
            name: importlib.metadata.version(name)
            for name in [
                "torch",
                "torchvision",
                "numpy",
                "transformers",
                "timm",
                "x-transformers",
                "albumentations",
                "Pillow",
            ]
        },
    }
    (root / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    count = 0
    try:
        import numpy as np
        import pandas.io.clipboard as clipboard
        import torch
        from PIL import Image

        sys.path.insert(0, str(source))
        from pix2tex.cli import LatexOCR

        clipboard.copy = lambda *_args, **_kwargs: None
        torch.set_num_threads(8)
        options = Namespace(
            config=str(source / "pix2tex/model/settings/config.yaml"),
            tokenizer=str(source / "pix2tex/model/dataset/tokenizer.json"),
            checkpoint=str((args.checkpoints / "weights.pth").resolve()),
            no_cuda=True,
            no_resize=False,
            temperature=0.2,
        )
        model = LatexOCR(options)
        with (root / "predictions.jsonl").open("w") as sink:
            for row in records:
                seed = int(
                    hashlib.sha256(("20261007:" + row["family_id"]).encode()).hexdigest()[:8], 16
                )
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                capture = {}
                generate = model.model.generate

                def observed_generate(image, **kwargs):
                    generated = generate(image, **kwargs)
                    capture.update(
                        image_shape=list(image.shape),
                        generated_tokens=int(generated.shape[-1]),
                        reached_author_cap=int(generated.shape[-1]) >= model.args.max_seq_len,
                    )
                    return generated

                model.model.generate = observed_generate
                started = time.perf_counter()
                try:
                    with Image.open(args.inputs.parent / row["source_image"]) as image:
                        prediction = model(image.convert("RGB"))
                finally:
                    model.model.generate = generate
                result = {
                    **row,
                    "prediction": prediction,
                    "seed": seed,
                    "generation_seconds": time.perf_counter() - started,
                    **capture,
                }
                sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                sink.flush()
                count += 1
                print(f"LaTeX-OCR {count}/{len(records)} frozen sources", flush=True)
        receipt.update(
            status="completed",
            completed_sources=count,
            author_max_sequence_length=model.args.max_seq_len,
            predictions_sha256=sha256(root / "predictions.jsonl"),
        )
    except Exception as error:
        receipt.update(
            status="failed",
            completed_sources=count,
            error_type=type(error).__name__,
            error=str(error)[:800],
        )
        raise
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        (root / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
