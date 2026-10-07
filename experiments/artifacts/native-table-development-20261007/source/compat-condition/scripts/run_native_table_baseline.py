#!/usr/bin/env python3
"""Fixed PP-Structure inference with an explicitly recorded coordinate convention."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from ocr_edr.native_tables import (
    load_native_table_sources,
    original_frame_box,
    recognize_table_source,
)
from ocr_edr.sft import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["dataset", "config", "author-source", "checkpoints", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    box_mode = cfg.get("box_decode", "author")
    if box_mode not in {"author", "original_frame_compatibility"}:
        raise ValueError("Unknown native box coordinate convention")
    if box_mode != "author" and cfg["options"]["table_algorithm"] != "SLANet":
        raise ValueError("Original-frame compatibility is specific to standard SLANet")
    source = args.author_source.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        revision != cfg["author_revision"]
        or subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True
        ).strip()
    ):
        raise ValueError("Use the clean pinned author source")
    if cfg["options"]["use_gpu"] or cfg["options"]["use_onnx"]:
        raise ValueError("This native development protocol uses Paddle CPU inference only")
    inputs = load_native_table_sources(args.dataset)
    if len(inputs) != cfg["expected_sources"]:
        raise ValueError("Retain every frozen native source")
    weights = json.loads((args.checkpoints / "receipt.json").read_text())
    if weights["source_revision"] != revision:
        raise ValueError("Model/source provenance mismatch")
    verified = set()
    weight_root = args.checkpoints.resolve()
    for asset in weights["assets"]:
        for name, expected in asset["files"].items():
            path = (weight_root / name).resolve(strict=True)
            path.relative_to(weight_root)
            if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
                raise ValueError("Native model checksum mismatch")
            verified.add(path)
    for directory in cfg["models"].values():
        path = (weight_root / directory).resolve()
        path.relative_to(weight_root)
        files = {p.resolve() for p in path.rglob("*") if p.is_file()}
        if not files or not files <= verified:
            raise ValueError("Model directory contains unverified files")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "initializing",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "parser": "PaddleOCR English table det/rec + SLANet + official TableMatch",
        "author_revision": revision,
        "config": cfg,
        "config_sha256": sha256(args.config),
        "model_receipt_sha256": sha256(args.checkpoints / "receipt.json"),
        "dataset_sha256": sha256(args.dataset / "dataset.json"),
        "input_sha256": sha256(args.dataset / "model_dev-source-inputs.jsonl"),
        "cases": len(inputs),
        "completed_sources": 0,
        "reference_access": "none",
        "calibration_locked_images_loaded": False,
        "source_selection_uses_model_outputs": False,
        "retries_per_source": 0,
        "device": "cpu",
        "precision": "fp32",
        "box_decode": box_mode,
        "runtime_override": (
            None
            if box_mode == "author"
            else "TableLabelDecode._bbox_decode uses original-frame width/height, restoring TableBoxEncode inverse and author pre-ac5313d semantics; all weights/logits/other inference unchanged"
        ),
        "python": sys.version,
        "machine_architecture": platform.machine(),
        "driver_sha256": sha256(Path(__file__)),
        "wrapper_sha256": sha256(
            Path(__file__).resolve().parents[1] / "src/ocr_edr/native_tables.py"
        ),
        "author_file_sha256": {
            str(p.relative_to(source)): sha256(p)
            for pattern in ["*.py", "*.txt"]
            for p in sorted(source.rglob(pattern))
            if ".git" not in p.parts
        },
    }
    (root / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    count, failures = 0, 0
    try:
        for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
            if os.environ.get(name) != str(cfg["options"]["cpu_threads"]):
                raise ValueError("Set the frozen CPU thread limits before launch")
        import cv2
        import numpy as np
        import paddle

        if paddle.is_compiled_with_cuda() or paddle.get_device() != "cpu":
            raise ValueError("Use the isolated CPU Paddle build")
        sys.path.insert(0, str(source))
        from ppstructure.table.predict_table import TableSystem
        from ppstructure.utility import init_args

        receipt["versions"] = {
            n: importlib.metadata.version(n)
            for n in [
                "paddlepaddle",
                "numpy",
                "opencv-python-headless",
                "albumentations",
                "Pillow",
                "scipy",
                "shapely",
                "pyclipper",
            ]
        }
        options = init_args().parse_args([])
        for key, value in cfg["options"].items():
            setattr(options, key, value)
        for key, directory in cfg["models"].items():
            setattr(options, key + "_model_dir", str(weight_root / directory))
        for key, relative in cfg["character_dictionaries"].items():
            path = (source / relative).resolve(strict=True)
            path.relative_to(source)
            setattr(options, key, str(path))
        receipt["resolved_author_options"] = vars(options)
        np.random.seed(cfg["seed"])
        paddle.seed(cfg["seed"])
        started = time.perf_counter()
        engine = TableSystem(options)
        if box_mode == "original_frame_compatibility":
            import types

            decoder = engine.table_structurer.postprocess_op
            if type(decoder).__name__ != "TableLabelDecode":
                raise ValueError("Unexpected decoder for the declared compatibility override")

            def decode_original(_decoder, bbox, shape):
                return np.asarray(original_frame_box(bbox, shape), dtype=bbox.dtype)

            decoder._bbox_decode = types.MethodType(decode_original, decoder)
        receipt.update(status="running", initialization_seconds=time.perf_counter() - started)
        (root / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")

        def recognize(path):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError("OpenCV could not decode the verified source image")
            result, times = engine(image, return_ocr_result_in_table=True)
            ocr = [
                {
                    "text": str(text),
                    "confidence": float(confidence) if math.isfinite(float(confidence)) else None,
                }
                for text, confidence in result["rec_res"]
            ]
            details = {
                "cell_bbox": result["cell_bbox"],
                "ocr_boxes": result["boxes"],
                "ocr_recognition": ocr,
                "backend_seconds": {k: float(v) for k, v in times.items()},
            }
            json.dumps(details, allow_nan=False)
            return result["html"], details

        with (root / "predictions.jsonl").open("w") as sink:
            for row in inputs:
                result = recognize_table_source(row, args.dataset.resolve(), recognize=recognize)
                sink.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                sink.flush()
                count += 1
                failures += bool(result["parser_error"])
                print(f"Native table {count}/{len(inputs)}; parser failures={failures}", flush=True)
        receipt.update(
            status="completed",
            parser_failures=failures,
            predictions_sha256=sha256(root / "predictions.jsonl"),
        )
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error)[:1000])
        raise
    finally:
        receipt.update(completed_sources=count, finished_at=datetime.now(timezone.utc).isoformat())
        (root / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
