#!/usr/bin/env python3
"""Bounded LoRA direct-target screen; independently validated dev never enters training."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from ocr_edr.sft import (
    load_supervision,
    sha256,
    training_schedule,
    validate_disjoint,
)
from ocr_edr.sft_training import train_fixed_schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--model-receipt", required=True, type=Path)
    parser.add_argument("--arm", choices=["all", "no_explicit_preservation"], required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.model_path.resolve().name != cfg["model_revision"]:
        parser.error("Use the pinned base-model snapshot")
    steps, accumulation = cfg["steps"], cfg["gradient_accumulation"]
    if type(steps) is not int or type(accumulation) is not int or min(steps, accumulation) < 1:
        parser.error("Positive integer step/accumulation budgets required")
    dataset = args.dataset.resolve()
    summary = json.loads((dataset / "supervision/summary.json").read_text())
    train = load_supervision(
        dataset / "supervision/train-sft.jsonl",
        dataset,
        split="train",
        expected_sha256=summary["splits"]["train"]["sft_sha256"],
    )
    dev = load_supervision(
        dataset / "supervision/dev-sft.jsonl",
        dataset,
        split="dev",
        expected_sha256=summary["splits"]["dev"]["sft_sha256"],
    )
    validate_disjoint(train, dev)
    if len(train) != cfg["data"]["train_records_all"] or len(dev) != cfg["data"]["dev_records"]:
        parser.error("Frozen train/dev counts changed")
    schedule = training_schedule(
        train, arm=args.arm, exposures=steps * accumulation, seed=cfg["seed"]
    )
    model_receipt = json.loads(args.model_receipt.read_text())
    if model_receipt["revision"] != cfg["model_revision"]:
        parser.error("Model integrity receipt revision mismatch")
    for name, expected in model_receipt["files"].items():
        path = (args.model_path / name).resolve()
        path.relative_to(args.model_path.resolve())
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError("Pinned model integrity mismatch")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    source_paths = [
        Path(__file__),
        repo / "src/ocr_edr/sft.py",
        repo / "src/ocr_edr/formula_pilot.py",
        repo / "src/ocr_edr/sft_training.py",
        repo / "src/ocr_edr/training_precision.py",
        repo / "src/ocr_edr/supervised_projection.py",
    ]
    receipt = {
        "status": "initializing",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "study": cfg["study"],
        "arm": args.arm,
        "config": cfg,
        "config_sha256": sha256(args.config),
        "train_manifest_sha256": summary["splits"]["train"]["sft_sha256"],
        "dev_manifest_sha256": summary["splits"]["dev"]["sft_sha256"],
        "dev_optimizer_examples": 0,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "source_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
        ),
        "source_sha256": {str(p.relative_to(repo)): sha256(p) for p in source_paths},
        "exposures": len(schedule),
        "family_exposures": dict(Counter(train[i]["family_id"] for i in schedule)),
        "variant_exposures": dict(Counter(train[i]["variant"] for i in schedule)),
        "versions": {},
        "checkpoint_selection": "fixed terminal step; no dev loss or score selection",
    }
    (root / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        receipt["versions"] = {
            n: importlib.metadata.version(n)
            for n in ["torch", "transformers", "peft", "accelerate", "Pillow"]
        }
        train_fixed_schedule(args, cfg, train, schedule, root, receipt)
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error)[:1000])
        raise
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        (root / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
