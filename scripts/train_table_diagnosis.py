#!/usr/bin/env python3
"""Train a separate class/location diagnosis adapter; only train-role targets."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from ocr_edr.sft import sha256, verify_model_files
from ocr_edr.sft_training import train_fixed_schedule
from ocr_edr.table_diagnosis import diagnosis_from_action, diagnosis_prompt
from ocr_edr.table_training import document_balanced_schedule, load_table_supervision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-receipt", type=Path, required=True)
    parser.add_argument("--arm", choices=["all"], default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.model_path.resolve().name != cfg["model_revision"]:
        parser.error("Use the pinned base checkpoint, not a prior experiment adapter")
    admission = json.loads((args.admission / "admission.json").read_text())
    if set(admission["excluded_families"]) != set(cfg["data"]["excluded_training_families"]):
        raise ValueError("Training exclusion policy drift")
    policy = Path(cfg["data"]["admission_policy"])
    if sha256(policy) != admission["policy_sha256"]:
        raise ValueError("Frozen admission policy hash mismatch")
    train = load_table_supervision(args.admission, args.dataset, role="train")
    if (
        len(train) != cfg["data"]["admitted_train_records"]
        or len({r["document_id"] for r in train}) != cfg["data"]["admitted_train_documents"]
    ):
        raise ValueError("Complete admitted train source coverage changed")
    for row in train:
        row["prompt"] = diagnosis_prompt(row["candidate"])
        row["target"] = json.dumps(
            diagnosis_from_action(row["candidate"], json.loads(row["target"])),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    schedule = document_balanced_schedule(
        train, arm=args.arm, passes_per_document=cfg["passes_per_document"], seed=cfg["seed"]
    )
    if len(schedule) != cfg["steps"] * cfg["gradient_accumulation"]:
        raise ValueError("Optimizer and document-exposure budgets differ")
    model_receipt = json.loads(args.model_receipt.read_text())
    if model_receipt["revision"] != cfg["model_revision"]:
        raise ValueError("Pinned diagnostic base revision mismatch")
    verify_model_files(args.model_path, model_receipt)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__),
        repo / "src/ocr_edr/table_training.py",
        repo / "src/ocr_edr/table_diagnosis.py",
        repo / "src/ocr_edr/table_supervision.py",
        repo / "src/ocr_edr/table_pilot.py",
        repo / "src/ocr_edr/sft_training.py",
        repo / "src/ocr_edr/training_precision.py",
        repo / "src/ocr_edr/supervised_projection.py",
        repo / "src/ocr_edr/sft.py",
    ]
    receipt = {
        "status": "initializing",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "study": cfg["study"],
        "arm": args.arm,
        "config": cfg,
        "config_sha256": sha256(args.config),
        "model_receipt_sha256": sha256(args.model_receipt),
        "completed_steps": 0,
        "admission_sha256": sha256(args.admission / "admission.json"),
        "train_manifest_sha256": admission["file_sha256"]["train-sft.jsonl"],
        "dev_targets_loaded": False,
        "protocol_sha256": sha256(repo / "docs/research/TABLE_DIAGNOSIS_PROTOCOL_20261009.md"),
        "target_projection": "controlled verdict/class/current-region only; no repaired text or span",
        "diagnosis_targets_sha256": __import__("hashlib")
        .sha256(
            json.dumps(
                [(r["sample_id"], r["target"]) for r in train], separators=(",", ":")
            ).encode()
        )
        .hexdigest(),
        "dev_optimizer_examples": 0,
        "calibration_locked_optimizer_examples": 0,
        "target_label_scope": "published weak supervision; only known source conflict removed; other unreviewed annotation noise unresolved",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "source_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
        ),
        "source_sha256": {str(p.relative_to(repo)): sha256(p) for p in paths},
        "exposures": len(schedule),
        "document_exposures": dict(Counter(train[i]["document_id"] for i in schedule)),
        "family_exposures": dict(Counter(train[i]["family_id"] for i in schedule)),
        "variant_exposures": dict(Counter(train[i]["variant"] for i in schedule)),
        "checkpoint_selection": "fixed terminal step; no dev score or loss selection",
        "versions": {},
    }
    (root / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        for path in paths:
            destination = root / "source" / path.resolve().relative_to(repo)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        shutil.copyfile(args.config, root / "source/training-config.yaml")
        receipt["versions"] = {
            n: importlib.metadata.version(n)
            for n in ["torch", "transformers", "peft", "accelerate", "Pillow"]
        }
        if cfg.get("precision_profile") == "nf4_lora_8gb":
            receipt["versions"]["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
            if any(receipt["versions"].get(k) != v for k, v in cfg["versions"].items()):
                raise ValueError("Frozen NF4 runtime versions differ")
            occupancy = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
            ).strip()
            if occupancy:
                raise RuntimeError(
                    "Explicit local NF4 run requires no competing GPU compute process"
                )
        train_fixed_schedule(args, cfg, train, schedule, root, receipt)
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error)[:1000])
        raise
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        (root / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
