#!/usr/bin/env python3
"""Independently reload a completed-step NF4 checkpoint without changing its run."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.sft import sha256, verify_model_files
from ocr_edr.training_precision import configure_training_device, load_training_base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["training-run", "model-path", "model-receipt", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    run_path = args.training_run / "run.json"
    run = json.loads(run_path.read_text())
    receipt = {
        "status": "initializing",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "training_run_sha256": sha256(run_path),
        "training_run_original_status": run["status"],
        "driver_sha256": sha256(Path(__file__)),
        "precision_helper_sha256": sha256(
            Path(__file__).resolve().parents[1] / "src/ocr_edr/training_precision.py"
        ),
        "original_training_receipt_modified": False,
    }
    try:
        cfg = run["config"]
        if (
            cfg["precision_profile"] != "nf4_lora_8gb"
            or run["completed_steps"] != cfg["steps"]
            or run.get("adapter_tensors_changed", 0) < 1
            or run.get("frozen_gradients") != 0
            or run.get("frozen_weight_samples_checked", 0) < 1
            or run.get("dev_optimizer_examples") != 0
            or run.get("calibration_locked_optimizer_examples") != 0
        ):
            raise ValueError("Checkpoint lacks completed optimizer/gradient isolation evidence")
        checkpoint = args.training_run / "checkpoint"
        expected = run["checkpoint_sha256"]
        if {p.name for p in checkpoint.iterdir()} != set(expected):
            raise ValueError("Checkpoint inventory mismatch")
        for name, digest in expected.items():
            if Path(name).name != name or sha256(checkpoint / name) != digest:
                raise ValueError("Checkpoint integrity mismatch")
        verify_model_files(args.model_path, json.loads(args.model_receipt.read_text()))
        if sha256(args.model_receipt) != run["model_receipt_sha256"]:
            raise ValueError("Reload base checkpoint differs from training")
        receipt["versions"] = {name: importlib.metadata.version(name) for name in cfg["versions"]}
        if receipt["versions"] != cfg["versions"]:
            raise ValueError("Reload dependency versions differ")
        processes = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip()
        if processes:
            raise RuntimeError("Standalone reload needs an unoccupied GPU")
        import torch
        from peft import PeftModel, get_peft_model_state_dict
        from safetensors.torch import load_file

        receipt["precision"] = configure_training_device(cfg, torch_module=torch)
        base = load_training_base(args.model_path, cfg, receipt)
        model = PeftModel.from_pretrained(
            base, str(checkpoint), is_trainable=False, local_files_only=True
        )
        saved = load_file(str(checkpoint / "adapter_model.safetensors"), device="cpu")
        loaded = get_peft_model_state_dict(model)
        if set(saved) != set(loaded) or any(
            not torch.equal(saved[k], loaded[k].detach().cpu()) for k in saved
        ):
            raise ValueError("Saved and reloaded adapter tensors differ")
        if sha256(run_path) != receipt["training_run_sha256"]:
            raise ValueError("Original training evidence changed during verification")
        receipt.update(
            status="completed",
            adapter_reload_verified=True,
            adapter_reload_tensors=len(saved),
            checkpoint_sha256=expected,
            peak_memory_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
        )
        print(
            json.dumps(
                {
                    k: receipt[k]
                    for k in [
                        "status",
                        "adapter_reload_verified",
                        "adapter_reload_tensors",
                        "peak_memory_allocated_gib",
                    ]
                }
            )
        )
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error)[:1000])
        raise
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        (root / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
