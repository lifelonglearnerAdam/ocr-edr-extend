#!/usr/bin/env python3
"""Two frozen train-only NF4 optimizer steps plus an exact adapter reload check."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from ocr_edr.sft import assistant_labels, sha256, verify_model_files
from ocr_edr.sft_training import train_fixed_schedule
from ocr_edr.table_training import load_table_supervision
from ocr_edr.training_precision import (
    configure_training_device,
    load_training_base,
    select_readiness_examples,
)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "config",
        "dataset",
        "admission",
        "model-path",
        "model-receipt",
        "preflight",
        "output",
    ]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    args.arm = "nf4_readiness_only"
    project = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        project / "src/ocr_edr/training_precision.py",
        project / "src/ocr_edr/sft_training.py",
        project / "src/ocr_edr/supervised_projection.py",
        project / "src/ocr_edr/sft.py",
        project / "src/ocr_edr/table_training.py",
        project / "src/ocr_edr/table_supervision.py",
        project / "src/ocr_edr/table_pilot.py",
        args.config.resolve(),
    ]
    receipt = {
        "status": "initializing",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "study": cfg["study"],
        "arm": args.arm,
        "config": cfg,
        "config_sha256": sha256(args.config),
        "completed_steps": 0,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project, text=True
        ).strip(),
        "source_sha256": {p.relative_to(project).as_posix(): sha256(p) for p in paths},
        "source_role": "train",
        "dev_optimizer_examples": 0,
        "calibration_locked_optimizer_examples": 0,
        "accuracy_evaluation": False,
        "checkpoint_initializes_full_training": False,
        "adapter_reload_verified": False,
    }
    write(root / "run.json", receipt)
    for path in paths:
        target = root / "source" / path.relative_to(project)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    try:
        if (
            cfg["precision_profile"] != "nf4_lora_8gb"
            or cfg["steps"] != 2
            or cfg["gradient_accumulation"] != 1
        ):
            raise ValueError("Only the frozen two-step NF4 trial is supported")
        if args.model_path.resolve().name != cfg["model_revision"]:
            raise ValueError("Pinned original model revision required")
        verify_model_files(args.model_path, json.loads(args.model_receipt.read_text()))
        if sha256(args.admission / "admission.json") != cfg["admission_sha256"]:
            raise ValueError("Frozen training admission differs")
        if sha256(args.preflight) != cfg["selection"]["preflight_sha256"]:
            raise ValueError("Frozen boundary-sample selection metadata differs")
        train = load_table_supervision(args.admission, args.dataset, role="train")
        metadata = [json.loads(line) for line in args.preflight.read_text().splitlines()]
        schedule = select_readiness_examples(train, metadata)
        if [train[index]["sample_id"] for index in schedule] != cfg["selection"][
            "expected_samples"
        ]:
            raise ValueError("Predeclared readiness selection drift")
        receipt.update(
            exposures=len(schedule),
            selected_samples=cfg["selection"]["expected_samples"],
            admission_sha256=sha256(args.admission / "admission.json"),
            selection_preflight_sha256=sha256(args.preflight),
            model_receipt_sha256=sha256(args.model_receipt),
            versions={name: importlib.metadata.version(name) for name in cfg["versions"]},
        )
        if receipt["versions"] != cfg["versions"]:
            raise ValueError("Frozen NF4 readiness dependency versions differ")
        # This one-device workstation trial does not silently choose another GPU.
        visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visibility not in {None, "", "0"}:
            raise ValueError("Readiness requires the explicitly inspected physical GPU 0")
        devices = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.free,memory.total,utilization.gpu",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            .strip()
            .splitlines()
        )
        processes = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        if len(devices) != 1 or processes:
            raise RuntimeError("Readiness needs one local GPU with no competing compute process")
        receipt["gpu_observation_before_initialization"] = devices

        import torch
        from peft import PeftModel, get_peft_model_state_dict
        from PIL import Image
        from safetensors.torch import load_file
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            str(args.model_path),
            local_files_only=True,
            use_fast=False,
            min_pixels=cfg["min_pixels"],
            max_pixels=cfg["max_pixels"],
        )
        by_id = {row["sample_id"]: row for row in metadata}
        for index in schedule:
            row = train[index]
            user = {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": row["prompt"]}],
            }
            prefix_text = processor.apply_chat_template(
                [user], tokenize=False, add_generation_prompt=True
            )
            full_text = processor.apply_chat_template(
                [user, {"role": "assistant", "content": [{"type": "text", "text": row["target"]}]}],
                tokenize=False,
                add_generation_prompt=False,
            )
            with Image.open(row["resolved_source_image"]) as source:
                image = source.convert("RGB")
            prefix = processor(text=[prefix_text], images=[image], return_tensors="pt")
            full = processor(text=[full_text], images=[image], return_tensors="pt")
            labels = assistant_labels(
                prefix.input_ids[0].tolist(),
                full.input_ids[0].tolist(),
                full.attention_mask[0].tolist(),
            )
            actual = {
                "processed_tokens": len(labels),
                "masked_prefix_tokens": len(prefix.input_ids[0]),
                "supervised_tokens": sum(token != -100 for token in labels),
                "image_grid_thw": full.image_grid_thw.tolist(),
            }
            if any(by_id[row["sample_id"]][key] != value for key, value in actual.items()):
                raise ValueError("Actual processor differs from frozen boundary metadata")
        receipt["boundary_preflight_reproduced"] = True
        write(root / "run.json", receipt)
        train_fixed_schedule(args, cfg, train, schedule, root, receipt)
        # The kernel's terminal checkpoint is necessary but does not by itself
        # complete this trial; independently reload all saved adapter tensors.
        receipt["status"] = "checking_adapter_reload"
        write(root / "run.json", receipt)
        gc.collect()
        torch.cuda.empty_cache()
        reload_record = {"precision": configure_training_device(cfg, torch_module=torch)}
        base = load_training_base(args.model_path, cfg, reload_record)
        model = PeftModel.from_pretrained(base, str(root / "checkpoint"), is_trainable=False)
        saved = load_file(str(root / "checkpoint/adapter_model.safetensors"), device="cpu")
        loaded = get_peft_model_state_dict(model)
        if set(saved) != set(loaded) or any(
            not torch.equal(saved[key], loaded[key].detach().cpu()) for key in saved
        ):
            raise ValueError("Saved and reloaded adapter tensors differ")
        receipt.update(
            status="completed",
            adapter_reload_verified=True,
            adapter_reload_tensors=len(saved),
            reload_quantization=reload_record["quantization"],
        )
        print(
            json.dumps(
                {
                    key: receipt[key]
                    for key in [
                        "status",
                        "completed_steps",
                        "peak_memory_allocated_gib",
                        "peak_memory_reserved_gib",
                        "training_seconds",
                        "adapter_reload_verified",
                        "adapter_reload_tensors",
                    ]
                }
            ),
            flush=True,
        )
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error)[:1200])
        raise
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        write(root / "run.json", receipt)


if __name__ == "__main__":
    main()
