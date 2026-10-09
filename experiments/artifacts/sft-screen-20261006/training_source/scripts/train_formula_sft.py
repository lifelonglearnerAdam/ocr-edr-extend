#!/usr/bin/env python3
"""Bounded LoRA direct-target screen; independently validated dev never enters training."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from ocr_edr.sft import (
    assistant_labels,
    load_supervision,
    sha256,
    training_schedule,
    validate_disjoint,
)


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
        "versions": {
            n: importlib.metadata.version(n)
            for n in ["torch", "transformers", "peft", "accelerate", "Pillow"]
        },
        "checkpoint_selection": "fixed terminal step; no dev loss or score selection",
    }
    (root / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    import torch
    from peft import LoraConfig, get_peft_model
    from PIL import Image
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    if not torch.cuda.is_available() or torch.cuda.mem_get_info(0)[0] < 16 * 1024**3:
        raise RuntimeError("Selected GPU is unavailable or has less than 16 GiB free")
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    processor = AutoProcessor.from_pretrained(
        str(args.model_path),
        local_files_only=True,
        use_fast=False,
        min_pixels=cfg["min_pixels"],
        max_pixels=cfg["max_pixels"],
    )
    processed = {}
    preflight = []
    for index in sorted(set(schedule)):
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
        if len(labels) > cfg["max_sequence_tokens"]:
            raise ValueError("Sequence exceeds frozen cap; no silent truncation or row filtering")
        processed[index] = (full, torch.tensor([labels], dtype=torch.long))
        preflight.append(
            {
                "sample_id": row["sample_id"],
                "family_id": row["family_id"],
                "variant": row["variant"],
                "tokens": len(labels),
                "supervised_tokens": sum(token != -100 for token in labels),
                "masked_tokens": len(prefix.input_ids[0]),
                "grid": full.image_grid_thw.tolist(),
            }
        )
    (root / "preflight.jsonl").write_text("".join(json.dumps(row) + "\n" for row in preflight))
    (root / "schedule.jsonl").write_text(
        "".join(
            json.dumps({"exposure": i, "sample_id": train[index]["sample_id"]}) + "\n"
            for i, index in enumerate(schedule)
        )
    )
    metadata = {r["sample_id"]: r for r in preflight}
    receipt.update(
        preflight_sha256=sha256(root / "preflight.jsonl"),
        schedule_sha256=sha256(root / "schedule.jsonl"),
        total_processed_tokens=sum(metadata[train[i]["sample_id"]]["tokens"] for i in schedule),
        total_supervised_tokens=sum(
            metadata[train[i]["sample_id"]]["supervised_tokens"] for i in schedule
        ),
    )
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        str(args.model_path),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda:0")
    base.config.use_cache = False
    model = get_peft_model(
        base,
        LoraConfig(
            r=cfg["lora_rank"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=0,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    frozen = [(name, p) for name, p in model.named_parameters() if not p.requires_grad]
    if not trainable or any("lora_" not in name or "visual" in name for name, _ in trainable):
        raise ValueError("Only language-model LoRA parameters may be optimized")
    frozen_samples = [
        (name, p, p.detach().reshape(-1)[:64].clone())
        for name, p in [frozen[0], frozen[len(frozen) // 2], frozen[-1]]
    ]
    adapters_before = {name: p.detach().clone() for name, p in trainable}
    optimizer = torch.optim.AdamW(
        [p for _, p in trainable], lr=cfg["learning_rate"], weight_decay=0
    )
    receipt.update(
        status="running",
        trainable_parameters=sum(p.numel() for _, p in trainable),
        gpu=torch.cuda.get_device_name(0),
        cuda_runtime=torch.version.cuda,
    )
    (root / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    completed = 0
    try:
        with (root / "training.jsonl").open("w") as log:
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for index in schedule[step * accumulation : (step + 1) * accumulation]:
                    full, labels = processed[index]
                    inputs = {key: value.to("cuda:0") for key, value in full.items()}
                    result = model(**inputs, labels=labels.to("cuda:0"), use_cache=False)
                    if not torch.isfinite(result.loss):
                        raise ValueError("Nonfinite supervised loss")
                    (result.loss / accumulation).backward()
                    losses.append(float(result.loss.detach()))
                    del inputs, result
                if any(p.grad is not None for _, p in frozen):
                    raise ValueError("Frozen base parameter received a gradient")
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for _, p in trainable], cfg["clip_gradient"]
                )
                if not torch.isfinite(norm):
                    raise ValueError("Nonfinite LoRA gradient norm")
                optimizer.step()
                torch.cuda.synchronize()
                completed = step + 1
                log.write(
                    json.dumps(
                        {
                            "step": completed,
                            "loss_mean": sum(losses) / len(losses),
                            "gradient_norm_before_clip": float(norm),
                            "elapsed_seconds": time.perf_counter() - start,
                        }
                    )
                    + "\n"
                )
                log.flush()
                if completed % 24 == 0:
                    receipt.update(completed_steps=completed)
                    (root / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
                    print(
                        f"{args.arm}: step {completed}/{steps}; loss {sum(losses)/len(losses):.5f}",
                        flush=True,
                    )
        if any(
            not torch.equal(p.detach().reshape(-1)[:64], before) for _, p, before in frozen_samples
        ):
            raise ValueError("Frozen weight sample changed")
        changed = sum(not torch.equal(p.detach(), adapters_before[name]) for name, p in trainable)
        if not changed:
            raise ValueError("Training changed no adapter parameters")
        model.save_pretrained(root / "checkpoint", safe_serialization=True)
        receipt.update(
            status="completed",
            completed_steps=completed,
            optimizer_examples=len(schedule),
            dev_optimizer_examples=0,
            adapter_tensors_changed=changed,
            frozen_gradients=0,
            frozen_weight_samples_checked=len(frozen_samples),
            training_seconds=time.perf_counter() - start,
            peak_memory_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            peak_memory_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
            checkpoint_sha256={
                p.name: sha256(p) for p in sorted((root / "checkpoint").iterdir()) if p.is_file()
            },
        )
    except Exception as error:
        receipt.update(
            status="failed",
            completed_steps=completed,
            error_type=type(error).__name__,
            error=str(error)[:1000],
        )
        raise
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        (root / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in [
                    "arm",
                    "completed_steps",
                    "exposures",
                    "total_supervised_tokens",
                    "peak_memory_allocated_gib",
                    "training_seconds",
                    "checkpoint_sha256",
                ]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
