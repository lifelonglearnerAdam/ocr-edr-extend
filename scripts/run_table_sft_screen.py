#!/usr/bin/env python3
"""Freeze one reference-free Qwen JSON-action call for every model-dev table case."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from ocr_edr.qwen import QwenFormulaProposer
from ocr_edr.sft import sha256, verify_model_files
from ocr_edr.table_sft_screen import (
    PROMPT_FORMATS,
    load_table_screen_inputs,
    table_messages,
    validate_table_adapter,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-receipt", type=Path, required=True)
    parser.add_argument("--adapter-run", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument(
        "--condition", choices=["base", "all", "no_explicit_preservation"], required=True
    )
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cuda:0")
    parser.add_argument("--prompt-format", choices=PROMPT_FORMATS, default="descriptive_schema")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    cases = load_table_screen_inputs(args.dataset)
    if (
        len(cases) != cfg["data"]["model_dev_records"]
        or len({r["family_id"] for r in cases}) != cfg["data"]["model_dev_documents"]
        or cfg["inference"]["do_sample"] is not False
    ):
        raise ValueError("Frozen model-dev coverage or greedy protocol mismatch")
    integrity = json.loads(args.model_receipt.read_text())
    if integrity["revision"] != cfg["model_revision"]:
        raise ValueError("Use the configured pinned model")
    verify_model_files(args.model_path, integrity)
    adapter, checkpoint_hashes, admission_hash, adapter_run_hash = None, {}, None, None
    if args.condition == "base":
        if args.adapter_run is not None:
            parser.error("Base condition cannot load an adapter")
    else:
        if args.adapter_run is None or args.admission is None:
            parser.error("SFT condition requires its completed run and admission receipt")
        admission_hash = sha256(args.admission / "admission.json")
        checkpoint_hashes = validate_table_adapter(
            args.adapter_run,
            condition=args.condition,
            config=cfg,
            config_sha256=sha256(args.config),
            admission_sha256=admission_hash,
        )
        adapter = args.adapter_run / "checkpoint"
        adapter_run_hash = sha256(args.adapter_run / "run.json")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    source_paths = [
        Path(__file__),
        repo / "src/ocr_edr/qwen.py",
        repo / "src/ocr_edr/sft.py",
        repo / "src/ocr_edr/table_sft_screen.py",
        repo / "src/ocr_edr/table_supervision.py",
        repo / "src/ocr_edr/table_pilot.py",
        repo / "src/ocr_edr/formula_pilot.py",
    ]
    receipt = {
        "status": "initializing",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "study": cfg["study"],
        "condition": args.condition,
        "prompt_format": args.prompt_format,
        "device": args.device,
        "dtype": "bfloat16",
        "attention": "sdpa",
        "cpu_threads": 4,
        "model_revision": cfg["model_revision"],
        "model_receipt_sha256": sha256(args.model_receipt),
        "adapter_sha256": checkpoint_hashes,
        "adapter_run_sha256": adapter_run_hash,
        "admission_sha256": admission_hash,
        "dataset_sha256": sha256(args.dataset / "dataset.json"),
        "input_sha256": sha256(args.dataset / "model_dev-inputs.jsonl"),
        "cases": len(cases),
        "completed_calls": 0,
        "reference_access": "none",
        "calibration_locked_images_loaded": False,
        "config": cfg,
        "config_sha256": sha256(args.config),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "source_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
        ),
        "source_sha256": {str(p.relative_to(repo)): sha256(p) for p in source_paths},
        "versions": {},
        "gate": "raw calls frozen before offline strict JSON/cap/render rollback; no visual acceptance judge",
    }
    (root / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    count = 0
    try:
        packages = ["torch", "transformers", "Pillow"] + (["peft"] if adapter else [])
        receipt["versions"] = {n: importlib.metadata.version(n) for n in packages}
        import torch

        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True, warn_only=True)
        proposer = QwenFormulaProposer(
            args.model_path,
            device=args.device,
            adapter_path=adapter,
            min_pixels=cfg["min_pixels"],
            max_pixels=cfg["max_pixels"],
            max_new_tokens=cfg["inference"]["max_new_tokens"],
        )
        receipt["status"] = "running"
        if args.device.startswith("cuda"):
            receipt.update(gpu=torch.cuda.get_device_name(0), cuda_runtime=torch.version.cuda)
        (root / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
        with (root / "calls.jsonl").open("w") as sink:
            for case in cases:
                source = (args.dataset / case["source_image"]).resolve()
                messages, prompt = table_messages(
                    case["prediction"], prompt_format=args.prompt_format
                )
                call = proposer.generate([source], messages, prompt)
                if call["ordered_image_sha256"] != [case["source_sha256"]]:
                    raise ValueError("Source changed after input validation")
                sink.write(
                    json.dumps({**case, **call, "condition": args.condition}, ensure_ascii=False)
                    + "\n"
                )
                sink.flush()
                count += 1
                if count % 8 == 0:
                    receipt["completed_calls"] = count
                    (root / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
                    print(f"{args.condition}: {count}/{len(cases)} table calls", flush=True)
        receipt.update(status="completed", calls_sha256=sha256(root / "calls.jsonl"))
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error)[:1000])
        raise
    finally:
        receipt.update(completed_calls=count, finished_at=datetime.now(timezone.utc).isoformat())
        (root / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps({k: receipt[k] for k in ["condition", "completed_calls", "calls_sha256"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
