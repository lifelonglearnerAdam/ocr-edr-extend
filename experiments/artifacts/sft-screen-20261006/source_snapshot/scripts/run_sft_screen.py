#!/usr/bin/env python3
"""Same-runtime reference-free inference on every fixed development candidate."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from ocr_edr.formula_pilot import validate_pilot_inputs
from ocr_edr.qwen import QwenFormulaProposer
from ocr_edr.sft import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--model-receipt", required=True, type=Path)
    parser.add_argument("--adapter-run", type=Path)
    parser.add_argument(
        "--condition", required=True, choices=["base", "all", "no_explicit_preservation"]
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    cases = [json.loads(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    validate_pilot_inputs(cases, args.inputs.parent)
    if len(cases) != cfg["data"]["dev_records"]:
        parser.error("Every fixed development input must be retained")
    integrity = json.loads(args.model_receipt.read_text())
    if (
        integrity["revision"] != cfg["model_revision"]
        or args.model_path.resolve().name != cfg["model_revision"]
    ):
        parser.error("Use the pinned model and receipt")
    for name, expected in integrity["files"].items():
        p = (args.model_path / name).resolve()
        p.relative_to(args.model_path.resolve())
        if p.stat().st_size != expected["bytes"] or sha256(p) != expected["sha256"]:
            raise ValueError("Model integrity mismatch")
    adapter = None
    checkpoint_hashes = {}
    if args.adapter_run:
        run = json.loads((args.adapter_run / "run.json").read_text())
        if (
            args.condition != run["arm"]
            or run["status"] != "completed"
            or run["completed_steps"] != cfg["steps"]
            or run["dev_optimizer_examples"] != 0
        ):
            raise ValueError("Adapter experiment completion/condition mismatch")
        adapter = args.adapter_run / "checkpoint"
        checkpoint_hashes = run["checkpoint_sha256"]
        for name, expected in checkpoint_hashes.items():
            if sha256(adapter / name) != expected:
                raise ValueError("Adapter checksum mismatch")
    elif args.condition != "base":
        parser.error("SFT condition requires its complete adapter run")
    if args.condition == "base" and adapter is not None:
        parser.error("Base condition must not load any adapter")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    receipt = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "initializing",
        "condition": args.condition,
        "model_revision": cfg["model_revision"],
        "adapter_sha256": checkpoint_hashes,
        "input_sha256": sha256(args.inputs),
        "cases": len(cases),
        "reference_access": "none",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "source_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
        ),
        "source_sha256": {
            str(p.relative_to(repo)): sha256(p)
            for p in [
                Path(__file__),
                repo / "src/ocr_edr/qwen.py",
                repo / "src/ocr_edr/formula_pilot.py",
            ]
        },
        "config_sha256": sha256(args.config),
        "config": cfg,
        "versions": {
            n: importlib.metadata.version(n) for n in ["torch", "transformers", "peft", "Pillow"]
        },
        "gate": "none during generation; raw calls frozen before offline adapter and reference scoring",
    }
    (root / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    import torch

    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.set_num_threads(4)
    proposer = QwenFormulaProposer(
        args.model_path,
        device="cuda:0",
        max_new_tokens=cfg["inference"]["max_new_tokens"],
        min_pixels=cfg["min_pixels"],
        max_pixels=cfg["max_pixels"],
        adapter_path=adapter,
    )
    count = 0
    try:
        with (root / "calls.jsonl").open("w") as sink:
            for case in cases:
                source = (args.inputs.parent / case["source_image"]).resolve()
                call = proposer.propose(
                    source, case["prediction"], None, evidence_mode="source_only"
                )
                row = {
                    **case,
                    "condition": args.condition,
                    **call,
                    "hit_token_cap": call["output_tokens"] >= cfg["inference"]["max_new_tokens"],
                }
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                count += 1
                if count % 16 == 0:
                    print(f"{args.condition}: {count}/{len(cases)} frozen calls", flush=True)
        receipt.update(
            status="completed", completed_calls=count, calls_sha256=sha256(root / "calls.jsonl")
        )
    except Exception as error:
        receipt.update(
            status="failed",
            completed_calls=count,
            error_type=type(error).__name__,
            error=str(error)[:1000],
        )
        raise
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        (root / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps({k: receipt[k] for k in ["condition", "completed_calls", "calls_sha256"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
