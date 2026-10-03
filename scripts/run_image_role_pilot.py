#!/usr/bin/env python3
"""Explore image-role interventions on fixed reference-free formula inputs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.formula_pilot import (
    MathTextRenderer,
    extract_formula,
    sha256_file,
    validate_pilot_inputs,
)
from ocr_edr.loop import Observation, digest
from ocr_edr.qwen import EVIDENCE_MODES, QwenFormulaProposer

REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--renderer", choices=["mathtext", "tectonic"], default="mathtext")
    parser.add_argument("--modes", nargs="+", choices=EVIDENCE_MODES, default=list(EVIDENCE_MODES))
    args = parser.parse_args()
    if len(set(args.modes)) != len(args.modes):
        parser.error("Duplicate evidence modes")
    if args.model_path.resolve().name != REVISION:
        parser.error("Use the pinned model snapshot")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    input_path = args.inputs.resolve()
    cases = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    validate_pilot_inputs(cases, input_path.parent)
    if args.limit is not None:
        cases = cases[: args.limit]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": "Qwen/Qwen2-VL-2B-Instruct",
        "model_revision": REVISION,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "source_hashes": {
            str(p): sha256_file(p)
            for p in [
                Path(__file__),
                Path("src/ocr_edr/qwen.py"),
                Path("src/ocr_edr/formula_pilot.py"),
            ]
        },
        "input_sha256": sha256_file(input_path),
        "cases": len(cases),
        "arms": ["unchanged_0", *args.modes],
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "seed": 20261003,
        "do_sample": False,
        "versions": {
            n: importlib.metadata.version(n)
            for n in ["torch", "torchvision", "transformers", "pillow", "matplotlib", "numpy"]
        },
        "reference_access": "none",
        "status": "initializing",
        "cpu_threads": 8,
        "image_pixels": {"min": 128 * 28 * 28, "max": 256 * 28 * 28},
        "renderer": args.renderer,
        "syntax_gate": "rollback only on render error",
        "dtype": "bfloat16",
        "attention": "sdpa",
        "image_processor": "slow",
    }
    (root / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    import torch

    torch.manual_seed(metadata["seed"])
    if args.device == "cpu":
        torch.set_num_threads(metadata["cpu_threads"])
    proposer = QwenFormulaProposer(
        args.model_path, device=args.device, max_new_tokens=args.max_new_tokens
    )
    if args.renderer == "tectonic":
        from ocr_edr.tex import TectonicRenderer

        renderer = TectonicRenderer(root / "renders")
        metadata["tectonic_binary_sha256"] = renderer.binary_sha256
    else:
        renderer = MathTextRenderer(root / "renders")
    count = 0
    try:
        with (root / "predictions.jsonl").open("w") as results:
            for case in cases:
                source = input_path.parent / case["source_image"]
                initial = case["prediction"]
                initial_render_error = None
                try:
                    rendered = renderer.render(
                        Observation(case["sample_id"], "formula", str(source), initial)
                    )
                except Exception as exc:
                    rendered = None
                    initial_render_error = f"{type(exc).__name__}: {str(exc)[:400]}"
                results.write(
                    json.dumps(
                        {
                            "sample_id": case["sample_id"],
                            "family_id": case["family_id"],
                            "arm": "unchanged_0",
                            "initial_prediction": initial,
                            "final_prediction": initial,
                            "trace": [],
                        }
                    )
                    + "\n"
                )
                for mode in args.modes:
                    if rendered is None and mode not in {"source_only", "duplicate_source"}:
                        results.write(
                            json.dumps(
                                {
                                    "sample_id": case["sample_id"],
                                    "family_id": case["family_id"],
                                    "arm": mode,
                                    "initial_prediction": initial,
                                    "final_prediction": initial,
                                    "trace": [],
                                    "skip_reason": "initial_render_failed",
                                    "initial_render_error": initial_render_error,
                                }
                            )
                            + "\n"
                        )
                        continue
                    call = proposer.propose(
                        source,
                        initial,
                        Path(rendered.path) if rendered else None,
                        evidence_mode=mode,
                    )
                    count += 1
                    candidate, extraction = extract_formula(call["raw_output"])
                    error = None
                    try:
                        renderer.render(
                            Observation(case["sample_id"], "formula", str(source), candidate)
                        )
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {str(exc)[:400]}"
                    final = candidate if error is None else initial
                    call.update(
                        {
                            "call_id": count,
                            "evidence_mode": mode,
                            "input_prediction": initial,
                            "input_prediction_sha256": digest(initial),
                            "candidate": candidate,
                            "final_prediction": final,
                            "extraction": extraction,
                            "render_error": error,
                        }
                    )
                    results.write(
                        json.dumps(
                            {
                                "sample_id": case["sample_id"],
                                "family_id": case["family_id"],
                                "arm": mode,
                                "initial_prediction": initial,
                                "final_prediction": final,
                                "trace": [call],
                            }
                        )
                        + "\n"
                    )
                results.flush()
                print(
                    f"{case['sample_id']}: {len(args.modes)} conditions completed; {count} calls",
                    flush=True,
                )
        metadata["status"] = "completed"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        raise
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        metadata["calls"] = count
        (root / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
