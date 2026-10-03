#!/usr/bin/env python3
"""Run a pinned untrained model on four formula proposal conditions, without GT."""

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
from ocr_edr.qwen import QwenFormulaProposer

ARMS = ("source_only_1", "with_render_1", "fresh_render_2", "stale_render_2")
REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    input_path = args.inputs.resolve()
    cases = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        cases = cases[: args.limit]
    validate_pilot_inputs(cases, input_path.parent)
    if args.model_path.resolve().name != REVISION:
        parser.error("--model-path must be the pinned Hugging Face snapshot")
    versions = {
        name: importlib.metadata.version(name)
        for name in ["torch", "transformers", "accelerate", "pillow", "matplotlib", "numpy"]
    }
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": "Qwen/Qwen2-VL-2B-Instruct",
        "model_revision": REVISION,
        "model_config_sha256": sha256_file(args.model_path / "config.json"),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "source_hashes": {
            str(path): sha256_file(path)
            for path in [
                Path(__file__),
                Path("src/ocr_edr/formula_pilot.py"),
                Path("src/ocr_edr/qwen.py"),
            ]
        },
        "input_sha256": sha256_file(input_path),
        "cases": len(cases),
        "arms": ("unchanged_0", *ARMS),
        "versions": versions,
        "device": args.device,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "attention": "sdpa",
        "dtype": "bfloat16",
        "image_processor": "slow; use_fast=False",
        "status": "initializing",
        "image_pixels": {"min": 128 * 28 * 28, "max": 256 * 28 * 28},
        "reference_access": "none",
        "gate": "syntax rollback only; no learned correctness verifier",
        "first_render_turn_shared": True,
    }
    (root / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    import torch

    torch.manual_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(8)
        metadata["cpu_threads"] = torch.get_num_threads()
    if args.device.startswith("cuda"):
        metadata["gpu"] = torch.cuda.get_device_name()
        metadata["cuda_version"] = torch.version.cuda
        torch.cuda.reset_peak_memory_stats()
    (root / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    proposer = QwenFormulaProposer(
        args.model_path.resolve(), device=args.device, max_new_tokens=args.max_new_tokens
    )
    renderer = MathTextRenderer(root / "renders")
    calls = 0
    with (root / "predictions.jsonl").open("w") as results:
        for case in cases:
            source = (input_path.parent / case["source_image"]).resolve()
            initial = case["prediction"]
            initial_render = renderer.render(
                Observation(case["sample_id"], "formula", str(source), initial)
            )

            def propose(prediction: str, rendered: Path | None) -> dict:
                nonlocal calls
                call = proposer.propose(source, prediction, rendered)
                calls += 1
                candidate, extraction = extract_formula(call["raw_output"])
                error = None
                try:
                    candidate_render = renderer.render(
                        Observation(case["sample_id"], "formula", str(source), candidate)
                    )
                except Exception as exc:
                    candidate_render = None
                    error = f"{type(exc).__name__}: {str(exc)[:400]}"
                return {
                    **call,
                    "call_id": calls,
                    "input_prediction": prediction,
                    "input_prediction_sha256": digest(prediction),
                    "render_sha256": sha256_file(rendered) if rendered is not None else None,
                    "render_matches_current": (
                        None
                        if rendered is None
                        else rendered
                        == Path(
                            renderer.render(
                                Observation(case["sample_id"], "formula", str(source), prediction)
                            ).path
                        )
                    ),
                    "extraction": extraction,
                    "candidate": candidate,
                    "render_error": error,
                    "final_prediction": candidate if candidate_render is not None else prediction,
                }

            source_call = propose(initial, None)
            render_call = propose(initial, Path(initial_render.path))
            current = render_call["final_prediction"]
            fresh = renderer.render(Observation(case["sample_id"], "formula", str(source), current))
            fresh_call = propose(current, Path(fresh.path))
            stale_call = propose(current, Path(initial_render.path))
            paths = [
                ("unchanged_0", []),
                (ARMS[0], [source_call]),
                (ARMS[1], [render_call]),
                (ARMS[2], [render_call, fresh_call]),
                (ARMS[3], [render_call, stale_call]),
            ]
            for arm, trace in paths:
                results.write(
                    json.dumps(
                        {
                            "sample_id": case["sample_id"],
                            "family_id": case["family_id"],
                            "arm": arm,
                            "initial_prediction": initial,
                            "final_prediction": trace[-1]["final_prediction"] if trace else initial,
                            "trace": trace,
                        }
                    )
                    + "\n"
                )
            results.flush()
            print(
                f"{case['sample_id']}: completed four model conditions and unchanged baseline ({calls} unique calls)",
                flush=True,
            )
    metadata.update(
        {
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "unique_calls": calls,
            "status": "completed",
        }
    )
    if args.device.startswith("cuda"):
        metadata["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
        metadata["peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
    (root / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
