#!/usr/bin/env python3
"""Replay fixed one-call outputs through exploratory, reference-free adapters."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from evaluate_formula_pilot import evaluate

from ocr_edr.formula_audit import audit_candidate
from ocr_edr.formula_pilot import sha256_file
from ocr_edr.loop import Observation
from ocr_edr.tex import TectonicRenderer


def replay(predictions: list[dict], renderer, policy: str) -> list[dict]:
    results = []
    for original in predictions:
        result = copy.deepcopy(original)
        if len(result["trace"]) > 1:
            raise ValueError("Post-hoc control requires fixed single-call predictions")
        result["arm"] += "__" + policy
        result["final_prediction"] = result["initial_prediction"]
        if result["trace"]:
            call = result["trace"][0]
            candidate, decision = audit_candidate(call["raw_output"], policy)
            call["audit_policy"] = policy
            call["audit_transform"] = decision
            call["original_render_error"] = call["render_error"]
            call["candidate"] = candidate
            call["render_error"] = None
            if candidate is not None:
                try:
                    renderer.render(Observation(result["sample_id"], "formula", "", candidate))
                    result["final_prediction"] = candidate
                except Exception as exc:
                    call["render_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            call["final_prediction"] = result["final_prediction"]
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory to preserve audit provenance")
    args.output.mkdir(parents=True)
    path = args.run / "predictions.jsonl"
    predictions = [json.loads(line) for line in path.read_text().splitlines()]
    renderer = TectonicRenderer(args.output / "renders")
    all_results = []
    for policy in ["normalize_environment", "require_latex_tags"]:
        all_results.extend(replay(predictions, renderer, policy))
    # Decisions and rendering above are complete before any labels are opened.
    references = [json.loads(line) for line in args.references.read_text().splitlines()]
    rows, summary = evaluate(all_results, references, args.output, renderer=renderer)
    (args.output / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in all_results)
    )
    metadata = {
        "study": "post-hoc output-adapter control",
        "benchmark_evaluation": False,
        "new_model_calls": 0,
        "policy_reference_access": "none",
        "prediction_sha256": sha256_file(path),
        "reference_sha256": sha256_file(args.references),
        "adapter_sha256": sha256_file(Path("src/ocr_edr/formula_audit.py")),
        "summary": summary,
        "cases": rows,
    }
    (args.output / "evaluation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
