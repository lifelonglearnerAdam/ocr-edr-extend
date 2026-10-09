#!/usr/bin/env python3
"""Offline join/evaluation of image-only recognition; never feed labels to Qwen."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from evaluate_formula_pilot import evaluate

from ocr_edr.formula_audit import normalize_outer_environment
from ocr_edr.formula_pilot import sha256_file
from ocr_edr.loop import Observation
from ocr_edr.tex import TectonicRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recognition", type=Path, required=True)
    parser.add_argument("--native-inputs", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory")
    args.output.mkdir(parents=True)
    native = list(map(json.loads, args.native_inputs.read_text().splitlines()))
    by_family = {r["family_id"]: r for r in native}
    if len(by_family) != len(native):
        raise ValueError("Duplicate native family")
    path = args.recognition / "predictions.jsonl"
    recognitions = list(map(json.loads, path.read_text().splitlines()))
    renderer = TectonicRenderer(args.output / "renders")
    results = [
        {
            "sample_id": r["sample_id"],
            "family_id": r["family_id"],
            "arm": "unchanged_0",
            "initial_prediction": r["prediction"],
            "final_prediction": r["prediction"],
            "trace": [],
        }
        for r in native
    ]
    for recognition in recognitions:
        case = by_family[recognition["family_id"]]
        if case["source_sha256"] != recognition["source_sha256"]:
            raise ValueError("Source hash mismatch in offline join")
        for normalized in [False, True]:
            call = copy.deepcopy(recognition["call"])
            candidate = call["candidate"]
            if normalized:
                candidate, _ = normalize_outer_environment(candidate)
            error = None
            try:
                renderer.render(Observation(case["sample_id"], "formula", "", candidate))
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
            call.update({"candidate": candidate, "render_error": error})
            results.append(
                {
                    "sample_id": case["sample_id"],
                    "family_id": case["family_id"],
                    "arm": recognition["arm"] + ("__normalized" if normalized else ""),
                    "initial_prediction": case["prediction"],
                    "final_prediction": candidate,  # Failed recognition is not rolled back.
                    "trace": [call],
                }
            )
    refs = list(map(json.loads, args.references.read_text().splitlines()))
    rows, summary = evaluate(results, refs, args.output, renderer=renderer)
    (args.output / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in results)
    )
    (args.output / "evaluation.json").write_text(
        json.dumps(
            {
                "benchmark_evaluation": False,
                "recognition_sha256": sha256_file(path),
                "native_inputs_sha256": sha256_file(args.native_inputs),
                "reference_sha256": sha256_file(args.references),
                "summary": summary,
                "cases": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
