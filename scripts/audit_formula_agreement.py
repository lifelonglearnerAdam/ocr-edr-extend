#!/usr/bin/env python3
"""Post-hoc fixed agreement gates; shared errors can pass these gates."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from evaluate_formula_pilot import evaluate

from ocr_edr.formula_pilot import pixel_signature, sha256_file
from ocr_edr.loop import Observation
from ocr_edr.tex import TectonicRenderer

POLICIES = {
    "agree_first_last": ("source_last", ["source_first", "source_last"]),
    "agree_only_last": ("source_only", ["source_only", "source_last"]),
    "agree_all_three": ("source_only", ["source_only", "source_first", "source_last"]),
}


def replay_agreement(predictions: list[dict], renderer) -> tuple[list[dict], list[dict]]:
    index = {(row["sample_id"], row["arm"]): row for row in predictions}
    if len(index) != len(predictions):
        raise ValueError("Duplicate sample/arm")
    sample_ids = [row["sample_id"] for row in predictions if row["arm"] == "unchanged_0"]
    rows, decisions = [], []
    for sample_id in sample_ids:
        signatures = {}
        for arm in ["source_only", "source_first", "source_last"]:
            result = index[(sample_id, arm)]
            if len(result["trace"]) > 1:
                raise ValueError("Single-call outputs required")
            try:
                render = renderer.render(
                    Observation(sample_id, "formula", "", result["final_prediction"])
                )
                signatures[arm] = pixel_signature(Path(render.path))
            except Exception:
                signatures[arm] = None
        for policy, (selected, compared) in POLICIES.items():
            values = [signatures[arm] for arm in compared]
            agreed = all(value is not None and value == values[0] for value in values)
            result = copy.deepcopy(index[(sample_id, "unchanged_0")])
            result["arm"] = policy
            if agreed:
                result["final_prediction"] = index[(sample_id, selected)]["final_prediction"]
            # Pay for all calls used by this policy, including rejected outputs.
            result["trace"] = [
                call for arm in compared for call in copy.deepcopy(index[(sample_id, arm)]["trace"])
            ]
            rows.append(result)
            decisions.append({"sample_id": sample_id, "policy": policy, "agreement": agreed})
    return rows, decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory")
    args.output.mkdir(parents=True)
    path = args.run / "predictions.jsonl"
    predictions = list(map(json.loads, path.read_text().splitlines()))
    renderer = TectonicRenderer(args.output / "renders")
    results, decisions = replay_agreement(predictions, renderer)
    references = list(map(json.loads, args.references.read_text().splitlines()))
    rows, summary = evaluate(results, references, args.output, renderer=renderer)
    for item in summary:
        item["agreement_n"] = sum(d["agreement"] for d in decisions if d["policy"] == item["arm"])
        item["accepted_changes"] = sum(row["changed"] for row in rows if row["arm"] == item["arm"])
    (args.output / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in results)
    )
    (args.output / "evaluation.json").write_text(
        json.dumps(
            {
                "study": "post-hoc agreement control on existing outputs",
                "benchmark_evaluation": False,
                "new_model_calls": 0,
                "policy_reference_access": "none",
                "prediction_sha256": sha256_file(path),
                "reference_sha256": sha256_file(args.references),
                "summary": summary,
                "cases": rows,
                "decisions": decisions,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
