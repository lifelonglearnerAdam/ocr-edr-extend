#!/usr/bin/env python3
"""Variant and grouped contrasts for the frozen SFT screen, retaining all cases."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import _bootstrap  # noqa: F401

from ocr_edr.screen_statistics import paired_source_bootstrap, variant_summary
from ocr_edr.sft import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text())
    cases = evaluation["cases"]
    predictions = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    if sha256(args.predictions) != evaluation["prediction_sha256"]:
        raise ValueError("Evaluated prediction checksum drift")
    indexed = {(r["arm"], r["sample_id"]): r for r in predictions}
    if len(indexed) != len(predictions) or set(indexed) != {
        (r["arm"], r["sample_id"]) for r in cases
    }:
        raise ValueError("Case/cost full coverage mismatch")
    report = {
        "variants": variant_summary(cases),
        "comparisons": [
            paired_source_bootstrap(cases, arm, "base", seed=20261006)
            for arm in ["all", "no_explicit_preservation"]
        ],
        "cost_and_adapter": [],
        "metric_proxy_utility_sensitivity": [],
        "evaluation_sha256": sha256(args.evaluation),
        "predictions_sha256": sha256(args.predictions),
        "status": "single-seed development screen; no visual accuracy or benchmark transfer claim",
    }
    for arm in ["base", "all", "no_explicit_preservation"]:
        rows = [r for r in predictions if r["arm"] == arm]
        calls = [r["trace"][0] for r in rows]
        report["cost_and_adapter"].append(
            {
                "arm": arm,
                "calls": len(calls),
                "cap_hits": sum(c["hit_token_cap"] for c in calls),
                "contract_or_render_rollbacks": sum(
                    c["decision"] != "accepted_syntax_only" for c in calls
                ),
                "raw_contract_accepts": sum(c["decision"] == "accepted_syntax_only" for c in calls),
                "changed_finals": sum(
                    r["initial_prediction"] != r["final_prediction"] for r in rows
                ),
                "mean_input_tokens": mean(c["input_tokens"] for c in calls),
                "mean_output_tokens": mean(c["output_tokens"] for c in calls),
                "generation_seconds_total": sum(c["generation_seconds"] for c in calls),
                "mean_generation_seconds": mean(c["generation_seconds"] for c in calls),
                "timing_scope": "GPU generation only; not end-to-end; parallel GPU contention possible",
            }
        )
        selected = [r for r in cases if r["arm"] == arm]
        for weight in [1, 2, 4]:
            scores = [
                r["delta_cdm"]
                - weight * int(r["initial_cdm"]["F1_score"] == 1 and r["final_cdm"]["F1_score"] < 1)
                for r in selected
            ]
            report["metric_proxy_utility_sensitivity"].append(
                {
                    "arm": arm,
                    "regression_weight": weight,
                    "mean_utility": mean(scores),
                    "formula": "delta core CDM minus weight * initial-metric-Good regression indicator; quality units, no cost term",
                }
            )
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (root / "variants.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["variants"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(report["variants"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
