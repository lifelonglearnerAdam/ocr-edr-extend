#!/usr/bin/env python3
"""Post-hoc synthetic proxies only: no CDM, VisFix, or benchmark claims."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import _bootstrap  # noqa: F401

from ocr_edr.formula_pilot import (
    MathTextRenderer,
    pixel_signature,
    sha256_file,
    strip_math_wrappers,
)
from ocr_edr.loop import Observation


def evaluate(
    predictions: list[dict], references: list[dict], root: Path, renderer=None
) -> tuple[list, list]:
    refs = {ref["sample_id"]: ref for ref in references}
    if len(refs) != len(references):
        raise ValueError("Duplicate reference sample IDs")
    if not refs or not predictions:
        raise ValueError("Empty reference or prediction coverage")
    renderer = renderer or MathTextRenderer(root / "evaluation_renders")
    rows, groups = [], defaultdict(list)
    seen = set()
    for result in predictions:
        key = (result["sample_id"], result["arm"])
        if key in seen:
            raise ValueError("Duplicate prediction per sample/arm")
        seen.add(key)
        if result["sample_id"] not in refs:
            raise ValueError("Unexpected prediction sample ID")
        ref = refs[result["sample_id"]]
        if result.get("family_id", ref["family_id"]) != ref["family_id"]:
            raise ValueError("Prediction/reference family mismatch")
        signatures = []
        errors = []
        for markup in [ref["reference"], result["initial_prediction"], result["final_prediction"]]:
            try:
                render = renderer.render(Observation(result["sample_id"], "formula", "", markup))
                signatures.append(pixel_signature(Path(render.path)))
                errors.append(None)
            except Exception as exc:
                signatures.append(None)
                errors.append(type(exc).__name__)
        if signatures[0] is None:
            raise ValueError("Reference must render for this diagnostic")
        row = {
            "sample_id": result["sample_id"],
            "family_id": ref["family_id"],
            "variant": ref["variant"],
            "source_kind": ref["source_kind"],
            "error_type": ref["error_type"],
            "arm": result["arm"],
            "initial_raster_exact_proxy": signatures[1] == signatures[0],
            "final_raster_exact_proxy": signatures[2] == signatures[0],
            "initial_normalized_exact": strip_math_wrappers(result["initial_prediction"])
            == ref["reference"],
            "final_normalized_exact": strip_math_wrappers(result["final_prediction"])
            == ref["reference"],
            "changed": result["initial_prediction"] != result["final_prediction"],
            "syntax_failures": sum(call["render_error"] is not None for call in result["trace"]),
            "wrapper_contract_violations": sum(
                call["extraction"] != "latex_tags" for call in result["trace"]
            ),
            "generation_seconds": sum(call["generation_seconds"] for call in result["trace"]),
            "input_tokens": sum(call["input_tokens"] for call in result["trace"]),
            "output_tokens": sum(call["output_tokens"] for call in result["trace"]),
            "final_render_error": errors[2],
        }
        rows.append(row)
        groups[result["arm"]].append(row)
    arm_ids = [{row["sample_id"] for row in group} for group in groups.values()]
    if any(ids != set(refs) for ids in arm_ids):
        raise ValueError("Unpaired arm/reference coverage; do not compare partial cases")
    summary = []
    for arm, group in groups.items():
        bad = [r for r in group if not r["initial_raster_exact_proxy"]]
        good = [r for r in group if r["initial_raster_exact_proxy"]]
        correct = [r for r in good if r["variant"] == "correct"]
        equivalent = [r for r in good if r["variant"] == "equivalent"]
        summary.append(
            {
                "arm": arm,
                "cases": len(group),
                "families": len({r["family_id"] for r in group}),
                "bad_n": len(bad),
                "good_n": len(good),
                "bad_fixed_proxy": sum(r["final_raster_exact_proxy"] for r in bad),
                "good_regressions_proxy": sum(not r["final_raster_exact_proxy"] for r in good),
                "correct_regressions_proxy": sum(
                    not r["final_raster_exact_proxy"] for r in correct
                ),
                "equivalent_regressions_proxy": sum(
                    not r["final_raster_exact_proxy"] for r in equivalent
                ),
                "counterfactual_fixed_proxy": sum(
                    r["final_raster_exact_proxy"]
                    for r in bad
                    if r["source_kind"] == "counterfactual"
                ),
                "counterfactual_good_regressions_proxy": sum(
                    not r["final_raster_exact_proxy"]
                    for r in good
                    if r["source_kind"] == "counterfactual"
                ),
                "full_set_raster_exact_proxy": mean(r["final_raster_exact_proxy"] for r in group),
                "normalized_exact_rate": mean(r["final_normalized_exact"] for r in group),
                "syntax_failures": sum(r["syntax_failures"] for r in group),
                "wrapper_contract_violations": sum(r["wrapper_contract_violations"] for r in group),
                "mean_generation_seconds": mean(r["generation_seconds"] for r in group),
                "mean_input_tokens": mean(r["input_tokens"] for r in group),
                "mean_output_tokens": mean(r["output_tokens"] for r in group),
            }
        )
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--renderer", choices=["mathtext", "tectonic"], default="mathtext")
    args = parser.parse_args()
    root = args.run.resolve()

    def read_rows(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    renderer = None
    if args.renderer == "tectonic":
        from ocr_edr.tex import TectonicRenderer

        renderer = TectonicRenderer(root / "evaluation_renders")
    rows, summary = evaluate(
        read_rows(root / "predictions.jsonl"), read_rows(args.references), root, renderer
    )
    (root / "evaluation.json").write_text(
        json.dumps(
            {
                "metric": "same_renderer_exact_raster_proxy",
                "benchmark_evaluation": False,
                "reference_sha256": sha256_file(args.references),
                "prediction_sha256": sha256_file(root / "predictions.jsonl"),
                "summary": summary,
                "cases": rows,
            },
            indent=2,
        )
        + "\n"
    )
    with (root / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
