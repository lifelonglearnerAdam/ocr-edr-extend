#!/usr/bin/env python3
"""Offline table scores using verified upstream TEDS/normalization, with fixed pairs."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.formula_pilot import sha256_file
from ocr_edr.official_tables import (
    load_official_table_normalizer,
    load_official_teds,
    verify_official_table_sources,
)
from ocr_edr.table_evaluation import evaluate_tables


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--official-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory")
    run = json.loads((args.run / "run.json").read_text())
    if run["status"] != "completed":
        raise ValueError("Evaluate only a completed frozen run")
    if run["input_sha256"] != sha256_file(args.inputs):
        raise ValueError("Frozen input hash differs from the inference receipt")
    dataset = json.loads((args.inputs.parent / "dataset.json").read_text())
    if dataset["reference_sha256"] != sha256_file(args.references):
        raise ValueError("Frozen reference hash mismatch")
    official_root = args.official_root.resolve()
    source_receipt = verify_official_table_sources(official_root, dataset["revision"])
    normalize = load_official_table_normalizer(official_root)
    teds_class = load_official_teds(official_root)
    content, structure = teds_class(), teds_class(structure_only=True)

    def score(prediction, reference):
        return {
            "teds": content.evaluate(prediction, reference),
            "teds_structure": structure.evaluate(prediction, reference),
        }

    predictions_path = args.run / "predictions.jsonl"
    evaluation = evaluate_tables(
        read_rows(predictions_path),
        read_rows(args.inputs),
        read_rows(args.references),
        run["arms"],
        normalize=normalize,
        score=score,
    )
    if sum(c["model_calls"] for c in evaluation["cases"]) != run["calls"]:
        raise ValueError("Model-call count differs from the inference receipt")
    if len(read_rows(args.inputs)) != run["cases"]:
        raise ValueError("Case count differs from the inference receipt")
    evaluation.update(
        {
            "metric": "unmodified official OmniDocBench TEDS and TEDS-S",
            "normalization": "table_content_post_process(normalized_html_table(markup)) for reference/initial/final",
            "benchmark_evaluation": False,
            "pairing": "fixed sample/page/annotation; all cases retained; no end-to-end rematching",
            "matching_label": "released-reference metric agreement; not proof of visual correctness",
            "aggregation": "cases and mean of within-page means; repeated variants are correlated",
            "inference_run_sha256": sha256_file(args.run / "run.json"),
            "prediction_sha256": sha256_file(predictions_path),
            "input_sha256": sha256_file(args.inputs),
            "reference_sha256": sha256_file(args.references),
            "official_source": source_receipt,
            "versions": {
                p: importlib.metadata.version(p)
                for p in ["lxml", "apted", "Levenshtein", "beautifulsoup4", "pylatexenc"]
            },
        }
    )
    args.output.mkdir(parents=True)
    (args.output / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n"
    )
    for name, rows in [("summary", evaluation["summary"]), ("per_page", evaluation["per_page"])]:
        with (args.output / (name + ".csv")).open("w") as sink:
            writer = csv.DictWriter(sink, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps([r for r in evaluation["summary"] if r["variant"] == "all"], indent=2))


if __name__ == "__main__":
    main()
