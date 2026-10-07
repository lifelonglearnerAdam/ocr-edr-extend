#!/usr/bin/env python3
"""Add paired original-document intervals to a complete frozen table evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.sft import sha256
from ocr_edr.table_statistics import ARMS, summarize_table_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = args.evaluation_dir
    receipt = json.loads((folder / "run.json").read_text())
    evaluation = folder / "evaluation.json"
    predictions = folder / "predictions.jsonl"
    if (
        receipt["status"] != "completed"
        or receipt["arms"] != list(ARMS)
        or sha256(evaluation) != receipt["evaluation_sha256"]
        or sha256(predictions) != receipt["prediction_sha256"]
    ):
        raise ValueError("Complete same-cohort four-arm evaluation with frozen hashes required")
    records = json.loads(evaluation.read_text())["cases"]
    predicted = [json.loads(line) for line in predictions.read_text().splitlines()]
    pairs = {(r["arm"], r["sample_id"]) for r in predicted}
    if len(pairs) != len(predicted) or pairs != {(r["arm"], r["sample_id"]) for r in records}:
        raise ValueError("Complete evaluated prediction/case correspondence required")
    report = summarize_table_statistics(records)
    if report["documents"] != 32 or report["cases_per_arm"] != 103:
        raise ValueError("This frozen NF4 report requires all 32 documents/103 cases")
    report.update(
        created_at=datetime.now(timezone.utc).isoformat(),
        evaluation_run_sha256=sha256(folder / "run.json"),
        evaluation_sha256=sha256(evaluation),
        predictions_sha256=sha256(predictions),
        source_sha256={
            "scripts/summarize_table_sft_screen.py": sha256(Path(__file__)),
            "src/ocr_edr/table_statistics.py": sha256(
                Path(__file__).resolve().parents[1] / "src/ocr_edr/table_statistics.py"
            ),
        },
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for name in ["contrasts", "metric_proxy_utility", "costs"]:
        with (output / (name + ".csv")).open("w") as sink:
            writer = csv.DictWriter(sink, fieldnames=list(report[name][0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(report[name])
    print(json.dumps([r for r in report["contrasts"] if r["variant"] == "all"], indent=2))


if __name__ == "__main__":
    main()
