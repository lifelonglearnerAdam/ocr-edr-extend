#!/usr/bin/env python3
"""Offline core CDM scores for all frozen formula pairs; never model inference."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.cdm_evaluation import evaluate_cdm
from ocr_edr.formula_pilot import sha256_file
from ocr_edr.official_cdm import load_official_cdm

DEFAULT_REVISION = "f133a71e9e91c3621c7ce8994200a7b394a06eb3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-env", type=Path)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory")
    if args.runtime_env:
        values = json.loads(args.runtime_env.read_text())
        if not isinstance(values, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in values.items()
        ):
            raise ValueError("Runtime environment must contain string values")
        os.environ.update(values)
    predictions = [
        json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()
    ]
    references = [
        json.loads(line) for line in args.references.read_text().splitlines() if line.strip()
    ]
    arms = list(dict.fromkeys(r["arm"] for r in predictions))
    upstream, source_receipt = load_official_cdm(args.official_root, args.revision)
    import numpy as np

    def metric(reference, prediction):
        np.random.seed(args.seed)
        values = upstream(
            reference, prediction, tmp_dir=str(args.tmp_dir.resolve()), save_vis=False
        )
        return values

    control_refs = [
        {"sample_id": str(i), "family_id": "control", "reference": "x^2+1"} for i in range(4)
    ]
    control_predictions = [
        {
            "sample_id": str(i),
            "family_id": "control",
            "arm": "control",
            "initial_prediction": "x^2+1",
            "final_prediction": pred,
        }
        for i, pred in enumerate(["x^2+1", "x^{2}+1", "x^3+1", r"\frac{"])
    ]
    controls = evaluate_cdm(control_predictions, control_refs, ["control"], metric=metric)
    scores = [r["final_cdm"]["F1_score"] for r in controls["cases"]]
    if scores[:2] != [1.0, 1.0] or not 0 < scores[2] < 1 or scores[3] != 0:
        raise ValueError(f"Official CDM control readiness failed: {scores}")
    print(f"CDM controls ready: {scores}; scoring {len(predictions)} frozen pairs", flush=True)
    args.output.mkdir(parents=True)
    calls_path = args.output / "metric_calls.jsonl"

    def recorded_metric(reference, prediction):
        start = time.monotonic()
        values = metric(reference, prediction)
        row = {
            "reference": reference,
            "prediction": prediction,
            "metrics": values,
            "seconds": time.monotonic() - start,
        }
        with calls_path.open("a") as sink:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Scored unique pair: F1={values['F1_score']:.4f}", flush=True)
        return values

    def runtime_version(command):
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT, timeout=15
        ).strip()

    repo_root = Path(__file__).resolve().parents[1]
    receipt = {
        "status": "running",
        "metric": "unmodified official OmniDocBench cdm_metrics core callable",
        "benchmark_evaluation": False,
        "pairing": "fixed sample/family pairs; all arms must cover every reference",
        "normalization": "exact frozen strings; only unmodified upstream internal token preprocessing",
        "failure_policy": "reference selfcheck must render and score 1; failed/empty predictions retained with zero",
        "new_model_calls": 0,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip(),
        "source_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo_root, text=True
            ).strip()
        ),
        "random_seed_per_metric_call": args.seed,
        "official_source": source_receipt,
        "prediction_sha256": sha256_file(args.predictions),
        "reference_sha256": sha256_file(args.references),
        "versions": {
            p: importlib.metadata.version(p) for p in ["numpy", "scipy", "Pillow", "pylatexenc"]
        },
        "runtime": {
            "pdflatex": runtime_version([os.environ.get("CDM_PDFLATEX", "pdflatex"), "--version"]),
            "magick": runtime_version(["magick", "-version"]),
            "ghostscript": runtime_version(["gs", "--version"]),
            "temporary_directory": str(args.tmp_dir.resolve()),
            "runtime_env_sha256": sha256_file(args.runtime_env) if args.runtime_env else None,
        },
        "source_hashes": {
            str(p.relative_to(repo_root)): sha256_file(p)
            for p in [
                Path(__file__).resolve(),
                repo_root / "src/ocr_edr/official_cdm.py",
                repo_root / "src/ocr_edr/cdm_evaluation.py",
            ]
        },
        "controls": controls,
    }
    (args.output / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        result = evaluate_cdm(predictions, references, arms, metric=recorded_metric)
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        (args.output / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
        raise
    receipt.update(
        status="completed",
        case_records=len(predictions),
        reference_cases=len(references),
        arms=arms,
    )
    (args.output / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    result.update({k: v for k, v in receipt.items() if k != "status"})
    (args.output / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    with (args.output / "summary.csv").open("w") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(result["summary"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["summary"])
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
