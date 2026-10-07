#!/usr/bin/env python3
"""Offline official TEDS and current-editor invariants for complete native outputs."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.native_tables import (
    evaluate_native_tables,
    load_native_table_sources,
    native_editor_profile,
)
from ocr_edr.official_tables import (
    load_official_table_normalizer,
    load_official_teds,
    verify_official_table_sources,
)
from ocr_edr.sft import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["dataset", "run", "official-root", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    inputs = load_native_table_sources(args.dataset)
    run_path = args.run / "run.json"
    run = json.loads(run_path.read_text())
    predictions_path = args.run / "predictions.jsonl"
    if (
        run["status"] != "completed"
        or run["completed_sources"] != len(inputs)
        or run["cases"] != len(inputs)
        or run["reference_access"] != "none"
        or run["calibration_locked_images_loaded"] is not False
        or run["dataset_sha256"] != sha256(args.dataset / "dataset.json")
        or run["input_sha256"] != sha256(args.dataset / "model_dev-source-inputs.jsonl")
        or run["predictions_sha256"] != sha256(predictions_path)
    ):
        raise ValueError("Frozen native inference coverage/provenance mismatch")
    dataset = json.loads((args.dataset / "dataset.json").read_text())
    ref_path = args.dataset / "model_dev-references.jsonl"
    if sha256(ref_path) != dataset["file_sha256"][ref_path.name]:
        raise ValueError("Frozen native reference hash mismatch")
    refs = [json.loads(line) for line in ref_path.read_text().splitlines()]
    predictions = [json.loads(line) for line in predictions_path.read_text().splitlines()]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "initializing",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "inference_run_sha256": sha256(run_path),
        "reference_sha256": sha256(ref_path),
        "predictions_sha256": sha256(predictions_path),
        "source_sha256": {
            str(p): sha256(p)
            for p in [
                Path(__file__),
                Path(__file__).resolve().parents[1] / "src/ocr_edr/native_tables.py",
                Path(__file__).resolve().parents[1] / "src/ocr_edr/table_pilot.py",
                Path(__file__).resolve().parents[1] / "src/ocr_edr/official_tables.py",
            ]
        },
        "metric": "unmodified official OmniDocBench TEDS and TEDS-S; symmetric HTML normalization",
        "parser_failure_policy": "zero scores with source/call/time retained; no row removal",
        "label_scope": "released weak-reference agreement, not visual correctness",
        "editor_scope": "raw DOM counts and aligned text/span comparison; not exhaustive TEDS action oracle",
        "parser_pretraining_overlap": "PubTabNet advertised; current dev sources derive from public train",
        "benchmark_evaluation": False,
    }
    (root / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        official = args.official_root.resolve()
        receipt["official_source"] = verify_official_table_sources(
            official, "f133a71e9e91c3621c7ce8994200a7b394a06eb3"
        )
        normalize = load_official_table_normalizer(official)
        metric = load_official_teds(official)
        content, structure = metric(), metric(structure_only=True)
        report = evaluate_native_tables(
            predictions,
            inputs,
            refs,
            normalize=normalize,
            score=lambda p, r: {
                "teds": content.evaluate(p, r),
                "teds_structure": structure.evaluate(p, r),
            },
        )
        by_ref = {r["family_id"]: r for r in refs}
        profiles = []
        for row in predictions:
            profile = (
                {"editor_input_valid": False, "editor_input_error": "parser_failure"}
                if row["parser_error"]
                else native_editor_profile(row["prediction"], by_ref[row["family_id"]]["reference"])
            )
            profiles.append({"family_id": row["family_id"], **profile})
        if report["summary"]["parser_failures"] != run["parser_failures"]:
            raise ValueError("Parser failure count drift")
        report["editor_summary"] = {
            "editor_valid_sources": sum(p["editor_input_valid"] for p in profiles),
            "reference_dom_row_deficit": sum(
                p.get("requires_row_insertion_for_reference_dom", False) for p in profiles
            ),
            "reference_dom_cell_deficit": sum(
                p.get("requires_cell_insertion_for_reference_dom", False) for p in profiles
            ),
            "aligned_multiple_text_mismatches": sum(
                p.get("aligned_plain_text_mismatches") is not None
                and p["aligned_plain_text_mismatches"] > 1
                for p in profiles
            ),
        }
        report["editor_profiles"] = profiles
        (root / "evaluation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        with (root / "per_source.csv").open("w") as sink:
            writer = csv.DictWriter(sink, fieldnames=list(report["cases"][0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(report["cases"])
        receipt["versions"] = {
            n: importlib.metadata.version(n)
            for n in ["lxml", "apted", "Levenshtein", "beautifulsoup4", "pylatexenc"]
        }
        receipt.update(status="completed", evaluation_sha256=sha256(root / "evaluation.json"))
        print(
            json.dumps(
                {"summary": report["summary"], "editor_summary": report["editor_summary"]}, indent=2
            )
        )
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error)[:1000])
        raise
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        (root / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
