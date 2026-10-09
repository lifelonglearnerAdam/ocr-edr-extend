#!/usr/bin/env python3
"""Offline full-cohort diagnosis/refinement scoring after every call is frozen."""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import _bootstrap  # noqa: F401
from run_table_diagnosis import load_cohort

from ocr_edr.native_table_repair import verify_native_calls
from ocr_edr.official_tables import (
    load_official_table_normalizer,
    load_official_teds,
    verify_official_table_sources,
)
from ocr_edr.sft import sha256
from ocr_edr.table_diagnosis import verify_diagnosis_binding, verify_guided_runtime
from ocr_edr.table_evaluation import evaluate_tables
from ocr_edr.table_pilot import HTMLTableRenderer
from ocr_edr.table_sft_screen import adapt_table_call


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def interval(values, seed=20261007):
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(10000))
    return [draws[249], draws[9749]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["dataset", "study", "native-run", "prompt-run", "official-root", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    packets = read(args.study / "packets/run.json")
    if packets["status"] != "completed":
        raise ValueError("Incomplete diagnosis packets")
    for cohort, pins in packets["diagnosis_evidence_sha256"].items():
        folder = args.study / "diagnoses" / cohort
        run = read(folder / "run.json")
        if (
            sha256(folder / "run.json") != pins["receipt"]
            or sha256(folder / "calls.jsonl") != pins["calls"]
            or run["status"] != "completed"
            or run["completed_calls"] != (103 if cohort == "controlled" else 32)
        ):
            raise ValueError("Generating full diagnosis evidence changed")
        verify_native_calls(
            rows(folder / "calls.jsonl"), load_cohort(args.dataset, cohort, args.native_run), "all"
        )
    for key, n in [
        ("controlled-learned", 103),
        ("controlled-displaced_region", 103),
        ("controlled-oracle_controlled", 103),
        ("native-learned", 32),
    ]:
        folder = args.study / "refinement" / key
        run = read(folder / "run.json")
        if (
            run["status"] != "completed"
            or run["completed_calls"] != n
            or sha256(folder / "calls.jsonl") != run["calls_sha256"]
        ):
            raise ValueError("All341 refinement outputs must be frozen before evaluation")
    gold = read(args.study / "packets/diagnosis-scores.json")
    if sha256(args.study / "packets/diagnosis-scores.json") != packets["diagnosis_scores_sha256"]:
        raise ValueError("Diagnosis scores changed")
    official = verify_official_table_sources(
        args.official_root, "f133a71e9e91c3621c7ce8994200a7b394a06eb3"
    )
    normalize = load_official_table_normalizer(args.official_root)
    metric = load_official_teds(args.official_root)
    teds, structure = metric(), metric(structure_only=True)
    seals = {}
    all_reports = {}
    cost = []
    for cohort, conditions in [
        ("controlled", ["learned", "displaced_region", "oracle_controlled"]),
        ("native", ["learned"]),
    ]:
        inputs = load_cohort(args.dataset, cohort, args.native_run)
        predictions = []
        # Verify original evaluation, generating calls and checkpoint identities.
        original = (
            args.prompt_run / "descriptive_schema" if cohort == "controlled" else args.native_run
        )
        original_eval = original / "evaluation"
        baseline_receipt = read(original_eval / "run.json")
        if sha256(original_eval / "evaluation.json") != baseline_receipt[
            "evaluation_sha256"
        ] or sha256(original_eval / "predictions.jsonl") != baseline_receipt.get(
            "prediction_sha256", baseline_receipt.get("predictions_sha256")
        ):
            raise ValueError("Frozen no-diagnosis evaluation changed")
        original_folder = original / "inference/all"
        original_run = read(original_folder / "run.json")
        if cohort == "native":
            audit = read(args.native_run / "audit.json")
            if (
                sha256(original_folder / "run.json") != audit["evidence_sha256"]["all_receipt"]
                or audit["status"] != "audit_completed"
            ):
                raise ValueError("Historical unhinted native audit binding differs")
            original_run = {
                **original_run,
                "model_receipt_sha256": audit["current_model_receipt_sha256"],
            }
        if (
            original_run["status"] != "completed"
            or original_run["completed_calls"] != len(inputs)
            or sha256(original_folder / "calls.jsonl") != original_run["calls_sha256"]
        ):
            raise ValueError("Frozen no-diagnosis calls changed")
        verify_native_calls(rows(original_folder / "calls.jsonl"), inputs, "all")
        for r in rows(original_eval / "predictions.jsonl"):
            if r["arm"] == "all":
                predictions.append({**r, "arm": "without_diagnosis"})
        predictions.extend(
            {
                "sample_id": r["sample_id"],
                "family_id": r["family_id"],
                "arm": "unchanged_0",
                "initial_prediction": r["prediction"],
                "final_prediction": r["prediction"],
                "trace": [],
            }
            for r in inputs
        )
        renderer = HTMLTableRenderer(out / (cohort + "-renders"))
        for condition in conditions:
            folder = args.study / "refinement" / (cohort + "-" + condition)
            run = read(folder / "run.json")
            packet_file = args.study / "packets" / (cohort + "-" + condition + ".jsonl")
            if (
                run["status"] != "completed"
                or run["completed_calls"] != len(inputs)
                or sha256(folder / "calls.jsonl") != run["calls_sha256"]
                or sha256(packet_file) != packets["packets"][packet_file.name]["sha256"]
                or run["packet_sha256"] != sha256(packet_file)
                or run["adapter_sha256"] != original_run["adapter_sha256"]
                or run["model_receipt_sha256"]
                != original_run.get(
                    "model_receipt_sha256",
                    read(args.study / "diagnoses" / cohort / "run.json")["model_receipt_sha256"],
                )
            ):
                raise ValueError("Full matched guided refiner calls/checkpoint required")
            calls = rows(folder / "calls.jsonl")
            verify_guided_runtime(run, original_run)
            verify_native_calls(calls, inputs, "all")
            hints = rows(packet_file)
            for item, call, hint in zip(inputs, calls, hints):
                if call["diagnosis"] != hint["diagnosis"]:
                    raise ValueError("Refiner diagnosis differs from sealed packet")
                verify_diagnosis_binding(item, call["diagnosis"])
                result = adapt_table_call(
                    item,
                    call,
                    renderer=renderer.render,
                    max_new_tokens=192,
                    diagnosis=call["diagnosis"],
                )
                result["arm"] = condition
                predictions.append(result)
            seals[cohort + "-" + condition] = {
                "receipt": sha256(folder / "run.json"),
                "calls": sha256(folder / "calls.jsonl"),
            }
        # All four generation folders were completed by the controller before this command.
        references = rows(args.dataset / "model_dev-references.jsonl")
        metadata = read(args.dataset / "dataset.json")
        if (
            sha256(args.dataset / "model_dev-references.jsonl")
            != metadata["file_sha256"]["model_dev-references.jsonl"]
        ):
            raise ValueError("Frozen model-dev reference drift")
        if any(r["role"] != "model_dev" for r in references):
            raise ValueError("Invalid reference role")
        if cohort == "controlled":
            ref = [
                {
                    "sample_id": r["sample_id"],
                    "family_id": r["family_id"],
                    "parent_page": r["document_id"],
                    "reference": r["reference"],
                    "variant": r["variant"],
                    "annotation_id": None,
                    "gt_position": None,
                }
                for r in references
            ]
        else:
            index = {r["family_id"]: (r["document_id"], r["reference"]) for r in references}
            ref = [
                {
                    "sample_id": r["sample_id"],
                    "family_id": r["family_id"],
                    "parent_page": index[r["family_id"]][0],
                    "reference": index[r["family_id"]][1],
                    "variant": "native",
                    "annotation_id": None,
                    "gt_position": None,
                }
                for r in inputs
            ]
        report = evaluate_tables(
            predictions,
            inputs,
            ref,
            ["unchanged_0", "without_diagnosis", *conditions],
            normalize=normalize,
            score=lambda p, r: {
                "teds": teds.evaluate(p, r),
                "teds_structure": structure.evaluate(p, r),
            },
        )
        all_reports[cohort] = report
        (out / (cohort + "-evaluation.json")).write_text(json.dumps(report, indent=2) + "\n")
        (out / (cohort + "-predictions.jsonl")).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in predictions)
        )
        base = {
            r["parent_page"]: r["final_teds"]
            for r in report["per_page"]
            if r["arm"] == "without_diagnosis" and r["variant"] == "all"
        }
        contrast = []
        for condition in conditions:
            page = [
                r for r in report["per_page"] if r["arm"] == condition and r["variant"] == "all"
            ]
            delta = [r["final_teds"] - base[r["parent_page"]] for r in page]
            contrast.append(
                {
                    "arm": condition,
                    "comparator": "without_diagnosis",
                    "documents": len(delta),
                    "document_mean_delta_teds": mean(delta),
                    "descriptive_percentile95": interval(delta),
                }
            )
        (out / (cohort + "-contrasts.json")).write_text(json.dumps(contrast, indent=2) + "\n")
    summary = []
    for variant in ["all", *sorted({r["variant"] for r in gold})]:
        group = [r for r in gold if variant == "all" or r["variant"] == variant]
        summary.append(
            {
                "variant": variant,
                "cases": len(group),
                "valid_outputs": sum(r["valid_output"] for r in group),
                "verdict_correct": sum(r["verdict_correct"] for r in group),
                "strict_joint": sum(r["strict_joint"] for r in group),
                "valid_false_positive": sum(
                    r["target"]["verdict"] == "valid"
                    and (r["prediction"] or {}).get("verdict") == "invalid"
                    for r in group
                ),
                "invalid_missed": sum(
                    r["target"]["verdict"] == "invalid"
                    and (r["prediction"] or {}).get("verdict") == "valid"
                    for r in group
                ),
            }
        )
    for cohort in ["controlled", "native"]:
        calls = rows(args.study / "diagnoses" / cohort / "calls.jsonl")
        cost.append(
            {
                "component": "diagnosis",
                "cohort": cohort,
                "calls": len(calls),
                "input_tokens": sum(r["input_tokens"] for r in calls),
                "output_tokens": sum(r["output_tokens"] for r in calls),
                "generation_seconds": sum(r["generation_seconds"] for r in calls),
            }
        )
    receipt = {
        "status": "completed",
        "reference_access": "offline_model_dev_only",
        "calibration_locked_access": False,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "diagnosis_summary": summary,
        "component_cost": cost,
        "refinement_seals": seals,
        "official_source": official,
        "evaluation_sha256": {
            cohort: sha256(out / (cohort + "-evaluation.json")) for cohort in all_reports
        },
        "packet_receipt_sha256": sha256(args.study / "packets/run.json"),
        "diagnosis_gold_scope": "controlled construction labels, not independent visual gold",
        "oracle_scope": "label-assisted class/location upper bound, no corrected content",
        "limits": [
            "single-seed post-hoc inspected model-dev",
            "same base/weak training sources means correlated diagnosis/refiner errors",
            "no independently calibrated acceptance or Jev/RL claim",
        ],
    }
    (out / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for cohort, report in all_reports.items():
        print(
            cohort,
            json.dumps(
                [
                    {
                        k: r[k]
                        for k in [
                            "arm",
                            "cases",
                            "case_mean_final_teds",
                            "teds_nonmatching_fixed",
                            "teds_matching_regressions",
                        ]
                    }
                    for r in report["summary"]
                    if r["variant"] == "all"
                ]
            ),
        )
    print("diagnosis", json.dumps(summary))


if __name__ == "__main__":
    main()
