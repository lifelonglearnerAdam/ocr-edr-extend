#!/usr/bin/env python3
"""Replay complete frozen calls, then score all model-dev cases with official TEDS."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.official_tables import (
    load_official_table_normalizer,
    load_official_teds,
    verify_official_table_sources,
)
from ocr_edr.sft import sha256
from ocr_edr.table_pilot import HTMLTableRenderer
from ocr_edr.table_screen_evaluation import evaluate_table_screen
from ocr_edr.table_sft_screen import adapt_table_call, load_table_screen_inputs


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--official-revision", default="f133a71e9e91c3621c7ce8994200a7b394a06eb3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = load_table_screen_inputs(args.dataset)
    by_input = {case["sample_id"]: case for case in cases}
    dataset = json.loads((args.dataset / "dataset.json").read_text())
    reference_path = args.dataset / "model_dev-references.jsonl"
    if sha256(reference_path) != dataset["file_sha256"][reference_path.name]:
        raise ValueError("Frozen model-dev reference hash mismatch")
    references = read_rows(reference_path)
    source = verify_official_table_sources(args.official_root.resolve(), args.official_revision)
    normalize = load_official_table_normalizer(args.official_root.resolve())
    metric = load_official_teds(args.official_root.resolve())
    teds, teds_s = metric(), metric(structure_only=True)
    runs, arms, cohort = [], ["unchanged_0"], None
    sft_cohort = None
    for run_root in args.runs:
        run_path, calls_path = run_root / "run.json", run_root / "calls.jsonl"
        run = json.loads(run_path.read_text())
        if (
            run["status"] != "completed"
            or run["cases"] != len(cases)
            or run["completed_calls"] != len(cases)
            or run["input_sha256"] != sha256(args.dataset / "model_dev-inputs.jsonl")
            or run["dataset_sha256"] != sha256(args.dataset / "dataset.json")
            or run["calls_sha256"] != sha256(calls_path)
            or run["reference_access"] != "none"
            or run["calibration_locked_images_loaded"] is not False
        ):
            raise ValueError("Inference completion/coverage/provenance mismatch")
        signature = {
            k: run[k]
            for k in [
                "config_sha256",
                "model_receipt_sha256",
                "device",
                "dtype",
                "attention",
                "cpu_threads",
                "source_sha256",
            ]
        }
        signature["versions"] = {n: run["versions"][n] for n in ["torch", "transformers", "Pillow"]}
        signature["prompt_format"] = run.get("prompt_format", "descriptive_schema")
        signature["precision_profile"] = run.get("precision_profile", "bf16_lora")
        signature["logits_projection"] = run.get("logits_projection", "full")
        if signature["precision_profile"] == "nf4_lora_8gb":
            if not run.get("quantization") or not all(
                run["versions"].get(n) for n in ["bitsandbytes", "peft"]
            ):
                raise ValueError("NF4 pairing requires quantization and dependency provenance")
            signature["quantization"] = run["quantization"]
            signature["versions"].update({n: run["versions"][n] for n in ["bitsandbytes", "peft"]})
        if run["device"].startswith("cuda"):
            for field in ["gpu", "cuda_runtime"]:
                if not run.get(field):
                    raise ValueError("GPU inference receipt is missing hardware/runtime identity")
                signature[field] = run[field]
        if cohort is not None and cohort != signature:
            raise ValueError("Paired conditions require the same model/protocol/runtime/source")
        cohort = signature
        condition = run["condition"]
        if condition not in {"base", "all", "no_explicit_preservation"} or condition in arms:
            raise ValueError("Unexpected or duplicate inference condition")
        if condition != "base":
            admission_hash, peft_version = run.get("admission_sha256"), run["versions"].get("peft")
            if not admission_hash or not peft_version:
                raise ValueError("SFT pairing requires admission and PEFT provenance")
            sft_signature = {"admission_sha256": admission_hash, "peft": peft_version}
            if sft_cohort is not None and sft_cohort != sft_signature:
                raise ValueError("SFT conditions differ in admission snapshot or PEFT runtime")
            sft_cohort = sft_signature
        calls = read_rows(calls_path)
        if (
            len(calls) != len(cases)
            or {c["sample_id"] for c in calls} != set(by_input)
            or any(c["condition"] != condition for c in calls)
        ):
            raise ValueError("Frozen call identity/condition mismatch")
        arms.append(condition)
        runs.append((run, calls, sha256(run_path)))
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "input_sha256": sha256(args.dataset / "model_dev-inputs.jsonl"),
        "reference_sha256": sha256(reference_path),
        "arms": arms,
        "inference_run_sha256": {run["condition"]: digest for run, _, digest in runs},
        "official_source": source,
        "cohort": cohort,
        "sft_cohort": sft_cohort,
        "source_sha256": {
            str(p): sha256(p)
            for p in [
                Path(__file__),
                Path(__file__).resolve().parents[1] / "src/ocr_edr/table_screen_evaluation.py",
                Path(__file__).resolve().parents[1] / "src/ocr_edr/table_sft_screen.py",
                Path(__file__).resolve().parents[1] / "src/ocr_edr/table_evaluation.py",
            ]
        },
        "benchmark_evaluation": False,
        "metric": "unmodified official OmniDocBench TEDS and TEDS-S",
        "label_scope": "published weak-reference agreement; not source-visual correctness",
        "aggregation": "all cases and means within original PMC documents; one source per document",
        "action_scope": "controlled single edits; address and exact action agreement scored offline only",
        "gate": "strict JSON/cap/render fallback only; no independent visual acceptance",
        "timing_scope": "generation excludes model loading and image/prompt preprocessing; render replay uses a shared content cache and is reported separately; not end-to-end deployment latency",
        "versions": {},
    }
    (root / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        metadata["versions"] = {
            n: importlib.metadata.version(n)
            for n in [
                "lxml",
                "apted",
                "Levenshtein",
                "beautifulsoup4",
                "pylatexenc",
                "weasyprint",
                "Pillow",
            ]
        }
        renderer = HTMLTableRenderer(root / "renders")
        metadata["fonts"] = renderer.fonts
        predictions = [
            {
                "sample_id": c["sample_id"],
                "family_id": c["family_id"],
                "arm": "unchanged_0",
                "initial_prediction": c["prediction"],
                "final_prediction": c["prediction"],
                "trace": [],
            }
            for c in cases
        ]
        for run, calls, _ in runs:
            for call in calls:
                predictions.append(
                    adapt_table_call(
                        by_input[call["sample_id"]],
                        call,
                        renderer=renderer.render,
                        max_new_tokens=run["config"]["inference"]["max_new_tokens"],
                        prompt_format=run.get("prompt_format", "descriptive_schema"),
                    )
                )
        predictions_path = root / "predictions.jsonl"
        predictions_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in predictions)
        )
        evaluation = evaluate_table_screen(
            predictions,
            cases,
            references,
            arms,
            normalize=normalize,
            score=lambda p, r: {
                "teds": teds.evaluate(p, r),
                "teds_structure": teds_s.evaluate(p, r),
            },
        )
        if sum(row["model_calls"] for row in evaluation["cases"]) != len(cases) * len(runs):
            raise ValueError("Scored model-call accounting mismatch")
        (root / "evaluation.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n"
        )
        for name in ["summary", "per_document"]:
            with (root / (name + ".csv")).open("w") as sink:
                writer = csv.DictWriter(
                    sink, fieldnames=list(evaluation[name][0]), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(evaluation[name])
        metadata.update(
            status="completed",
            prediction_sha256=sha256(predictions_path),
            evaluation_sha256=sha256(root / "evaluation.json"),
        )
        print(json.dumps([r for r in evaluation["summary"] if r["variant"] == "all"], indent=2))
    except Exception as error:
        metadata.update(status="failed", error_type=type(error).__name__, error=str(error)[:1000])
        raise
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        (root / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
