#!/usr/bin/env python3
"""Freeze32 native-table calls per checkpoint, then separately score all outputs."""

import argparse
import importlib.metadata
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from ocr_edr.native_table_repair import (
    native_model_identity,
    native_repair_inputs,
    verify_native_calls,
)
from ocr_edr.native_tables import load_native_table_sources
from ocr_edr.official_tables import (
    load_official_table_normalizer,
    load_official_teds,
    verify_official_table_sources,
)
from ocr_edr.qwen import QwenFormulaProposer
from ocr_edr.sft import sha256, verify_model_files
from ocr_edr.table_evaluation import evaluate_tables
from ocr_edr.table_pilot import HTMLTableRenderer
from ocr_edr.table_sft_screen import adapt_table_call, table_messages, validate_table_adapter


def read(p):
    return json.loads(p.read_text())


def rows(p):
    return [json.loads(line) for line in p.read_text().splitlines()]


def save(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["inference", "evaluation"], required=True)
    for name in ["dataset", "native-run", "config", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ["model-path", "model-receipt", "admission", "adapter-run", "official-root"]:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--condition", choices=["base", "all", "no_explicit_preservation"])
    parser.add_argument("--runs", nargs="+", type=Path)
    parser.add_argument(
        "--legacy-provenance",
        type=Path,
        help="Explicit later audit for original runs lacking a historical base receipt hash",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    native = read(args.native_run / "run.json")
    frozen_hash = "6bfe68cd225e3cac1f645849174209725936367b31fe8a91bf7a248c93ccd2f3"
    if (
        native["status"] != "completed"
        or native["cases"] != 32
        or native["reference_access"] != "none"
        or native["predictions_sha256"] != frozen_hash
        or sha256(args.native_run / "predictions.jsonl") != frozen_hash
    ):
        raise ValueError("Expected the complete frozen original-frame native pool")
    sources = load_native_table_sources(args.dataset, role="model_dev")
    inputs = native_repair_inputs(
        rows(args.native_run / "predictions.jsonl"), sources, role="model_dev"
    )
    if len(inputs) != 32:
        raise ValueError("All32 native sources required")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    files = [
        Path(__file__),
        project / "src/ocr_edr/native_table_repair.py",
        project / "src/ocr_edr/qwen.py",
        project / "src/ocr_edr/table_sft_screen.py",
        project / "src/ocr_edr/table_evaluation.py",
    ]
    receipt = {
        "status": "initializing",
        "phase": args.phase,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cases": 32,
        "condition": args.condition,
        "reference_access": "none" if args.phase == "inference" else "offline_model_dev_only",
        "calibration_locked_access": False,
        "native_prediction_sha256": frozen_hash,
        "config_sha256": sha256(args.config),
        "source_sha256": {str(p.relative_to(project)): sha256(p) for p in files},
        "prompt_format": "descriptive_schema",
        "completed_calls": 0,
        "scope": "same inspected development articles; no untouched-source claim",
    }
    save(out / "run.json", receipt)
    try:
        if args.phase == "inference":
            if not args.model_path or not args.model_receipt or not args.condition:
                parser.error("Inference requires model and condition")
            model_receipt = read(args.model_receipt)
            if model_receipt["revision"] != cfg["model_revision"]:
                raise ValueError("Base model revision differs from the frozen protocol")
            verify_model_files(args.model_path, model_receipt)
            receipt["model_receipt_sha256"] = sha256(args.model_receipt)
            checkpoint = None
            if args.condition != "base":
                if not args.adapter_run or not args.admission:
                    parser.error("Adapter/admission required")
                receipt["adapter_sha256"] = validate_table_adapter(
                    args.adapter_run,
                    condition=args.condition,
                    config=cfg,
                    config_sha256=sha256(args.config),
                    admission_sha256=sha256(args.admission / "admission.json"),
                )
                checkpoint = args.adapter_run / "checkpoint"
            elif args.adapter_run:
                raise ValueError("Base must not load adapter")
            receipt["versions"] = {n: importlib.metadata.version(n) for n in cfg["versions"]}
            if receipt["versions"] != cfg["versions"]:
                raise ValueError("Frozen inference runtime differs")
            if subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
            ).strip():
                raise RuntimeError("GPU occupied before native diagnostic")
            for p in files:
                dst = out / "source" / p.relative_to(project)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dst)
            import torch

            torch.manual_seed(cfg["seed"])
            torch.cuda.manual_seed_all(cfg["seed"])
            torch.set_num_threads(4)
            torch.use_deterministic_algorithms(True, warn_only=True)
            proposer = QwenFormulaProposer(
                args.model_path,
                device="cuda:0",
                precision_profile="nf4_lora_8gb",
                adapter_path=checkpoint,
                min_pixels=cfg["min_pixels"],
                max_pixels=cfg["max_pixels"],
                max_new_tokens=192,
            )
            receipt.update(
                status="running",
                gpu=torch.cuda.get_device_name(0),
                cuda_runtime=torch.version.cuda,
                **proposer.precision_metadata,
            )
            save(out / "run.json", receipt)
            with (out / "calls.jsonl").open("w") as log:
                for item in inputs:
                    messages, prompt = table_messages(
                        item["prediction"], prompt_format="descriptive_schema"
                    )
                    call = proposer.generate(
                        [args.dataset / item["source_image"]], messages, prompt
                    )
                    if call["ordered_image_sha256"] != [item["source_sha256"]]:
                        raise ValueError("Image identity changed")
                    log.write(
                        json.dumps(
                            {**item, **call, "condition": args.condition}, ensure_ascii=False
                        )
                        + "\n"
                    )
                    log.flush()
                    receipt["completed_calls"] += 1
                    if receipt["completed_calls"] % 8 == 0:
                        save(out / "run.json", receipt)
                        print(args.condition, receipt["completed_calls"], "/32", flush=True)
            receipt.update(status="completed", calls_sha256=sha256(out / "calls.jsonl"))
        else:
            if not args.runs or not args.official_root:
                parser.error("Evaluation requires3 frozen runs and official metric")
            candidates = []
            conditions = []
            cohort = None
            legacy = read(args.legacy_provenance) if args.legacy_provenance else None
            if legacy:
                if not args.model_receipt:
                    parser.error("Legacy replay also requires the explicit model receipt")
                if (
                    read(args.model_receipt)["revision"] != cfg["model_revision"]
                    or sha256(args.model_receipt) != legacy["current_model_receipt_sha256"]
                ):
                    raise ValueError("Legacy audit model identity differs from the fixed protocol")
                receipt["legacy_provenance_sha256"] = sha256(args.legacy_provenance)
            for folder in args.runs:
                run = read(folder / "run.json")
                if (
                    run["status"] != "completed"
                    or run["completed_calls"] != 32
                    or run["calls_sha256"] != sha256(folder / "calls.jsonl")
                    or run["reference_access"] != "none"
                    or run["config_sha256"] != sha256(args.config)
                    or run["native_prediction_sha256"] != frozen_hash
                    or run["prompt_format"] != "descriptive_schema"
                ):
                    raise ValueError("Incomplete native calls")
                signature = {
                    k: run[k]
                    for k in [
                        "native_prediction_sha256",
                        "config_sha256",
                        "prompt_format",
                        "source_sha256",
                        "versions",
                        "gpu",
                        "cuda_runtime",
                        "quantization",
                    ]
                }
                signature["model_receipt_sha256"] = native_model_identity(
                    run, sha256(folder / "run.json"), legacy
                )
                if cohort is not None and cohort != signature:
                    raise ValueError("Unmatched native runtime/source cohort")
                cohort = signature
                conditions.append(run["condition"])
                calls = rows(folder / "calls.jsonl")
                verify_native_calls(calls, inputs, run["condition"])
                candidates.extend(calls)
            if conditions != ["base", "all", "no_explicit_preservation"]:
                raise ValueError("Fixed complete3 condition order required")
            official = verify_official_table_sources(
                args.official_root, "f133a71e9e91c3621c7ce8994200a7b394a06eb3"
            )
            normalize = load_official_table_normalizer(args.official_root)
            metric = load_official_teds(args.official_root)
            teds, structure = metric(), metric(structure_only=True)
            reference_path = args.dataset / "model_dev-references.jsonl"
            metadata = read(args.dataset / "dataset.json")
            if sha256(reference_path) != metadata["file_sha256"][reference_path.name]:
                raise ValueError("Reference identity drift")
            references = {}
            for r in rows(reference_path):
                if r["role"] != "model_dev":
                    raise ValueError("Heldout reference role rejected")
                identity = (r["document_id"], r["reference"])
                if references.setdefault(r["family_id"], identity) != identity:
                    raise ValueError("Inconsistent document reference")
            refs = [
                {
                    "sample_id": r["sample_id"],
                    "family_id": r["family_id"],
                    "parent_page": references[r["family_id"]][0],
                    "reference": references[r["family_id"]][1],
                    "variant": "native",
                    "annotation_id": None,
                    "gt_position": None,
                }
                for r in inputs
            ]
            for ref in refs:
                normalized = normalize(ref["reference"])
                if (
                    teds.evaluate(normalized, normalized) != 1
                    or structure.evaluate(normalized, normalized) != 1
                ):
                    raise ValueError("Native diagnostic reference readiness failed")
            renderer = HTMLTableRenderer(out / "renders")
            indexed = {r["sample_id"]: r for r in inputs}
            predictions = [
                {
                    "sample_id": r["sample_id"],
                    "family_id": r["family_id"],
                    "arm": "unchanged_0",
                    "initial_prediction": r["prediction"],
                    "final_prediction": r["prediction"],
                    "trace": [],
                }
                for r in inputs
            ]
            for call in candidates:
                predictions.append(
                    adapt_table_call(
                        indexed[call["sample_id"]],
                        call,
                        renderer=renderer.render,
                        max_new_tokens=192,
                        prompt_format="descriptive_schema",
                    )
                )
            report = evaluate_tables(
                predictions,
                inputs,
                refs,
                ["unchanged_0", *conditions],
                normalize=normalize,
                score=lambda p, r: {
                    "teds": teds.evaluate(p, r),
                    "teds_structure": structure.evaluate(p, r),
                },
            )
            save(out / "evaluation.json", report)
            (out / "predictions.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in predictions)
            )
            receipt.update(
                status="completed",
                official_source=official,
                evaluation_sha256=sha256(out / "evaluation.json"),
                predictions_sha256=sha256(out / "predictions.jsonl"),
                cohort=cohort,
                inference_run_sha256={
                    read(p / "run.json")["condition"]: sha256(p / "run.json") for p in args.runs
                },
            )
            print(json.dumps([r for r in report["summary"] if r["variant"] == "all"], indent=2))
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error)[:900])
        raise
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        save(out / "run.json", receipt)


if __name__ == "__main__":
    main()
