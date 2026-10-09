#!/usr/bin/env python3
"""Fixed32-source white-image intervention; no new training or reference access."""

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml
from PIL import Image

from ocr_edr.native_table_repair import native_repair_inputs, verify_native_calls
from ocr_edr.native_tables import load_native_table_sources
from ocr_edr.official_tables import load_official_table_normalizer, verify_official_table_sources
from ocr_edr.qwen import QwenFormulaProposer
from ocr_edr.sft import sha256, verify_model_files
from ocr_edr.table_pilot import HTMLTableRenderer
from ocr_edr.table_sft_screen import adapt_table_call, table_messages, validate_table_adapter


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "native-run",
        "dataset",
        "training",
        "config",
        "model-path",
        "model-receipt",
        "output",
        "official-root",
    ]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    project = Path(__file__).resolve().parents[1]
    protocol = project / "docs/research/TABLE_SOURCE_EVIDENCE_PROTOCOL_20261009.md"
    native = args.native_run
    audit = read(native / "audit.json")
    if audit["status"] != "audit_completed" or not audit["full_metric_recomputation_exact"]:
        raise ValueError("Probe requires fully audited original native outputs")
    if sha256(native / "evaluation/evaluation.json") != audit["evaluation_sha256"]:
        raise ValueError("Original native evaluation changed")
    inputs = native_repair_inputs(
        rows(native / "frozen-native-inputs/predictions.jsonl"),
        load_native_table_sources(args.dataset, role="model_dev"),
        role="model_dev",
    )
    frozen_hash = "6bfe68cd225e3cac1f645849174209725936367b31fe8a91bf7a248c93ccd2f3"
    if sha256(native / "frozen-native-inputs/predictions.jsonl") != frozen_hash:
        raise ValueError("Unexpected native cohort")
    model_receipt = read(args.model_receipt)
    if model_receipt["revision"] != cfg["model_revision"]:
        raise ValueError("Fixed base revision differs")
    verify_model_files(args.model_path, model_receipt)
    versions = {name: importlib.metadata.version(name) for name in cfg["versions"]}
    if versions != cfg["versions"]:
        raise ValueError("Fixed software runtime differs")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    blank = out / "blank"
    blank.mkdir()
    for item in inputs:
        with Image.open(args.dataset / item["source_image"]) as source:
            size = source.size
        Image.new("RGB", size, (255, 255, 255)).save(blank / (item["family_id"] + ".png"))
        with Image.open(blank / (item["family_id"] + ".png")) as check:
            if check.size != size or check.getextrema() != ((255, 255),) * 3:
                raise ValueError("Intervention must preserve dimensions and be uniformly white")
    receipt = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cases": 32,
        "conditions": ["all", "no_explicit_preservation"],
        "completed_calls": {},
        "reference_access": "none",
        "calibration_locked_access": False,
        "model_receipt_sha256": sha256(args.model_receipt),
        "config_sha256": sha256(args.config),
        "protocol_sha256": sha256(protocol),
        "script_sha256": sha256(Path(__file__)),
        "native_prediction_sha256": frozen_hash,
        "native_audit_sha256": sha256(native / "audit.json"),
        "intervention": "all-white RGB PNG, same source width and height",
        "scope": "post-hoc source sensitivity; no independent quality or untouched-source claim",
        "versions": versions,
        "original_call_sha256": {},
        "adapter_sha256": {},
    }
    save(out / "run.json", receipt)
    shutil.copyfile(Path(__file__), out / "execution.py")
    receipt["official_source"] = verify_official_table_sources(
        args.official_root, "f133a71e9e91c3621c7ce8994200a7b394a06eb3"
    )
    normalize = load_official_table_normalizer(args.official_root)
    all_cases, summary = [], []
    try:
        import torch

        for arm in receipt["conditions"]:
            original_folder = native / "inference" / arm
            if sha256(original_folder / "run.json") != audit["evidence_sha256"][arm + "_receipt"]:
                raise ValueError("Original model receipt changed")
            original_receipt = read(original_folder / "run.json")
            if sha256(original_folder / "calls.jsonl") != audit["evidence_sha256"][
                arm + "_calls"
            ] or original_receipt["config_sha256"] != sha256(args.config):
                raise ValueError("Original model calls/config changed")
            original_calls = rows(original_folder / "calls.jsonl")
            verify_native_calls(original_calls, inputs, arm)
            receipt["original_call_sha256"][arm] = sha256(original_folder / "calls.jsonl")
            trained = args.training / arm
            adapter = validate_table_adapter(
                trained,
                condition=arm,
                config=cfg,
                config_sha256=sha256(args.config),
                admission_sha256=sha256(args.dataset / "admitted-supervision/admission.json"),
            )
            if adapter != original_receipt["adapter_sha256"]:
                raise ValueError("Intervention checkpoint differs")
            receipt["adapter_sha256"][arm] = adapter
            pids = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
            ).splitlines()
            if {int(pid.strip()) for pid in pids if pid.strip()} - {os.getpid()}:
                raise RuntimeError("GPU occupied before source evidence probe")
            torch.manual_seed(cfg["seed"])
            torch.cuda.manual_seed_all(cfg["seed"])
            torch.set_num_threads(4)
            torch.use_deterministic_algorithms(True, warn_only=True)
            model = QwenFormulaProposer(
                args.model_path,
                device="cuda:0",
                precision_profile="nf4_lora_8gb",
                adapter_path=trained / "checkpoint",
                min_pixels=cfg["min_pixels"],
                max_pixels=cfg["max_pixels"],
                max_new_tokens=192,
            )
            if model.precision_metadata["quantization"] != original_receipt["quantization"]:
                raise ValueError("Intervention precision differs")
            receipt["completed_calls"][arm] = 0
            folder = out / arm
            folder.mkdir()
            renderer = HTMLTableRenderer(folder / "renders")
            generated = []
            with (folder / "calls.jsonl").open("w") as stream:
                for item, original in zip(inputs, original_calls):
                    path = blank / (item["family_id"] + ".png")
                    messages, prompt = table_messages(
                        item["prediction"], prompt_format="descriptive_schema"
                    )
                    call = model.generate([path], messages, prompt)
                    if (
                        call["ordered_image_sha256"] != [sha256(path)]
                        or call["prompt"] != original["prompt"]
                        or call["messages"] != original["messages"]
                        or call["image_grid_thw"] != original["image_grid_thw"]
                        or call["input_tokens"] != original["input_tokens"]
                    ):
                        raise ValueError(
                            "Pair differs beyond the intended image content intervention"
                        )
                    row = {
                        **item,
                        **call,
                        "condition": arm,
                        "original_source_sha256": item["source_sha256"],
                        "intervention_image_sha256": sha256(path),
                    }
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    generated.append(row)
                    receipt["completed_calls"][arm] += 1
                    if receipt["completed_calls"][arm] % 8 == 0:
                        save(out / "run.json", receipt)
                        print(arm, receipt["completed_calls"][arm], "/32", flush=True)
            # Compare only after this arm's complete output log is closed/frozen.
            original_predictions = {
                r["sample_id"]: r
                for r in rows(native / "evaluation/predictions.jsonl")
                if r["arm"] == arm
            }
            paired = []
            for item, call, original_call in zip(inputs, generated, original_calls):
                final = adapt_table_call(
                    item,
                    call,
                    renderer=renderer.render,
                    max_new_tokens=192,
                    prompt_format="descriptive_schema",
                )
                original = original_predictions[item["sample_id"]]
                trace = final["trace"][0]
                paired.append(
                    {
                        "sample_id": item["sample_id"],
                        "arm": arm,
                        "raw_output_equal": call["raw_output"] == original_call["raw_output"],
                        "action_equal": trace.get("action") == original["trace"][0].get("action"),
                        "final_html_equal": final["final_prediction"]
                        == original["final_prediction"],
                        "normalized_html_equal": normalize(final["final_prediction"])
                        == normalize(original["final_prediction"]),
                        "blank_action": {
                            k: v for k, v in (trace.get("action") or {}).items() if k != "text"
                        },
                        "real_action": {
                            k: v
                            for k, v in (original["trace"][0].get("action") or {}).items()
                            if k != "text"
                        },
                        "blank_contract": trace["contract"],
                        "hit_length_cap": trace["hit_length_cap"],
                        "input_tokens": call["input_tokens"],
                        "output_tokens": call["output_tokens"],
                        "generation_seconds": call["generation_seconds"],
                        "original_image_sha256": item["source_sha256"],
                        "blank_image_sha256": call["intervention_image_sha256"],
                    }
                )
            summary.append(
                {
                    "arm": arm,
                    "cases": 32,
                    **{
                        k: sum(r[k] for r in paired)
                        for k in [
                            "raw_output_equal",
                            "action_equal",
                            "final_html_equal",
                            "normalized_html_equal",
                            "hit_length_cap",
                            "input_tokens",
                            "output_tokens",
                            "generation_seconds",
                        ]
                    },
                    "blank_actions": dict(
                        Counter(r["blank_action"].get("action", "rejected") for r in paired)
                    ),
                    "contracts": dict(Counter(r["blank_contract"] for r in paired)),
                    "calls_sha256": sha256(folder / "calls.jsonl"),
                }
            )
            all_cases.extend(paired)
            save(out / "comparison.json", {"summary": summary, "cases": all_cases})
            del model
            import gc

            gc.collect()
            torch.cuda.empty_cache()
        receipt.update(
            status="completed",
            comparison_sha256=sha256(out / "comparison.json"),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as error:
        receipt.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error)[:600],
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        raise
    finally:
        save(out / "run.json", receipt)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
