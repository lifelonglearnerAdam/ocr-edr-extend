#!/usr/bin/env python3
"""Freeze class/location diagnoses for complete controlled or native model-dev inputs."""

import argparse
import importlib.metadata
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from ocr_edr.native_table_repair import native_repair_inputs
from ocr_edr.native_tables import load_native_table_sources
from ocr_edr.qwen import QwenFormulaProposer
from ocr_edr.sft import sha256, verify_model_files
from ocr_edr.table_diagnosis import diagnosis_prompt
from ocr_edr.table_sft_screen import load_table_screen_inputs, validate_table_adapter


def read(path):
    return json.loads(path.read_text())


def load_cohort(dataset, cohort, native_run):
    if cohort == "controlled":
        return load_table_screen_inputs(dataset)
    predictions = native_run / "frozen-native-inputs/predictions.jsonl"
    if sha256(predictions) != "6bfe68cd225e3cac1f645849174209725936367b31fe8a91bf7a248c93ccd2f3":
        raise ValueError("Frozen native source cohort differs")
    return native_repair_inputs(
        [json.loads(line) for line in predictions.read_text().splitlines()],
        load_native_table_sources(dataset, role="model_dev"),
        role="model_dev",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "dataset",
        "config",
        "model-path",
        "model-receipt",
        "training",
        "admission",
        "output",
    ]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cohort", choices=["controlled", "native"], required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    cases = load_cohort(args.dataset, args.cohort, args.native_run)
    count = 103 if args.cohort == "controlled" else 32
    if len(cases) != count or len({r["family_id"] for r in cases}) != 32:
        raise ValueError("Complete frozen diagnosis cohort required")
    model_receipt = read(args.model_receipt)
    if model_receipt["revision"] != cfg["model_revision"]:
        raise ValueError("Diagnostic base revision mismatch")
    verify_model_files(args.model_path, model_receipt)
    adapter = validate_table_adapter(
        args.training,
        condition="all",
        config=cfg,
        config_sha256=sha256(args.config),
        admission_sha256=sha256(args.admission / "admission.json"),
    )
    if read(args.training / "run.json")["model_receipt_sha256"] != sha256(args.model_receipt):
        raise ValueError("Diagnostic base differs from training")
    versions = {n: importlib.metadata.version(n) for n in cfg["versions"]}
    if versions != cfg["versions"]:
        raise ValueError("Diagnostic inference dependency versions changed")
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        raise RuntimeError("Diagnostic inference requires an unoccupied GPU")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__),
        project / "src/ocr_edr/table_diagnosis.py",
        project / "src/ocr_edr/qwen.py",
        project / "src/ocr_edr/table_sft_screen.py",
        project / "src/ocr_edr/native_table_repair.py",
    ]
    receipt = {
        "status": "initializing",
        "cohort": args.cohort,
        "cases": count,
        "completed_calls": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "reference_access": "none",
        "target_access": "none",
        "calibration_locked_access": False,
        "model_receipt_sha256": sha256(args.model_receipt),
        "config_sha256": sha256(args.config),
        "training_run_sha256": sha256(args.training / "run.json"),
        "adapter_sha256": adapter,
        "source_sha256": {str(p.relative_to(project)): sha256(p) for p in paths},
        "versions": versions,
        "input_record_digest": __import__("hashlib")
        .sha256(json.dumps(cases, sort_keys=True).encode())
        .hexdigest(),
    }

    def save():
        (out / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    try:
        for p in paths:
            dest = out / "source" / p.relative_to(project)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
        import torch

        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True, warn_only=True)
        model = QwenFormulaProposer(
            args.model_path,
            device="cuda:0",
            adapter_path=args.training / "checkpoint",
            precision_profile=cfg["precision_profile"],
            min_pixels=cfg["min_pixels"],
            max_pixels=cfg["max_pixels"],
            max_new_tokens=192,
        )
        receipt.update(
            status="running",
            gpu=torch.cuda.get_device_name(0),
            cuda_runtime=torch.version.cuda,
            **model.precision_metadata,
        )
        save()
        with (out / "calls.jsonl").open("w") as sink:
            for case in cases:
                prompt = diagnosis_prompt(case["prediction"])
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                    }
                ]
                call = model.generate([args.dataset / case["source_image"]], messages, prompt)
                if call["ordered_image_sha256"] != [case["source_sha256"]]:
                    raise ValueError("Diagnosis source changed during generation")
                sink.write(
                    json.dumps({**case, **call, "condition": "all"}, ensure_ascii=False) + "\n"
                )
                sink.flush()
                receipt["completed_calls"] += 1
                if receipt["completed_calls"] % 8 == 0:
                    save()
                    print(args.cohort, receipt["completed_calls"], "/", count, flush=True)
        receipt.update(status="completed", calls_sha256=sha256(out / "calls.jsonl"))
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error)[:800])
        raise
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
