#!/usr/bin/env python3
"""Fixed refiner on a complete hash-bound advice packet, without reference reads."""

import argparse
import importlib.metadata
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml
from run_table_diagnosis import load_cohort

from ocr_edr.native_table_repair import verify_native_calls
from ocr_edr.qwen import QwenFormulaProposer
from ocr_edr.sft import sha256, verify_model_files
from ocr_edr.table_diagnosis import verify_diagnosis_binding
from ocr_edr.table_sft_screen import table_messages, validate_table_adapter


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "packet",
        "dataset",
        "native-run",
        "config",
        "training",
        "admission",
        "model-path",
        "model-receipt",
        "output",
    ]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    packet_receipt = read(args.packet.parent / "run.json")
    definition = packet_receipt["packets"][args.packet.name]
    if packet_receipt["status"] != "completed" or sha256(args.packet) != definition["sha256"]:
        raise ValueError("Advisory packet is not complete and sealed")
    for cohort, pins in packet_receipt["diagnosis_evidence_sha256"].items():
        folder = args.packet.parent.parent / "diagnoses" / cohort
        if (
            sha256(folder / "run.json") != pins["receipt"]
            or sha256(folder / "calls.jsonl") != pins["calls"]
        ):
            raise ValueError("Generating diagnosis calls changed")
    cases = load_cohort(args.dataset, definition["cohort"], args.native_run)
    packets = [json.loads(line) for line in args.packet.read_text().splitlines()]
    if len(packets) != len(cases) or len(cases) != definition["cases"]:
        raise ValueError("Complete advice source coverage required")
    for case, row in zip(cases, packets):
        if any(row.get(k) != v for k, v in case.items()):
            raise ValueError("Advice case identity differs")
        verify_diagnosis_binding(case, row["diagnosis"])
    cfg = yaml.safe_load(args.config.read_text())
    model_receipt = read(args.model_receipt)
    if model_receipt["revision"] != cfg["model_revision"]:
        raise ValueError("Fixed refiner base differs")
    verify_model_files(args.model_path, model_receipt)
    adapter = validate_table_adapter(
        args.training,
        condition="all",
        config=cfg,
        config_sha256=sha256(args.config),
        admission_sha256=sha256(args.admission / "admission.json"),
    )
    if read(args.training / "run.json")["model_receipt_sha256"] != sha256(args.model_receipt):
        raise ValueError("Refiner model receipt differs from original training")
    versions = {n: importlib.metadata.version(n) for n in cfg["versions"]}
    if versions != cfg["versions"]:
        raise ValueError("Fixed refiner runtime differs")
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        raise RuntimeError("Guided refinement needs an unoccupied GPU")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__),
        project / "src/ocr_edr/table_diagnosis.py",
        project / "src/ocr_edr/table_sft_screen.py",
        project / "src/ocr_edr/qwen.py",
    ]
    receipt = {
        "status": "initializing",
        "cohort": definition["cohort"],
        "hint_condition": definition["condition"],
        "cases": len(cases),
        "completed_calls": 0,
        "reference_access": "none",
        "oracle_location_assistance": definition["condition"] == "oracle_controlled",
        "target_access": (
            "projected_controlled_class_location_only"
            if definition["condition"] == "oracle_controlled"
            else "none"
        ),
        "calibration_locked_access": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "packet_sha256": sha256(args.packet),
        "packet_receipt_sha256": sha256(args.packet.parent / "run.json"),
        "config_sha256": sha256(args.config),
        "model_receipt_sha256": sha256(args.model_receipt),
        "training_run_sha256": sha256(args.training / "run.json"),
        "adapter_sha256": adapter,
        "versions": versions,
        "source_sha256": {str(p.relative_to(project)): sha256(p) for p in paths},
    }

    def save():
        (out / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    try:
        for p in paths:
            dst = out / "source" / p.relative_to(project)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
        import torch

        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True, warn_only=True)
        model = QwenFormulaProposer(
            args.model_path,
            device="cuda:0",
            precision_profile=cfg["precision_profile"],
            adapter_path=args.training / "checkpoint",
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
            for case, row in zip(cases, packets):
                messages, prompt = table_messages(
                    case["prediction"], diagnosis=row["diagnosis"], case=case
                )
                call = model.generate([args.dataset / case["source_image"]], messages, prompt)
                if call["ordered_image_sha256"] != [case["source_sha256"]]:
                    raise ValueError("Guided model source changed")
                sink.write(
                    json.dumps(
                        {**case, **call, "diagnosis": row["diagnosis"], "condition": "all"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                sink.flush()
                receipt["completed_calls"] += 1
                if receipt["completed_calls"] % 8 == 0:
                    save()
                    print(
                        definition["cohort"],
                        definition["condition"],
                        receipt["completed_calls"],
                        "/",
                        len(cases),
                        flush=True,
                    )
        calls = [json.loads(line) for line in (out / "calls.jsonl").read_text().splitlines()]
        verify_native_calls(calls, cases, "all")
        receipt.update(status="completed", calls_sha256=sha256(out / "calls.jsonl"))
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error)[:800])
        raise
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
