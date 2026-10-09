#!/usr/bin/env python3
"""Continue the owned training service through reload,135diagnoses and341refinements."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.sft import sha256


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", type=Path, default=project / "experiments/runs/table-diagnosis-20261009"
    )
    parser.add_argument(
        "--training-unit", default="ocr-edr-table-diagnosis-training-20261009.service"
    )
    args = parser.parse_args()
    root = args.study.resolve()
    root.relative_to(project / "experiments/runs")
    receipt_path = root / "pipeline-status.json"
    if receipt_path.exists():
        raise ValueError(
            "A pipeline receipt already exists; inspect/recover it instead of rerunning"
        )
    python = sys.executable
    dataset = project / "data/processed/pubtabnet-four-roles-20261007"
    model = (
        project
        / "data/raw/model-cache/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c"
    )
    model_receipt = project / "experiments/runs/server-selection-20261006/model-receipt.json"
    native = project / "experiments/runs/native-table-sft-20261009"
    state = {
        "status": "waiting_training",
        "stage": "waiting_owned_training",
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_stages": [],
        "driver_sha256": sha256(Path(__file__)),
        "owned_training_unit": args.training_unit,
        "calibration_locked_access": False,
    }

    def save():
        receipt_path.write_text(json.dumps(state, indent=2) + "\n")

    save()

    def stage(name, script, arguments, output):
        state.update(status="running", stage=name, child_pid=None)
        save()
        command = [
            python,
            str(project / "scripts" / script),
            *map(str, arguments),
            "--output",
            str(output),
        ]
        log = root / (name + ".log")
        with log.open("w") as sink:
            child = subprocess.Popen(command, cwd=project, stdout=sink, stderr=subprocess.STDOUT)
            state.update(
                child_pid=child.pid, child_script_sha256=sha256(project / "scripts" / script)
            )
            save()
            if child.wait() != 0:
                raise RuntimeError("Stage failed: " + name + "; preserve output/log and inspect it")
        result = json.loads((output / "run.json").read_text())
        if result["status"] != "completed":
            raise ValueError("Nonterminal child receipt: " + name)
        state["completed_stages"].append(
            {"stage": name, "receipt_sha256": sha256(output / "run.json")}
        )
        state["child_pid"] = None
        save()
        print("Completed", name, flush=True)

    try:
        while True:
            activity = subprocess.run(
                ["systemctl", "--user", "is-active", state["owned_training_unit"]],
                capture_output=True,
                text=True,
            ).stdout.strip()
            state.update(
                training_service_activity=activity,
                checked_at=datetime.now(timezone.utc).isoformat(),
            )
            save()
            if activity != "active":
                break
            time.sleep(20)
        training = json.loads((root / "training/run.json").read_text())
        if training["status"] != "completed" or training["completed_steps"] != 381:
            raise ValueError("Owned training stopped without complete381-step checkpoint")
        state["completed_stages"].append(
            {"stage": "training", "receipt_sha256": sha256(root / "training/run.json")}
        )
        save()
        base_args = ["--model-path", model, "--model-receipt", model_receipt]
        stage(
            "reload",
            "verify_table_nf4_checkpoint.py",
            ["--training-run", root / "training", *base_args],
            root / "reload",
        )
        diagnosis_args = [
            "--dataset",
            dataset,
            "--admission",
            dataset / "admitted-supervision",
            "--config",
            project / "configs/train/table_diagnosis_nf4_20261009.yaml",
            "--training",
            root / "training",
            "--native-run",
            native,
            *base_args,
        ]
        for cohort in ["controlled", "native"]:
            stage(
                "diagnosis_" + cohort,
                "run_table_diagnosis.py",
                [*diagnosis_args, "--cohort", cohort],
                root / "diagnoses" / cohort,
            )
        stage(
            "packets",
            "prepare_table_diagnosis_packets.py",
            [
                "--dataset",
                dataset,
                "--admission",
                dataset / "admitted-supervision",
                "--diagnoses",
                root / "diagnoses",
                "--native-run",
                native,
            ],
            root / "packets",
        )
        refinement_args = [
            "--dataset",
            dataset,
            "--admission",
            dataset / "admitted-supervision",
            "--config",
            project / "configs/train/sft_table_nf4_screen_20261007.yaml",
            "--training",
            project / "experiments/runs/table-nf4-screen-recovery-20261009/training/all",
            "--native-run",
            native,
            *base_args,
        ]
        for key in [
            "controlled-learned",
            "controlled-displaced_region",
            "controlled-oracle_controlled",
            "native-learned",
        ]:
            stage(
                "refinement_" + key,
                "run_diagnosis_guided_refiner.py",
                [*refinement_args, "--packet", root / "packets" / (key + ".jsonl")],
                root / "refinement" / key,
            )
        stage(
            "evaluation",
            "evaluate_table_diagnosis.py",
            [
                "--dataset",
                dataset,
                "--study",
                root,
                "--native-run",
                native,
                "--prompt-run",
                project / "experiments/runs/table-prompt-ablation-20261009",
                "--official-root",
                project / "data/raw/omnidocbench-source",
            ],
            root / "evaluation",
        )
        state.update(
            status="completed",
            stage="completed",
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        save()
    except Exception as error:
        state.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error)[:800],
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        save()
        raise


if __name__ == "__main__":
    main()
