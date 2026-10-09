#!/usr/bin/env python3
"""Prepare anonymous, reference-free train tasks for an interactive teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.loop import Observation, digest
from ocr_edr.native_tables import load_native_table_sources
from ocr_edr.sft import sha256
from ocr_edr.table_pilot import HTMLTableRenderer, table_cell_map

SEED = "interactive-teacher-20261007"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["dataset", "native-run", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    sources = load_native_table_sources(args.dataset, role="train")
    source_map = {row["family_id"]: row for row in sources}
    run_path = args.native_run / "run.json"
    run = json.loads(run_path.read_text())
    prediction_path = args.native_run / "predictions.jsonl"
    if (
        run["status"] != "completed"
        or run["source_role"] != "train"
        or run["completed_sources"] != len(sources)
        or run["reference_access"] != "none"
        or run["admission_sha256"] != sha256(args.dataset / "admitted-supervision/admission.json")
        or run["input_sha256"] != sha256(args.dataset / "train-source-inputs.jsonl")
        or run["predictions_sha256"] != sha256(prediction_path)
    ):
        raise ValueError("Teacher native pool provenance mismatch")
    rows = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    native = {r["family_id"]: r for r in rows}
    if len(native) != len(rows) or set(native) != set(source_map):
        raise ValueError("Complete unique admitted native pool required")
    for family, record in native.items():
        if any(record[k] != v for k, v in source_map[family].items()):
            raise ValueError("Native pool source identity drift")
    meta = json.loads((args.dataset / "dataset.json").read_text())
    inputs_path = args.dataset / "train-inputs.jsonl"
    if sha256(inputs_path) != meta["file_sha256"][inputs_path.name]:
        raise ValueError("Frozen controlled-input identity mismatch")
    controlled = [json.loads(line) for line in inputs_path.read_text().splitlines()]
    families = [f"p{i:04d}" for i in range(9, 21)]
    if not set(families) <= set(source_map):
        raise ValueError("Predeclared unreviewed train cohort missing")

    def rank(label):
        return hashlib.sha256((SEED + ":" + label).encode()).hexdigest()

    native_families = set(sorted(families, key=lambda f: rank("route:" + f))[:6])
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    renderer = HTMLTableRenderer(root / "renders")
    mapping, tasks = [], []
    for family in families:
        source = source_map[family]
        if family in native_families:
            candidate = native[family]["prediction"]
            origin, original_id = "native", family + "-native"
        else:
            choices = [r for r in controlled if r["family_id"] == family]
            selected = min(choices, key=lambda r: rank("variant:" + r["sample_id"]))
            candidate = selected["prediction"]
            origin, original_id = "controlled", selected["sample_id"]
        case_id = "t_" + rank("opaque:" + family)[:12]
        work = root / "cases" / case_id
        work.mkdir(parents=True)
        shutil.copyfile(args.dataset / source["source_image"], work / "source.png")
        if sha256(work / "source.png") != source["source_sha256"]:
            raise ValueError("Copied teacher source differs")
        render_error, render_hash = None, None
        try:
            if candidate is None:
                raise ValueError("Native parser has no HTML candidate")
            rendered = renderer.render(Observation(case_id, "table", "", candidate))
            shutil.copyfile(rendered.path, work / "current.png")
            render_hash = sha256(work / "current.png")
            addresses = table_cell_map(candidate)
        except Exception as error:
            render_error = f"{type(error).__name__}: {str(error)[:250]}"
            addresses = None
        task = {
            "case_id": case_id,
            "source_image": "source.png",
            "source_sha256": source["source_sha256"],
            "current_html": candidate,
            "current_html_sha256": digest(candidate or ""),
            "current_render_sha256": render_hash,
            "current_render_error": render_error,
            "current_cell_addresses": addresses,
        }
        (work / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n")
        (work / "state.json").write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "html": candidate,
                    "html_sha256": digest(candidate or ""),
                    "edit_attempts": 0,
                    "terminal": False,
                    "steps": [],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        mapping.append(
            {
                "case_id": case_id,
                "family_id": family,
                "role": "train",
                "origin": origin,
                "original_sample_id": original_id,
                "source_sha256": source["source_sha256"],
            }
        )
        tasks.append(
            {
                "case_id": case_id,
                "task_sha256": sha256(work / "task.json"),
                "source_sha256": source["source_sha256"],
            }
        )
    (root / "private-mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    tasks.sort(key=lambda r: r["case_id"])
    receipt = {
        "status": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "teacher_model_label": "gpt-6-astra",
        "teacher_effort_label": "max",
        "teacher_identity_source": "active session turn_context observed 2026-10-07T08:51:15.935Z",
        "teacher_immutable_snapshot_available": False,
        "teacher_mode": "interactive_assistant",
        "source_documents": 12,
        "native_cases": 6,
        "controlled_cases": 6,
        "tasks": tasks,
        "selection_seed": SEED,
        "selection_uses_model_quality": False,
        "dataset_sha256": sha256(args.dataset / "dataset.json"),
        "native_run_sha256": sha256(run_path),
        "admission_sha256": run["admission_sha256"],
        "private_mapping_sha256": sha256(root / "private-mapping.json"),
        "reference_access_before_teacher_decisions": False,
        "max_edit_attempts_per_case": 3,
        "calibration_locked_access": False,
        "driver_sha256": sha256(Path(__file__)),
    }
    (root / "packet.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "prepared", "anonymous_case_ids": [r["case_id"] for r in tasks]}))


if __name__ == "__main__":
    main()
