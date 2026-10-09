#!/usr/bin/env python3
"""Freeze every terminal teacher episode before opening any reference labels."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.loop import digest
from ocr_edr.sft import sha256
from ocr_edr.table_sft_screen import _unique_object


def read(path):
    return json.loads(path.read_text(), object_pairs_hook=_unique_object)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.packet.resolve(), args.output.resolve()
    if output.exists() or output == root or root in output.parents:
        raise ValueError("Seal must be a new file outside the packet")
    packet = read(root / "packet.json")
    if sha256(root / "teacher-instructions.txt") != packet["teacher_prompt_sha256"]:
        raise ValueError("Frozen teacher prompt changed")
    project = Path(__file__).resolve().parents[1]
    executor_hash = sha256(project / "scripts/execute_teacher_table_step.py")
    helper_hash = sha256(project / "src/ocr_edr/teacher_tables.py")
    states = []
    for task in packet["tasks"]:
        case = task["case_id"]
        work = root / "cases" / case
        work.resolve().relative_to(root)
        state = read(work / "state.json")
        if (
            state.get("case_id") != case
            or not state["terminal"]
            or not state["steps"]
            or state["html_sha256"] != digest(state["html"] or "")
            or state["edit_attempts"] > packet["max_edit_attempts_per_case"]
            or sha256(work / "task.json") != task["task_sha256"]
            or sha256(work / "source.png") != task["source_sha256"]
        ):
            raise ValueError("Teacher episode is not terminal or its evidence changed")
        for index, saved in enumerate(state["steps"]):
            if (
                saved["file"] != f"step-{index:02d}.json"
                or sha256(work / saved["file"]) != saved["sha256"]
            ):
                raise ValueError("Teacher event hash/order changed before sealing")
            event = read(work / saved["file"])
            if event["executor_sha256"] != executor_hash or event["helper_sha256"] != helper_hash:
                raise ValueError(
                    "Execution code changed; recover its exact snapshot before sealing"
                )
        last = read(work / state["steps"][-1]["file"])
        if last["action"] != {"action": "stop"} or not last["applied"]:
            raise ValueError("Teacher terminal state needs an executed stop")
        states.append(state)
    if len(states) != 12 or len({s["case_id"] for s in states}) != 12:
        raise ValueError("All twelve unique teacher episodes are required")
    paths = [
        "scripts/prepare_teacher_table_pilot.py",
        "scripts/execute_teacher_table_step.py",
        "src/ocr_edr/teacher_tables.py",
        "src/ocr_edr/table_pilot.py",
        "src/ocr_edr/loop.py",
        "src/ocr_edr/table_sft_screen.py",
        "docs/research/INTERACTIVE_TEACHER_PILOT_20261007.md",
    ]
    if sha256(project / paths[0]) != packet["driver_sha256"]:
        raise ValueError("Preparation code changed; recover its exact snapshot before sealing")
    output.parent.mkdir(parents=True, exist_ok=True)
    code = output.parent / "sealed-code"
    code.mkdir(exist_ok=False)
    for relative in paths:
        target = code / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / relative, target)
    receipt = {
        "schema_version": 1,
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "status": "all_episodes_terminal_before_offline_reference_check",
        "teacher_reference_access_before_seal": "reported_none_for_this_cohort; not a model_pretraining_or_full_session_blinding_claim",
        "source_documents": len(states),
        "terminal_episodes": len(states),
        "edited_episodes": sum(s["edit_attempts"] > 0 for s in states),
        "edit_attempts": sum(s["edit_attempts"] for s in states),
        "teacher_self_check_episodes": sum(
            s["terminal_assessment"] == "teacher_self_check" for s in states
        ),
        "teacher_abstention_episodes": sum(
            s["terminal_assessment"] == "teacher_abstention" for s in states
        ),
        "packet_files_sha256": {
            p.relative_to(root).as_posix(): sha256(p)
            for p in sorted(root.rglob("*"))
            if p.is_file()
        },
        "sealed_code_sha256": {
            p.relative_to(code).as_posix(): sha256(p)
            for p in sorted(code.rglob("*"))
            if p.is_file()
        },
        "freeze_scope": "decisions, events, tasks, sources, renders, final states and executor snapshot; no independent judging claim",
    }
    with output.open("x") as handle:
        handle.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"terminal_episodes": len(states), "seal_sha256": sha256(output)}))


if __name__ == "__main__":
    main()
