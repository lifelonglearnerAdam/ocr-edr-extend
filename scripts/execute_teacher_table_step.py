#!/usr/bin/env python3
"""Execute a sealed teacher decision and bind the resulting HTML/render state."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.loop import digest
from ocr_edr.sft import sha256
from ocr_edr.table_pilot import HTMLTableRenderer, table_cell_map
from ocr_edr.table_sft_screen import _unique_object
from ocr_edr.teacher_tables import execute_teacher_decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    args = parser.parse_args()
    packet = json.loads((args.packet / "packet.json").read_text())
    decision = json.loads(args.decision.read_text(), object_pairs_hook=_unique_object)
    case = decision["case_id"]
    expected = next((r for r in packet["tasks"] if r["case_id"] == case), None)
    if expected is None:
        raise ValueError("Decision names an unknown teacher case")
    work = args.packet / "cases" / case
    if (
        sha256(work / "task.json") != expected["task_sha256"]
        or sha256(work / "source.png") != expected["source_sha256"]
    ):
        raise ValueError("Frozen teacher observation changed")
    state = json.loads((work / "state.json").read_text())
    if state["terminal"] or state["html_sha256"] != digest(state["html"] or ""):
        raise ValueError("Teacher episode is terminal or state checksum differs")
    stopping = decision["action"] == {"action": "stop"}
    if not stopping and state["edit_attempts"] >= packet["max_edit_attempts_per_case"]:
        raise ValueError("Teacher edit budget exhausted; retain state and stop/abstain")
    step = len(state["steps"])
    event_path = work / f"step-{step:02d}.json"
    if event_path.exists():
        raise ValueError("Existing event needs reconciliation; never silently replay it")
    renderer = HTMLTableRenderer(args.packet / "renders")
    event = execute_teacher_decision(state["html"], decision, renderer=renderer.render)
    event.update(
        recorded_at=datetime.now(timezone.utc).isoformat(),
        teacher_decision_sha256=sha256(args.decision),
        teacher_model_label=packet["teacher_model_label"],
        teacher_effort_label=packet["teacher_effort_label"],
        source_sha256=expected["source_sha256"],
        executor_sha256=sha256(Path(__file__)),
        helper_sha256=sha256(Path(__file__).resolve().parents[1] / "src/ocr_edr/teacher_tables.py"),
    )
    event_path.write_text(json.dumps(event, ensure_ascii=False, indent=2) + "\n")
    state.update(
        html=event["final_html"],
        html_sha256=digest(event["final_html"] or ""),
        edit_attempts=state["edit_attempts"] + (not stopping),
        terminal=stopping and event["applied"],
    )
    state["steps"].append({"file": event_path.name, "sha256": sha256(event_path)})
    if state["terminal"]:
        state["terminal_assessment"] = (
            "teacher_self_check" if decision["uncertainty"] == "low" else "teacher_abstention"
        )
    (work / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    try:
        addresses = table_cell_map(state["html"]) if state["html"] is not None else None
    except ValueError:
        # An explicit abstention on a malformed native candidate is still a
        # completed teacher event; the raw HTML and error history remain intact.
        addresses = None
    view = {
        "case_id": case,
        "current_html_sha256": state["html_sha256"],
        "current_html": state["html"],
        "terminal": state["terminal"],
        "edit_attempts": state["edit_attempts"],
        "action_error": event["action_error"],
        "render_error": event["render_error"],
        "render_path": event.get("render_path"),
        "current_cell_addresses": addresses,
    }
    (work / "latest-view.json").write_text(json.dumps(view, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: view[k]
                for k in [
                    "case_id",
                    "terminal",
                    "edit_attempts",
                    "action_error",
                    "render_error",
                    "render_path",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
