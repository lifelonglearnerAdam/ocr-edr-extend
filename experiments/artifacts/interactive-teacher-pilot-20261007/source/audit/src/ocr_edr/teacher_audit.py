"""Post-seal teacher checks; mechanical validity never implies visual correctness."""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path

from .loop import digest
from .sft import sha256
from .table_pilot import parse_table
from .teacher_tables import execute_teacher_decision


def verify_frozen_files(root: Path, manifest: dict[str, str]) -> None:
    root = root.resolve()
    if not manifest:
        raise ValueError("Empty frozen manifest")
    for relative, expected in manifest.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not re.fullmatch("[0-9a-f]{64}", expected):
            raise ValueError("Invalid frozen manifest path/hash")
        source = root / path
        source.resolve().relative_to(root)
        if source.is_symlink() or not source.is_file() or sha256(source) != expected:
            raise ValueError(f"Frozen file missing, linked or modified: {relative}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(manifest):
        raise ValueError("Frozen manifest file coverage differs")


def replay_teacher_episode(task, decisions, events, state, *, max_edit_attempts, renderer):
    """Replay edits against the initial state while preserving failed original calls.

    Successful frozen renders must match freshly generated pixels. A recorded render
    failure remains a rollback even if the independent diagnostic replay now succeeds.
    Nothing in this check establishes whether the teacher actually saw a given image.
    """
    case, current = task["case_id"], task["current_html"]
    if (
        not decisions
        or len(decisions) != len(events)
        or task["current_html_sha256"] != digest(current or "")
        or type(max_edit_attempts) is not int
        or max_edit_attempts < 0
    ):
        raise ValueError("Incomplete teacher episode or invalid initial identity/budget")
    counts = dict.fromkeys(
        [
            "edit_attempts",
            "applied_edits",
            "local_edits",
            "global_edits",
            "action_failures",
            "render_failures",
            "fresh_render_matches",
        ],
        0,
    )
    stopped = False
    for decision, event in zip(decisions, events):
        if stopped or decision["case_id"] != case or event["case_id"] != case:
            raise ValueError("Teacher case differs or contains events after terminal stop")
        if decision["observed_html_sha256"] != digest(current or ""):
            raise ValueError("Stale teacher decision in replay")
        stop = decision["action"] == {"action": "stop"}
        counts["edit_attempts"] += not stop
        if counts["edit_attempts"] > max_edit_attempts:
            raise ValueError("Teacher edit budget exceeded")
        replay = execute_teacher_decision(current, decision, renderer=renderer)
        for key in [
            "input_html_sha256",
            "action",
            "source_evidence",
            "uncertainty",
            "teacher_selfcheck_only",
            "candidate_html",
            "action_error",
            "edit_scope",
            "render_calls",
        ]:
            if event.get(key) != replay.get(key):
                raise ValueError(f"Frozen teacher event differs from replay: {key}")
        elapsed = event["render_seconds"]
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid teacher rendering cost")
        if event["render_error"]:
            if (
                not isinstance(event["render_error"], str)
                or event["action_error"]
                or event["render_calls"] != 1
                or event["applied"] is not False
                or event["changed"] is not False
                or event["final_html"] != current
                or any(k in event for k in ["render_image_sha256", "render_prediction_sha256"])
            ):
                raise ValueError(
                    "Failed render must retain the original state without success evidence"
                )
            counts["render_failures"] += 1
        else:
            for key in ["final_html", "applied", "changed", "render_error"]:
                if event[key] != replay[key]:
                    raise ValueError(f"Frozen teacher outcome differs from replay: {key}")
            if event["render_calls"]:
                for key in ["render_image_sha256", "render_prediction_sha256", "render_backend"]:
                    if event.get(key) != replay.get(key):
                        raise ValueError(f"Fresh teacher render differs: {key}")
                counts["fresh_render_matches"] += 1
        if event["action_error"]:
            counts["action_failures"] += 1
        if event["applied"] and not stop:
            counts["applied_edits"] += 1
            counts[event["edit_scope"] + "_edits"] += 1
        current = event["final_html"]
        stopped = stop and event["applied"]
    assessment = (
        "teacher_self_check" if decisions[-1]["uncertainty"] == "low" else "teacher_abstention"
    )
    expected = {
        "case_id": case,
        "html": current,
        "html_sha256": digest(current or ""),
        "edit_attempts": counts["edit_attempts"],
        "terminal": True,
        "terminal_assessment": assessment,
    }
    if not stopped or any(state.get(k) != v for k, v in expected.items()):
        raise ValueError("Frozen terminal state differs from the complete action chain")
    valid_final = False
    try:
        if current is not None:
            parse_table(current)
            valid_final = True
    except ValueError:
        pass  # A sealed abstention may intentionally retain malformed parser output.
    return {
        "case_id": case,
        "mechanically_valid": True,
        "events": len(events),
        **counts,
        "changed_from_initial": current != task["current_html"],
        "final_html_sha256": digest(current or ""),
        "final_html_valid": valid_final,
        "terminal_assessment": assessment,
        "independent_visual_judgment": False,
        "human_or_teacher_image_view_attestation": "not mechanically provable",
    }


def validate_teacher_mapping(mapping, sources, excluded_families):
    inventory = {r["family_id"]: r for r in sources}
    if not mapping or len(inventory) != len(sources):
        raise ValueError("Unique nonempty teacher mapping/source inventory required")
    heldout_docs = {r["document_id"] for r in sources if r["role"] != "train"}
    heldout_images = {r["source_sha256"] for r in sources if r["role"] != "train"}
    seen = {k: set() for k in ["case", "family", "document", "image"]}
    for row in mapping:
        source = inventory.get(row["family_id"])
        if (
            source is None
            or row["role"] != "train"
            or source["role"] != "train"
            or row["family_id"] in excluded_families
            or source["source_sha256"] != row["source_sha256"]
            or source["document_id"] in heldout_docs
            or source["source_sha256"] in heldout_images
        ):
            raise ValueError("Teacher source is excluded, held-out, overlapping or has drifted")
        for key, value in {
            "case": row["case_id"],
            "family": row["family_id"],
            "document": source["document_id"],
            "image": row["source_sha256"],
        }.items():
            if value in seen[key]:
                raise ValueError(f"Duplicate teacher {key}")
            seen[key].add(value)


def paired_teacher_views(episodes, audits):
    by_case = {r["case_id"]: r for r in audits}
    if (
        len(by_case) != len(audits)
        or len({r["case_id"] for r in episodes}) != len(episodes)
        or set(by_case) != {r["case_id"] for r in episodes}
    ):
        raise ValueError("Teacher views require complete unique audit coverage")
    final, trajectory = [], []
    for episode in episodes:
        if by_case[episode["case_id"]]["provisionally_admitted"] is not True:
            continue
        common = {
            "case_id": episode["case_id"],
            "role": "train",
            "input": {k: episode[k] for k in ["source_image", "source_sha256", "initial_html"]},
            "target_html": episode["final_html"],
            "target_html_sha256": digest(episode["final_html"]),
            "admission": "provisional_weak_reference_filter; not_independent_visual_gold",
        }
        final.append(copy.deepcopy(common))
        trajectory.append({**copy.deepcopy(common), "steps": copy.deepcopy(episode["steps"])})
    return final, trajectory
