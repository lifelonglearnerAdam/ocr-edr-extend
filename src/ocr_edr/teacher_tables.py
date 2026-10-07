"""Replayable teacher actions with state/render binding, without correctness labels."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .loop import Observation, digest
from .sft import sha256
from .table_pilot import apply_table_action, parse_table


def execute_teacher_decision(initial: str | None, decision: dict, *, renderer) -> dict:
    fields = {"case_id", "observed_html_sha256", "source_evidence", "uncertainty", "action"}
    if (
        set(decision) != fields
        or not isinstance(decision["case_id"], str)
        or not decision["case_id"]
    ):
        raise ValueError("Unexpected teacher decision schema")
    if (
        not isinstance(decision["source_evidence"], str)
        or not decision["source_evidence"].strip()
        or len(decision["source_evidence"]) > 2000
        or decision["uncertainty"] not in {"low", "ambiguous", "unreadable"}
        or not isinstance(decision["action"], dict)
    ):
        raise ValueError("Teacher evidence/uncertainty/action is invalid")
    observed = digest(initial or "")
    if decision["observed_html_sha256"] != observed:
        raise ValueError("Teacher observation is stale or uses a different HTML state")
    action = decision["action"]
    result = {
        "case_id": decision["case_id"],
        "input_html_sha256": observed,
        "action": action,
        "source_evidence": decision["source_evidence"],
        "uncertainty": decision["uncertainty"],
        "teacher_selfcheck_only": True,
        "candidate_html": None,
        "final_html": initial,
        "applied": False,
        "changed": False,
        "action_error": None,
        "render_error": None,
        "render_calls": 0,
        "render_seconds": 0.0,
    }
    try:
        if action == {"action": "stop"}:
            result.update(applied=True, edit_scope="stop")
            return result
        if action.get("action") == "global_patch":
            if set(action) != {"action", "html"} or not isinstance(action["html"], str):
                raise ValueError("Global patch requires exactly action and complete HTML")
            _, candidate = parse_table(action["html"])
            result["edit_scope"] = "global"
        else:
            if initial is None:
                raise ValueError("Local teacher edit requires an existing HTML candidate")
            candidate, _ = apply_table_action(initial, json.dumps(action, ensure_ascii=False))
            result["edit_scope"] = "local"
        result["candidate_html"] = candidate
    except (ValueError, TypeError) as error:
        result["action_error"] = f"{type(error).__name__}: {str(error)[:300]}"
        return result
    result["render_calls"] = 1
    started = time.perf_counter()
    try:
        rendered = renderer(Observation(decision["case_id"], "table", "", candidate))
        if rendered.prediction_sha256 != digest(candidate):
            raise ValueError("Renderer returned stale evidence for another HTML state")
        result.update(
            render_path=rendered.path,
            render_backend=rendered.backend,
            render_prediction_sha256=rendered.prediction_sha256,
            render_image_sha256=sha256(Path(rendered.path)),
            applied=True,
            changed=candidate != initial,
            final_html=candidate,
        )
    except Exception as error:
        # External renderer failures remain explicit and retain their attempted cost.
        result["render_error"] = f"{type(error).__name__}: {str(error)[:300]}"
    finally:
        result["render_seconds"] = time.perf_counter() - started
    return result
