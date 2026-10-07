"""Offline document-level TEDS and action-locality analysis for controlled tables."""

from __future__ import annotations

import json

from .table_evaluation import evaluate_tables
from .table_supervision import validate_table_target

ACTION_FIELDS = (
    "actions_with_valid_schema",
    "accepted_actions",
    "exact_target_actions",
    "target_address_matches",
    "render_attempts",
    "render_seconds",
)


def evaluate_table_screen(predictions, inputs, references, arms, *, normalize, score):
    """The reference action is diagnostic only; it cannot change a frozen proposal."""
    by_input = {row["sample_id"]: row for row in inputs}
    refs = []
    for ref in references:
        if ref["role"] != "model_dev":
            raise ValueError("Only model-dev references belong in this development screen")
        initial = by_input[ref["sample_id"]]["prediction"]
        validate_table_target(initial, json.dumps(ref["target_action"]), ref["reference"])
        # The older aggregator calls its grouping unit a page. Here the unit is
        # an original PMC document, with one source table per document by design.
        refs.append(
            {**ref, "parent_page": ref["document_id"], "annotation_id": None, "gt_position": None}
        )
    result = evaluate_tables(predictions, inputs, refs, arms, normalize=normalize, score=score)
    by_ref = {ref["sample_id"]: ref for ref in references}
    by_prediction = {(row["sample_id"], row["arm"]): row for row in predictions}
    for row in result["cases"]:
        row["document_id"] = row.pop("parent_page")
        row.pop("annotation_id")
        row.pop("gt_position")
        target = by_ref[row["sample_id"]]["target_action"]
        trace = by_prediction[row["sample_id"], row["arm"]]["trace"]
        call = trace[0] if trace else {}
        action = call.get("action")
        address_keys = {"action", "row", "cell"} & target.keys()
        row.update(
            target_action=target,
            proposed_action=action,
            actions_with_valid_schema=action is not None,
            accepted_actions=bool(action)
            and not (call.get("adapter_error") or call.get("render_error")),
            exact_target_actions=action == target,
            target_address_matches=bool(action)
            and all(action.get(k) == target[k] for k in address_keys),
            render_attempts=call.get("render_attempts", 0),
            render_seconds=call.get("render_seconds", 0.0),
        )
    for summary in result["summary"]:
        group = [
            row
            for row in result["cases"]
            if row["arm"] == summary["arm"]
            and (summary["variant"] == "all" or row["variant"] == summary["variant"])
        ]
        summary["documents"] = summary.pop("pages")
        for key in list(summary):
            if key.startswith("page_mean_"):
                summary[key.replace("page_mean_", "document_mean_", 1)] = summary.pop(key)
        summary.update({key: sum(row[key] for row in group) for key in ACTION_FIELDS})
    result["per_document"] = result.pop("per_page")
    for row in result["per_document"]:
        row["document_id"] = row.pop("parent_page")
    return result
