"""Bounded table-action prompts separate current evidence from offline targets."""

from __future__ import annotations

import json

from .table_pilot import apply_table_action, parse_table, table_cell_map


def table_action_prompt(initial: str) -> str:
    parse_table(initial)
    addresses = json.dumps(table_cell_map(initial), ensure_ascii=False, separators=(",", ":"))
    return (
        "Image 1 is the source table. Compare it with the current OCR HTML. "
        "Correct only visible differences. Preserve every unaffected cell, row and span. "
        "If no correction is supported, return stop; request no external text or reference.\n"
        "Current HTML:\n" + initial + "\n"
        "Addresses derived from this current HTML only, zero-based row and cell indices:\n"
        "<cell_map>" + addresses + "</cell_map>\n"
        "Return exactly one JSON object, no prose or additional fields. Allowed fields by action: "
        "stop: action only; replace_cell: action,row,cell,text; "
        "set_span: action,row,cell,rowspan,colspan; delete_row: action,row. "
        "action is one of stop/replace_cell/set_span/delete_row. "
        "row and cell are existing integer indices in the current DOM; spans are positive integers; "
        "text is the exact plain cell content supported by the image. "
        "Do not copy field descriptions as cell content. Perform at most one atomic edit."
    )


def validate_table_target(initial: str, target_json: str, reference: str) -> dict:
    """Offline training-target check; never call it in deployment acceptance."""
    final, action = apply_table_action(initial, target_json)
    if final != parse_table(reference)[1]:
        raise ValueError("Supervision action does not restore the frozen canonical reference")
    return action
