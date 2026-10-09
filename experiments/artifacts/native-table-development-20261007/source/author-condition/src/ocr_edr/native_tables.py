"""Source-only native table recognition and complete offline coverage checks."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from statistics import mean

from .sft import sha256


def load_native_table_sources(dataset_root: Path) -> list[dict]:
    root = dataset_root.resolve()
    meta = json.loads((root / "dataset.json").read_text())
    paths = [root / "model_dev-source-inputs.jsonl", root / "selected_sources.jsonl"]
    for path in paths:
        if sha256(path) != meta["file_sha256"][path.name]:
            raise ValueError("Frozen native source identity mismatch")
    rows, sources = [
        [json.loads(line) for line in p.read_text().splitlines() if line.strip()] for p in paths
    ]
    inventory = {r["family_id"]: r for r in sources}
    expected = meta["roles"]["model_dev"]["sources"]
    if not rows or len(rows) != expected or len({r["family_id"] for r in rows}) != expected:
        raise ValueError("Every unique frozen model-dev source is required")
    if len(inventory) != len(sources):
        raise ValueError("Duplicate source inventory identity")
    documents = set()
    for row in rows:
        if set(row) != {"family_id", "source_image", "source_sha256"}:
            raise ValueError("Recognizer may receive only source-image identity")
        source = inventory.get(row["family_id"])
        if source is None or source["role"] != "model_dev":
            raise ValueError("Native recognizer source role mismatch")
        image = (root / row["source_image"]).resolve()
        image.relative_to(root)
        if (
            image != (root / source["source_image"]).resolve()
            or row["source_sha256"] != source["source_sha256"]
            or sha256(image) != row["source_sha256"]
        ):
            raise ValueError("Native recognizer source-image hash/path mismatch")
        documents.add(source["document_id"])
    if len(documents) != expected:
        raise ValueError("Expected one native source per original document")
    return rows


def recognize_table_source(row: dict, source_root: Path, *, recognize) -> dict:
    """One model attempt, retaining expected external inference failures and cost."""
    result = {**row, "prediction": None, "details": None, "parser_error": None, "pipeline_calls": 1}
    started = time.perf_counter()
    try:
        prediction, details = recognize(source_root / row["source_image"])
        if not isinstance(prediction, str) or not prediction.strip():
            raise ValueError("Native parser returned no nonempty HTML string")
        result.update(prediction=prediction, details=details)
    except Exception as error:
        # Every attempted source remains represented; do not retry/select on quality.
        result["parser_error"] = f"{type(error).__name__}: {str(error)[:600]}"
    result["generation_seconds"] = time.perf_counter() - started
    return result


def evaluate_native_tables(predictions, inputs, references, *, normalize, score):
    expected = {r["family_id"]: r for r in inputs}
    if not expected or len(expected) != len(inputs):
        raise ValueError("Unique nonempty source inputs required")
    refs, documents = {}, {}
    for ref in references:
        if ref["role"] != "model_dev":
            raise ValueError("Only model-dev references allowed")
        identity = (ref["document_id"], ref["reference"])
        if refs.setdefault(ref["family_id"], identity) != identity:
            raise ValueError("Inconsistent reference within a source family")
        if documents.setdefault(ref["document_id"], ref["family_id"]) != ref["family_id"]:
            raise ValueError("Multiple native sources in one document")
    if set(refs) != set(expected):
        raise ValueError("Reference/source coverage mismatch")

    def checked_score(prediction, reference):
        values = score(prediction, reference)
        if set(values) != {"teds", "teds_structure"} or any(
            not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()
        ):
            raise ValueError("Invalid native table metric result")
        return values

    seen, rows = set(), []
    for result in predictions:
        family = result["family_id"]
        if family not in expected or family in seen:
            raise ValueError("Unexpected/duplicate native prediction identity")
        seen.add(family)
        if any(result.get(k) != v for k, v in expected[family].items()):
            raise ValueError("Native prediction source identity drift")
        if (
            type(result["pipeline_calls"]) is not int
            or result["pipeline_calls"] != 1
            or not math.isfinite(result["generation_seconds"])
            or result["generation_seconds"] < 0
        ):
            raise ValueError("Invalid parser-call accounting")
        document, reference = refs[family]
        ref_norm = normalize(reference)
        if any(v != 1 for v in checked_score(ref_norm, ref_norm).values()):
            raise ValueError("Reference not ready for the selected official metric")
        failed = bool(result["parser_error"])
        if failed:
            if result["prediction"] is not None:
                raise ValueError("Failed inference cannot carry a successful prediction")
            values = {"teds": 0.0, "teds_structure": 0.0}
        else:
            if not isinstance(result["prediction"], str) or not result["prediction"].strip():
                raise ValueError("Missing prediction without explicit parser failure")
            values = checked_score(normalize(result["prediction"]), ref_norm)
        rows.append(
            {
                "family_id": family,
                "document_id": document,
                **values,
                "parser_failure": failed,
                "pipeline_calls": 1,
                "generation_seconds": result["generation_seconds"],
            }
        )
    if seen != set(expected):
        raise ValueError("Incomplete native coverage; never drop failed sources")
    summary = {
        "sources": len(rows),
        "documents": len(documents),
        "parser_failures": sum(r["parser_failure"] for r in rows),
        "pipeline_calls": sum(r["pipeline_calls"] for r in rows),
        "generation_seconds": sum(r["generation_seconds"] for r in rows),
    }
    for metric in ["teds", "teds_structure"]:
        summary["mean_" + metric] = mean(r[metric] for r in rows)
        summary[metric + "_full_match"] = sum(r[metric] >= 1 - 1e-12 for r in rows)
    return {"summary": summary, "cases": rows}


def native_editor_profile(prediction: str, reference: str) -> dict:
    """Offline raw-DOM invariants, not an exhaustive TEDS repair oracle."""
    from .table_pilot import parse_table, table_cell_map

    ref_tree, _ = parse_table(reference)
    ref_rows = ref_tree.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr")
    ref_cells = table_cell_map(reference)
    result = {
        "reference_rows": len(ref_rows),
        "reference_cells": len(ref_cells),
        "editor_input_valid": False,
        "aligned_plain_text_mismatches": None,
        "aligned_span_mismatches": None,
    }
    try:
        tree, _ = parse_table(prediction)
    except (ValueError, TypeError) as error:
        result["editor_input_error"] = f"{type(error).__name__}: {str(error)[:200]}"
        return result
    rows = tree.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr")
    cells = table_cell_map(prediction)
    result.update(
        editor_input_valid=True,
        prediction_rows=len(rows),
        prediction_cells=len(cells),
        requires_row_insertion_for_reference_dom=len(rows) < len(ref_rows),
        requires_cell_insertion_for_reference_dom=len(cells) < len(ref_cells),
    )

    def cell_counts(table_rows):
        return [len(row.xpath("./td|./th")) for row in table_rows]

    if cell_counts(rows) == cell_counts(ref_rows):
        result["aligned_plain_text_mismatches"] = sum(
            a["text"] != b["text"] for a, b in zip(cells, ref_cells)
        )
        result["aligned_span_mismatches"] = sum(
            (a["rowspan"], a["colspan"]) != (b["rowspan"], b["colspan"])
            for a, b in zip(cells, ref_cells)
        )
    return result
