"""Fixed-pair table scores and preservation/cost summaries, evaluated offline."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean

EPSILON = 1e-12
METRICS = ("teds", "teds_structure")


def _unique(rows: list[dict], label: str) -> dict:
    by_id = {row["sample_id"]: row for row in rows}
    if not rows or len(by_id) != len(rows):
        raise ValueError(f"Empty or duplicate {label} sample IDs")
    return by_id


def evaluate_tables(
    predictions: list[dict],
    inputs: list[dict],
    references: list[dict],
    arms: list[str],
    *,
    normalize,
    score,
) -> dict:
    """Require all frozen cases/arms; labels never participate in proposal generation.

    `normalize` and `score(normalized_prediction, normalized_reference)` are the
    pinned upstream functions in production. Injection permits aggregation tests
    without adding optional metric dependencies to the core environment.
    """
    by_input = _unique(inputs, "input")
    by_ref = _unique(references, "reference")
    if set(by_input) != set(by_ref):
        raise ValueError("Input/reference coverage mismatch")
    if not arms or len(set(arms)) != len(arms) or "unchanged_0" not in arms:
        raise ValueError("Expected distinct frozen arms including unchanged_0")
    families, pages = {}, {}
    for sample_id, initial in by_input.items():
        ref = by_ref[sample_id]
        family, page = ref["family_id"], ref["parent_page"]
        if initial["family_id"] != family:
            raise ValueError("Input/reference family mismatch")
        if families.setdefault(family, page) != page or pages.setdefault(page, family) != family:
            raise ValueError("Inconsistent frozen family/page mapping")
    expected = {(sample_id, arm) for sample_id in by_input for arm in arms}
    seen, rows = set(), []
    normalized_cache, score_cache = {}, {}

    def normalize_once(markup):
        if markup not in normalized_cache:
            normalized_cache[markup] = normalize(markup)
        return normalized_cache[markup]

    def score_once(prediction, reference):
        key = (prediction, reference)
        if key not in score_cache:
            scores = score(prediction, reference)
            if set(scores) != set(METRICS):
                raise ValueError("Unexpected official metric names")
            if any(
                not math.isfinite(v) or not -EPSILON <= v <= 1 + EPSILON for v in scores.values()
            ):
                raise ValueError("Nonfinite or out-of-range official metric")
            score_cache[key] = scores
        return score_cache[key]

    for result in predictions:
        key = (result["sample_id"], result["arm"])
        if key not in expected or key in seen:
            raise ValueError("Unexpected or duplicate prediction sample/arm")
        seen.add(key)
        initial = by_input[result["sample_id"]]
        ref = by_ref[result["sample_id"]]
        if result["family_id"] != ref["family_id"]:
            raise ValueError("Prediction/reference family mismatch")
        if result["initial_prediction"] != initial["prediction"]:
            raise ValueError("Initial prediction changed from frozen input")
        trace = result["trace"]
        if result["arm"] == "unchanged_0" and (
            result["final_prediction"] != initial["prediction"] or trace
        ):
            raise ValueError("Unchanged baseline was modified or incurred model calls")
        if len(trace) > 1:
            raise ValueError("This protocol allows at most one model call per sample/arm")
        if result["arm"] != "unchanged_0" and not trace and not result.get("skip_reason"):
            raise ValueError("Missing model call without an explicit skip reason")
        rejected = any(c.get("adapter_error") or c.get("render_error") for c in trace)
        if (rejected or result.get("skip_reason")) and result["final_prediction"] != initial[
            "prediction"
        ]:
            raise ValueError("Rejected or skipped proposal did not roll back")
        reference = normalize_once(ref["reference"])
        before = normalize_once(initial["prediction"])
        after = normalize_once(result["final_prediction"])
        before_scores, after_scores = score_once(before, reference), score_once(after, reference)
        row = {
            "sample_id": result["sample_id"],
            "family_id": ref["family_id"],
            "parent_page": ref["parent_page"],
            "annotation_id": ref["annotation_id"],
            "gt_position": ref["gt_position"],
            "variant": ref["variant"],
            "arm": result["arm"],
            "changed": result["final_prediction"] != initial["prediction"],
            "normalized_changed": before != after,
            "initial_normalized_html": before,
            "final_normalized_html": after,
            "reference_normalized_html": reference,
            "model_calls": len(trace),
            "adapter_failures": sum(bool(c.get("adapter_error")) for c in trace),
            "render_rejections": sum(bool(c.get("render_error")) for c in trace),
            "initial_render_failure": bool(result.get("initial_render_error")),
            "skipped": bool(result.get("skip_reason")),
            "hit_length_cap": sum(bool(c.get("hit_length_cap")) for c in trace),
            "generation_seconds": sum(c["generation_seconds"] for c in trace),
            "input_tokens": sum(c["input_tokens"] for c in trace),
            "output_tokens": sum(c["output_tokens"] for c in trace),
            "stop_action": any((c.get("action") or {}).get("action") == "stop" for c in trace),
        }
        for metric in METRICS:
            row[f"initial_{metric}"] = before_scores[metric]
            row[f"final_{metric}"] = after_scores[metric]
            row[f"delta_{metric}"] = after_scores[metric] - before_scores[metric]
        rows.append(row)
    if seen != expected:
        raise ValueError(
            "Incomplete frozen sample/arm coverage; partial results cannot be compared"
        )

    def summarize(group: list[dict], arm: str, variant: str) -> tuple[dict, list]:
        row = {"arm": arm, "variant": variant, "cases": len(group)}
        page_groups = defaultdict(list)
        for case in group:
            page_groups[case["parent_page"]].append(case)
        row["pages"] = len(page_groups)
        page_rows = []
        for page, cases in page_groups.items():
            page_row = {"arm": arm, "variant": variant, "parent_page": page, "cases": len(cases)}
            for metric in METRICS:
                for phase in ("initial", "final", "delta"):
                    page_row[f"{phase}_{metric}"] = mean(c[f"{phase}_{metric}"] for c in cases)
            page_rows.append(page_row)
        for metric in METRICS:
            for phase in ("initial", "final", "delta"):
                row[f"case_mean_{phase}_{metric}"] = mean(c[f"{phase}_{metric}"] for c in group)
                row[f"page_mean_{phase}_{metric}"] = mean(c[f"{phase}_{metric}"] for c in page_rows)
            matching = [c for c in group if c[f"initial_{metric}"] >= 1 - EPSILON]
            nonmatching = [c for c in group if c[f"initial_{metric}"] < 1 - EPSILON]
            row[f"{metric}_initial_matching_n"] = len(matching)
            row[f"{metric}_initial_nonmatching_n"] = len(nonmatching)
            row[f"{metric}_matching_regressions"] = sum(
                c[f"final_{metric}"] < 1 - EPSILON for c in matching
            )
            row[f"{metric}_nonmatching_fixed"] = sum(
                c[f"final_{metric}"] >= 1 - EPSILON for c in nonmatching
            )
            row[f"{metric}_improved"] = sum(c[f"delta_{metric}"] > EPSILON for c in group)
            row[f"{metric}_degraded"] = sum(c[f"delta_{metric}"] < -EPSILON for c in group)
            row[f"{metric}_final_matching"] = sum(
                c[f"final_{metric}"] >= 1 - EPSILON for c in group
            )
        for field in [
            "changed",
            "normalized_changed",
            "model_calls",
            "adapter_failures",
            "render_rejections",
            "initial_render_failure",
            "skipped",
            "hit_length_cap",
            "stop_action",
            "generation_seconds",
            "input_tokens",
            "output_tokens",
        ]:
            row[field] = sum(c[field] for c in group)
        row["mean_generation_seconds"] = row["generation_seconds"] / len(group)
        row["mean_input_tokens"] = row["input_tokens"] / len(group)
        row["mean_output_tokens"] = row["output_tokens"] / len(group)
        return row, page_rows

    summaries, per_page = [], []
    variants = ["all", *dict.fromkeys(ref["variant"] for ref in references)]
    for arm in arms:
        for variant in variants:
            group = [
                r for r in rows if r["arm"] == arm and (variant == "all" or r["variant"] == variant)
            ]
            summary, page_rows = summarize(group, arm, variant)
            summaries.append(summary)
            per_page.extend(page_rows)
    return {"summary": summaries, "per_page": per_page, "cases": rows}
