"""Fixed-pair formula scoring with reference-render readiness checks."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean

METRIC_FIELDS = ("recall", "precision", "F1_score", "tp", "gt_tokens", "pred_tokens")


def evaluate_cdm(
    predictions: list[dict], references: list[dict], arms: list[str], *, metric
) -> dict:
    """Score all frozen pairs, failing closed on unavailable reference rendering.

    ``metric`` is the upstream cdm_metrics callable with its runtime/seed bound.
    Failed/empty predictions retain zero scores when the reference rendered.
    No string rewriting, reference-dependent candidate selection, or model call occurs.
    """
    refs = {r["sample_id"]: r for r in references}
    if not refs or len(refs) != len(references) or not arms or len(set(arms)) != len(arms):
        raise ValueError("Empty or duplicate CDM reference/arm coverage")
    seen, initial = set(), {}
    for row in predictions:
        key = (row["sample_id"], row["arm"])
        if key in seen or key[0] not in refs or key[1] not in arms:
            raise ValueError("Duplicate or unknown CDM sample/arm pair")
        ref = refs[key[0]]
        if row["family_id"] != ref["family_id"]:
            raise ValueError("CDM prediction/reference family mismatch")
        if key[0] in initial and initial[key[0]] != row["initial_prediction"]:
            raise ValueError("CDM initial prediction drift between arms")
        initial[key[0]] = row["initial_prediction"]
        seen.add(key)
    if seen != {(s, a) for s in refs for a in arms}:
        raise ValueError("Incomplete CDM sample/arm coverage")

    cache = {}

    def score(reference, prediction):
        if not isinstance(reference, str) or not isinstance(prediction, str):
            raise ValueError("CDM requires frozen string predictions/references")
        key = (reference, prediction)
        if key not in cache:
            result = metric(reference, prediction)
            if not isinstance(result, dict) or not all(k in result for k in METRIC_FIELDS):
                raise ValueError("Incomplete official CDM metric result")
            result = {k: result[k] for k in METRIC_FIELDS}
            for field in METRIC_FIELDS[:3]:
                value = result[field]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("Invalid official CDM metric value")
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Invalid official CDM metric value")
            for field in METRIC_FIELDS[3:]:
                value = result[field]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError("Invalid official CDM token count")
            if result["tp"] > min(result["gt_tokens"], result["pred_tokens"]):
                raise ValueError("Invalid official CDM matched token count")
            cache[key] = result
        return cache[key]

    readiness = []
    for ref in references:
        values = score(ref["reference"], ref["reference"])
        if values["gt_tokens"] == 0 or values["F1_score"] != 1.0:
            raise ValueError(f"Reference CDM readiness failed: {ref['sample_id']}: {values}")
        readiness.append({"sample_id": ref["sample_id"], "metrics": values})
    ready = {r["sample_id"]: r["metrics"] for r in readiness}
    cases, groups = [], defaultdict(list)
    for row in predictions:
        ref = refs[row["sample_id"]]
        before = score(ref["reference"], row["initial_prediction"])
        after = score(ref["reference"], row["final_prediction"])
        if any(r["gt_tokens"] != ready[row["sample_id"]]["gt_tokens"] for r in [before, after]):
            raise ValueError(f"CDM reference token count changed: {row['sample_id']}")
        case = {
            "sample_id": row["sample_id"],
            "family_id": row["family_id"],
            "arm": row["arm"],
            "variant": ref.get("variant", "unspecified"),
            "initial_cdm": before,
            "final_cdm": after,
            "delta_cdm": after["F1_score"] - before["F1_score"],
            "changed": row["initial_prediction"] != row["final_prediction"],
            "initial_prediction_unrenderable_or_empty": before["pred_tokens"] == 0,
            "final_prediction_unrenderable_or_empty": after["pred_tokens"] == 0,
        }
        cases.append(case)
        groups[row["arm"]].append(case)
    summaries = []
    for arm in arms:
        rows = groups[arm]
        sources = defaultdict(list)
        for row in rows:
            sources[row["family_id"]].append(row)
        summaries.append(
            {
                "arm": arm,
                "cases": len(rows),
                "sources": len(sources),
                "case_mean_initial_cdm": mean(r["initial_cdm"]["F1_score"] for r in rows),
                "case_mean_final_cdm": mean(r["final_cdm"]["F1_score"] for r in rows),
                "case_mean_delta_cdm": mean(r["delta_cdm"] for r in rows),
                "source_mean_initial_cdm": mean(
                    mean(r["initial_cdm"]["F1_score"] for r in rs) for rs in sources.values()
                ),
                "source_mean_final_cdm": mean(
                    mean(r["final_cdm"]["F1_score"] for r in rs) for rs in sources.values()
                ),
                "source_mean_delta_cdm": mean(
                    mean(r["delta_cdm"] for r in rs) for rs in sources.values()
                ),
                "initial_matching": sum(r["initial_cdm"]["F1_score"] == 1.0 for r in rows),
                "initial_nonmatching": sum(r["initial_cdm"]["F1_score"] < 1.0 for r in rows),
                "matching_regressions": sum(
                    r["initial_cdm"]["F1_score"] == 1.0 and r["final_cdm"]["F1_score"] < 1.0
                    for r in rows
                ),
                "nonmatching_fixed": sum(
                    r["initial_cdm"]["F1_score"] < 1.0 and r["final_cdm"]["F1_score"] == 1.0
                    for r in rows
                ),
                "improved": sum(r["delta_cdm"] > 0 for r in rows),
                "degraded": sum(r["delta_cdm"] < 0 for r in rows),
                "final_matching": sum(r["final_cdm"]["F1_score"] == 1.0 for r in rows),
                "changed": sum(r["changed"] for r in rows),
                "initial_prediction_unrenderable_or_empty": sum(
                    r["initial_prediction_unrenderable_or_empty"] for r in rows
                ),
                "final_prediction_unrenderable_or_empty": sum(
                    r["final_prediction_unrenderable_or_empty"] for r in rows
                ),
            }
        )
    return {
        "cases": cases,
        "summary": summaries,
        "reference_readiness": readiness,
        "metric_calls": len(cache),
    }
