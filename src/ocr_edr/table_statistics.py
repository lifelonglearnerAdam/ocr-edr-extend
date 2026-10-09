"""Paired original-document analysis for complete, fixed table development arms."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import mean

ARMS = ("unchanged_0", "base", "all", "no_explicit_preservation")
CONTRASTS = (
    ("base", "unchanged_0"),
    ("all", "base"),
    ("no_explicit_preservation", "base"),
    ("all", "no_explicit_preservation"),
)
METRICS = ("teds", "teds_structure")
COSTS = (
    "model_calls",
    "input_tokens",
    "output_tokens",
    "generation_seconds",
    "render_attempts",
    "render_seconds",
)
EPSILON = 1e-12


def _validated_pairs(rows):
    indexed = {arm: {} for arm in ARMS}
    for row in rows:
        arm, sample = row["arm"], row["sample_id"]
        if arm not in indexed or sample in indexed[arm]:
            raise ValueError("Unknown arm or duplicate table statistic record")
        if any(
            not isinstance(row[field], str) or not row[field]
            for field in ["sample_id", "family_id", "document_id", "variant"]
        ):
            raise ValueError("Nonempty sample/source/document/variant identities required")
        for metric in METRICS:
            before, after = row["initial_" + metric], row["final_" + metric]
            if any(
                type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1
                for value in [before, after]
            ):
                raise ValueError("Nonfinite or invalid table quality score")
            delta = row["delta_" + metric]
            if not math.isfinite(delta) or abs(delta - (after - before)) > EPSILON:
                raise ValueError("Frozen score delta is inconsistent")
        for field in COSTS:
            value = row[field]
            if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid nonnegative recorded cost")
            if field not in {"generation_seconds", "render_seconds"} and type(value) is not int:
                raise ValueError("Call/token accounting must use integers")
        indexed[arm][sample] = row
    baseline = indexed["unchanged_0"]
    if not baseline or any(set(indexed[arm]) != set(baseline) for arm in ARMS):
        raise ValueError("Complete paired case coverage across all four arms required")
    families, documents = {}, {}
    for sample in sorted(baseline):
        initial = baseline[sample]
        doc, family = initial["document_id"], initial["family_id"]
        if families.setdefault(family, doc) != doc or documents.setdefault(doc, family) != family:
            raise ValueError("Expected one source family per original document")
        if any(initial["delta_" + metric] != 0 for metric in METRICS) or any(
            initial[k] != 0 for k in COSTS
        ):
            raise ValueError("Unchanged baseline must remain unchanged and incur no calls/cost")
        for arm in ARMS:
            row = indexed[arm][sample]
            for key in [
                "family_id",
                "document_id",
                "variant",
                "initial_teds",
                "initial_teds_structure",
            ]:
                if row[key] != initial[key]:
                    raise ValueError("Paired identity/variant/initial-score drift")
    return indexed


def _document_values(rows, value):
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["sample_id"]):
        groups[row["document_id"]].append(value(row))
    return [mean(groups[doc]) for doc in sorted(groups)]


def _interval(values, *, seed, repeats):
    rng = random.Random(seed)
    bootstrap = sorted(mean(rng.choices(values, k=len(values))) for _ in range(repeats))
    return [bootstrap[int(0.025 * repeats)], bootstrap[min(repeats - 1, int(0.975 * repeats))]]


def summarize_table_statistics(rows, *, seed=20261007, repeats=10000):
    if type(seed) is not int or type(repeats) is not int or repeats < 100:
        raise ValueError("Integer seed and at least 100 bootstrap repetitions required")
    indexed = _validated_pairs(rows)
    samples = sorted(indexed["unchanged_0"])
    variants = ["all", *sorted({indexed["unchanged_0"][s]["variant"] for s in samples})]
    contrasts, utilities, costs = [], [], []
    for variant in variants:
        selected = [
            s
            for s in samples
            if variant == "all" or indexed["unchanged_0"][s]["variant"] == variant
        ]
        for arm, comparator in CONTRASTS:
            left = [indexed[arm][s] for s in selected]
            for metric in METRICS:

                def difference(row):
                    return (
                        row["final_" + metric]
                        - indexed[comparator][row["sample_id"]]["final_" + metric]
                    )

                values = _document_values(left, difference)
                contrasts.append(
                    {
                        "arm": arm,
                        "comparator": comparator,
                        "variant": variant,
                        "metric": metric,
                        "documents": len(values),
                        "case_pairs": len(left),
                        "document_mean_delta": mean(values),
                        "case_mean_delta": mean(difference(row) for row in left),
                        "descriptive_percentile_95_interval": _interval(
                            values, seed=seed, repeats=repeats
                        ),
                    }
                )
    for arm in ARMS:
        group = [indexed[arm][s] for s in samples]
        documents = len({row["document_id"] for row in group})
        matching = [row for row in group if row["initial_teds"] >= 1 - EPSILON]
        nonmatching = [row for row in group if row["initial_teds"] < 1 - EPSILON]
        for weight in [1, 2, 4]:

            def utility(row):
                regression = row["initial_teds"] >= 1 - EPSILON and row["final_teds"] < 1 - EPSILON
                return row["delta_teds"] - weight * int(regression)

            utilities.append(
                {
                    "arm": arm,
                    "regression_weight": weight,
                    "documents": documents,
                    "cases": len(group),
                    "document_mean_utility": mean(_document_values(group, utility)),
                    "case_mean_utility": mean(utility(row) for row in group),
                    "initial_matching_cases": len(matching),
                    "matching_regressions": sum(
                        row["final_teds"] < 1 - EPSILON for row in matching
                    ),
                    "initial_nonmatching_cases": len(nonmatching),
                    "nonmatching_fully_fixed": sum(
                        row["final_teds"] >= 1 - EPSILON for row in nonmatching
                    ),
                }
            )
        cost = {"arm": arm, "documents": documents, "cases": len(group)}
        for field in COSTS:
            cost[field + "_total"] = sum(row[field] for row in group)
            cost[field + "_case_mean"] = mean(row[field] for row in group)
            cost[field + "_document_mean"] = mean(_document_values(group, lambda row: row[field]))
        costs.append(cost)
    return {
        "documents": len({row["document_id"] for row in indexed["unchanged_0"].values()}),
        "cases_per_arm": len(samples),
        "arms": list(ARMS),
        "contrasts": contrasts,
        "metric_proxy_utility": utilities,
        "costs": costs,
        "seed": seed,
        "resamples": repeats,
        "resampling_unit": "original PMC document; mean within document then equal mean across documents",
        "interpretation": "single-seed inspected development; descriptive intervals without multiplicity correction; no confirmatory or visual-correctness claim",
        "utility_formula": "document_mean(case_mean(delta_TEDS - lambda * initial_metric_match_regression)); no cost conversion and not an RL reward validation",
        "cost_scope": "generation excludes initialization/preprocessing; render replay uses a content cache; not end-to-end latency",
    }
