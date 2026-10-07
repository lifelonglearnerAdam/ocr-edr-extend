"""Descriptive source-group statistics for fixed development CDM pairs."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import mean


def paired_source_bootstrap(
    rows: list[dict], arm: str, comparator: str, *, seed: int, repeats: int = 2000
) -> dict:
    if type(repeats) is not int or repeats < 100:
        raise ValueError("At least 100 bootstrap resamples required")
    left, right = {}, {}
    for row in rows:
        target = left if row["arm"] == arm else right if row["arm"] == comparator else None
        if target is not None:
            if row["sample_id"] in target:
                raise ValueError("Duplicate paired statistic record")
            target[row["sample_id"]] = row
    if not left or left.keys() != right.keys():
        raise ValueError("Incomplete paired comparison")
    sources = defaultdict(list)
    for sample, row in left.items():
        other = right[sample]
        if row["family_id"] != other["family_id"] or row["initial_cdm"] != other["initial_cdm"]:
            raise ValueError("Source or initial-score drift between paired arms")
        delta = row["final_cdm"]["F1_score"] - other["final_cdm"]["F1_score"]
        if not math.isfinite(delta):
            raise ValueError("Nonfinite paired score")
        sources[row["family_id"]].append(delta)
    values = [mean(group) for group in sources.values()]
    rng = random.Random(seed)
    distribution = sorted(mean(rng.choices(values, k=len(values))) for _ in range(repeats))
    return {
        "arm": arm,
        "comparator": comparator,
        "source_groups": len(values),
        "case_pairs": len(left),
        "mean_delta_vs_comparator": mean(values),
        "descriptive_percentile_95_interval": [
            distribution[int(0.025 * repeats)],
            distribution[min(repeats - 1, int(0.975 * repeats))],
        ],
        "resampling_unit": "source_image family, not repeated variants; original document IDs unavailable",
        "seed": seed,
        "resamples": repeats,
        "interpretation": "single-seed inspected development result; not a confirmatory significance claim",
    }


def variant_summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["arm"], row["variant"])].append(row)
    summaries = []
    for (arm, variant), cases in groups.items():
        good = [r for r in cases if r["initial_cdm"]["F1_score"] == 1.0]
        bad = [r for r in cases if r["initial_cdm"]["F1_score"] < 1.0]
        regression = sum(r["final_cdm"]["F1_score"] < 1.0 for r in good)
        summaries.append(
            {
                "arm": arm,
                "variant": variant,
                "cases": len(cases),
                "source_groups": len({r["family_id"] for r in cases}),
                "initial_mean_cdm": mean(r["initial_cdm"]["F1_score"] for r in cases),
                "final_mean_cdm": mean(r["final_cdm"]["F1_score"] for r in cases),
                "mean_delta_cdm": mean(r["delta_cdm"] for r in cases),
                "initial_metric_matching": len(good),
                "initial_metric_nonmatching": len(bad),
                "matching_regressions": regression,
                "nonmatching_fixed": sum(r["final_cdm"]["F1_score"] == 1 for r in bad),
                "preserve_metric_proxy": 1 - regression / len(good) if good else None,
                "improved": sum(r["delta_cdm"] > 0 for r in cases),
                "degraded": sum(r["delta_cdm"] < 0 for r in cases),
                "changed": sum(r["changed"] for r in cases),
            }
        )
    return summaries
