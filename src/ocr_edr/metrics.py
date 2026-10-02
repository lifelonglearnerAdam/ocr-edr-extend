"""Read and compare official OmniDocBench artifacts without reimplementing CDM/TEDS."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean

MODALITIES = {"formula": ("display_formula", "CDM"), "table": ("table", "TEDS")}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def result_path(directory: Path, suffix: str, prefix: str | None = None) -> Path:
    paths = sorted(directory.glob(f"{prefix or '*'}_{suffix}.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one {suffix} artifact in {directory}; found {len(paths)}")
    return paths[0]


def sample_key(row: dict) -> str:
    # Exclude prediction indices: editing can change them. Include reference identity
    # so changed splits/matches cannot silently turn into a favorable paired subset.
    value = [row["img_id"], row["gt_idx"], row["gt_position"], row.get("norm_gt", row["gt"])]
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def indexed_scores(rows: list[dict], metric: str) -> dict[str, tuple[dict, float]]:
    indexed = {}
    for row in rows:
        value = row["metric"][metric]
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError(f"Invalid {metric} score")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid {metric} score: {value}")
        key = sample_key(row)
        if key in indexed:
            raise ValueError("Duplicate official reference match identity")
        indexed[key] = (row, float(value))
    return indexed


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def compare_results(
    before_dir: Path,
    after_dir: Path | None = None,
    before_prefix: str | None = None,
    after_prefix: str | None = None,
    good_threshold: float = 1.0,
) -> dict:
    if not math.isfinite(good_threshold) or not 0 < good_threshold <= 1:
        raise ValueError("Good threshold must be in (0, 1]")
    control = after_dir is None
    after_dir = after_dir or before_dir
    if control:
        after_prefix = before_prefix
    summary = {
        "schema_version": 1,
        "run_type": "baseline_self_comparison" if control else "paired_evaluation",
        "good_definition": f"official per-match metric >= {good_threshold}; a proxy, not a visual label",
        "vis_fix": None,
        "modalities": {},
        "source_artifacts": [],
    }
    before_metrics = read_json(result_path(before_dir, "metric_result", before_prefix))
    after_metrics = read_json(result_path(after_dir, "metric_result", after_prefix))
    for modality, (category, metric) in MODALITIES.items():
        before_path = result_path(before_dir, f"{category}_result", before_prefix)
        after_path = result_path(after_dir, f"{category}_result", after_prefix)
        before = indexed_scores(read_json(before_path), metric)
        after = indexed_scores(read_json(after_path), metric)
        if before.keys() != after.keys():
            raise ValueError(
                f"{modality}: reference matching changed ({len(before.keys() - after.keys())} missing, "
                f"{len(after.keys() - before.keys())} added). Establish fixed matches before paired reporting."
            )
        good = [key for key, (_, value) in before.items() if value >= good_threshold]
        bad = [key for key in before if key not in good]
        initially_nonexact = [
            key
            for key, (row, _) in before.items()
            if row.get("norm_gt") is not None
            and row.get("norm_pred") is not None
            and row["norm_gt"] != row["norm_pred"]
        ]
        repaired_exact = sum(
            after[key][0].get("norm_pred") is not None
            and after[key][0].get("norm_gt") == after[key][0]["norm_pred"]
            for key in initially_nonexact
        )
        unchanged = sum(before[k][0]["pred"] == after[k][0]["pred"] for k in before)
        before_page = before_metrics[category]["page"][metric]["ALL"]
        after_page = after_metrics[category]["page"][metric]["ALL"]
        for value in (before_page, after_page):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid official page-average score")
        summary["modalities"][modality] = {
            "metric": metric,
            "matched_samples": len(before),
            "official_page_average_before": before_page,
            "official_page_average_after": after_page,
            "official_page_delta_pp": 100 * (after_page - before_page),
            "sample_average_before": mean(v for _, v in before.values()) if before else None,
            "sample_average_after": mean(v for _, v in after.values()) if after else None,
            "initial_good_proxy_count": len(good),
            "initial_bad_proxy_count": len(bad),
            "preserve_proxy": ratio(sum(after[k][1] >= good_threshold for k in good), len(good)),
            "regression_proxy": ratio(sum(after[k][1] < good_threshold for k in good), len(good)),
            "bad_fix_proxy": ratio(sum(after[k][1] >= good_threshold for k in bad), len(bad)),
            "initial_nonexact_count": len(initially_nonexact),
            "normalized_exact_fix": ratio(repaired_exact, len(initially_nonexact)),
            "unchanged_predictions": unchanged,
        }
        for role, path in [("before", before_path), ("after", after_path)]:
            summary["source_artifacts"].append(
                {"role": role, "file": path.name, "sha256": file_sha256(path)}
            )
    for role, directory, prefix in [
        ("before", before_dir, before_prefix),
        ("after", after_dir, after_prefix),
    ]:
        path = result_path(directory, "metric_result", prefix)
        summary["source_artifacts"].append(
            {"role": role, "file": path.name, "sha256": file_sha256(path)}
        )
    return summary
