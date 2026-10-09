"""Full-cohort diagnosis errors and paired recorded model costs, not visual truth."""

import math


def diagnosis_summary(rows, *, expected_ids):
    identities = {r["sample_id"] for r in rows}
    if not rows or len(identities) != len(rows) or identities != set(expected_ids):
        raise ValueError("Complete unique diagnostic score coverage required")
    result = []
    for variant in ["all", *sorted({r["variant"] for r in rows})]:
        group = [r for r in rows if variant == "all" or r["variant"] == variant]
        valid = [r for r in group if r["target"]["verdict"] == "valid"]
        invalid = [r for r in group if r["target"]["verdict"] == "invalid"]
        if len(valid) + len(invalid) != len(group):
            raise ValueError("Unknown controlled diagnostic target")
        result.append(
            {
                "variant": variant,
                "cases": len(group),
                "initial_valid_n": len(valid),
                "initial_invalid_n": len(invalid),
                "valid_outputs": sum(r["prediction"] is not None for r in group),
                "verdict_correct": sum(
                    r["prediction"] is not None
                    and r["prediction"]["verdict"] == r["target"]["verdict"]
                    for r in group
                ),
                "strict_joint": sum(r["prediction"] == r["target"] for r in group),
                "valid_false_positive": sum(
                    (r["prediction"] or {}).get("verdict") == "invalid" for r in valid
                ),
                "invalid_missed": sum(
                    (r["prediction"] or {}).get("verdict") == "valid" for r in invalid
                ),
                "valid_unknown": sum(r["prediction"] is None for r in valid),
                "invalid_unknown": sum(r["prediction"] is None for r in invalid),
                "invalid_type_correct": sum(
                    (r["prediction"] or {}).get("verdict") == "invalid"
                    and r["prediction"]["error"] == r["target"]["error"]
                    for r in invalid
                ),
                "invalid_region_correct": sum(
                    (r["prediction"] or {}).get("verdict") == "invalid"
                    and r["prediction"]["region"] == r["target"]["region"]
                    for r in invalid
                ),
            }
        )
    return result


def paired_component_cost(refinement, diagnoses, *, use_diagnosis):
    by_ref = {r["sample_id"]: r for r in refinement}
    by_diag = {r["sample_id"]: r for r in diagnoses}
    if not refinement or len(by_ref) != len(refinement):
        raise ValueError("Complete unique refinement costs required")
    if use_diagnosis and (len(by_diag) != len(diagnoses) or by_ref.keys() != by_diag.keys()):
        raise ValueError("Diagnostic/refinement cost pairing differs")
    included = [*refinement, *(diagnoses if use_diagnosis else [])]
    for row in included:
        for key in ["input_tokens", "output_tokens"]:
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError("Invalid model token accounting")
        seconds = row["generation_seconds"]
        if type(seconds) not in {int, float} or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid finite generation time")
    if any(type(r["model_calls"]) is not int or r["model_calls"] != 1 for r in refinement):
        raise ValueError("Each refinement condition must retain one call per input")
    return {
        "cases": len(refinement),
        "diagnosis_calls": len(diagnoses) if use_diagnosis else 0,
        "refinement_calls": len(refinement),
        "model_calls": len(included),
        "input_tokens": sum(r["input_tokens"] for r in included),
        "output_tokens": sum(r["output_tokens"] for r in included),
        "generation_seconds": sum(r["generation_seconds"] for r in included),
        "render_attempts": sum(r.get("render_attempts", 0) for r in refinement),
        "render_seconds": sum(r.get("render_seconds", 0) for r in refinement),
        "cost_scope": "paired recorded generation and offline render checks; excludes loading, preprocessing and initial parsing; shared diagnoses reused for ablations",
        "oracle_annotation_cost_included": False,
    }
