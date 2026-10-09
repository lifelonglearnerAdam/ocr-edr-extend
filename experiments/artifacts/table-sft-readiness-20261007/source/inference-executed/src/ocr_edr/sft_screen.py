"""Reference-free adaptation of fixed calls before independent offline scoring."""

from __future__ import annotations

import subprocess

from .formula_pilot import make_prompt
from .loop import Observation
from .sft import proposal_contract


def adapt_screen_calls(
    inputs: list[dict], calls_by_condition: dict[str, list[dict]], *, renderer, max_new_tokens: int
) -> list[dict]:
    cases = {r["sample_id"]: r for r in inputs}
    if not cases or len(cases) != len(inputs) or not calls_by_condition:
        raise ValueError("Empty/duplicate fixed input coverage")
    for condition, calls in calls_by_condition.items():
        if condition not in {"base", "all", "no_explicit_preservation"}:
            raise ValueError("Unknown inference condition")
        indexed = {r["sample_id"]: r for r in calls}
        if len(indexed) != len(calls) or indexed.keys() != cases.keys():
            raise ValueError("Missing/duplicate inference coverage")
        for sample, call in indexed.items():
            case = cases[sample]
            if {"reference", "target", "variant", "score"} & call.keys():
                raise ValueError("Offline label fields cannot be included in inference calls")
            if call["condition"] != condition or any(
                call[field] != case[field] for field in ["family_id", "prediction", "source_sha256"]
            ):
                raise ValueError("Inference source/candidate identity drift")
            if call["ordered_image_sha256"] != [case["source_sha256"]] or call[
                "prompt"
            ] != make_prompt(case["prediction"], False):
                raise ValueError("Inference prompt/image identity mismatch")
            if (
                type(call["output_tokens"]) is not int
                or call["output_tokens"] < 0
                or type(call["hit_token_cap"]) is not bool
            ):
                raise ValueError("Invalid inference cost/cap receipt")
            if call["hit_token_cap"] != (call["output_tokens"] >= max_new_tokens):
                raise ValueError("Token cap flag mismatch")
    results = [
        {
            "sample_id": case["sample_id"],
            "family_id": case["family_id"],
            "arm": "unchanged_0",
            "initial_prediction": case["prediction"],
            "final_prediction": case["prediction"],
            "trace": [],
        }
        for case in inputs
    ]
    for condition, calls in calls_by_condition.items():
        for call in calls:
            case = cases[call["sample_id"]]
            candidate, decision = proposal_contract(
                call["raw_output"], hit_cap=call["hit_token_cap"]
            )
            final = case["prediction"]
            error = None
            if candidate is not None:
                try:
                    renderer.render(
                        Observation(case["sample_id"], "formula", case["source_image"], candidate)
                    )
                    final = candidate
                    decision = "accepted_syntax_only"
                except (ValueError, FileNotFoundError, subprocess.SubprocessError) as failure:
                    error = type(failure).__name__ + ": " + str(failure)[:400]
                    decision = "render_failure_rollback"
            results.append(
                {
                    "sample_id": case["sample_id"],
                    "family_id": case["family_id"],
                    "arm": condition,
                    "initial_prediction": case["prediction"],
                    "final_prediction": final,
                    "trace": [
                        {
                            **call,
                            "candidate": candidate,
                            "decision": decision,
                            "render_error": error,
                        }
                    ],
                }
            )
    return results
