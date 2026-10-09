#!/usr/bin/env python3
"""Refresh public summary facts only from complete, content-hashed evidence."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.dashboard_evidence import (
    verify_completion_binding,
    verify_four_arm_coverage,
    verify_prompt_stage,
)
from ocr_edr.research_dashboard import render_dashboard, validate_snapshot
from ocr_edr.sft import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=Path("docs/site"))
    parser.add_argument(
        "--results", type=Path, default=Path("experiments/runs/table-nf4-screen-recovery-20261009")
    )
    parser.add_argument(
        "--prompt-results",
        type=Path,
        default=Path("experiments/runs/table-prompt-ablation-20261009"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    verified = json.loads((args.results / "verified-completion.json").read_text())
    bound = {
        "evaluation": sha256(args.results / "evaluation/run.json"),
        "statistics_report": sha256(args.results / "statistics/report.json"),
    }
    for arm in ["all", "no_explicit_preservation"]:
        bound["training_" + arm] = sha256(args.results / "training" / arm / "run.json")
        bound["reload_" + arm] = sha256(args.results / "reload" / arm / "run.json")
    for arm in ["base", "all", "no_explicit_preservation"]:
        bound["inference_" + arm] = sha256(args.results / "inference" / arm / "run.json")
    verify_completion_binding(verified, bound)
    if verified["status"] != "completed" or not verified["scientific_outputs_verified"]:
        raise ValueError("Only independently verified full outputs can populate research results")
    folder = args.results / "evaluation"
    receipt = json.loads((folder / "run.json").read_text())
    if (
        receipt["status"] != "completed"
        or sha256(folder / "evaluation.json") != receipt["evaluation_sha256"]
        or sha256(folder / "predictions.jsonl") != receipt["prediction_sha256"]
    ):
        raise ValueError("Frozen main-result content changed")
    evaluation = json.loads((folder / "evaluation.json").read_text())
    verify_four_arm_coverage(evaluation)
    stats = json.loads((args.results / "statistics/report.json").read_text())
    if stats["evaluation_run_sha256"] != sha256(folder / "run.json") or stats[
        "evaluation_sha256"
    ] != sha256(folder / "evaluation.json"):
        raise ValueError("Statistics are not bound to the complete evaluation")
    summary = [r for r in evaluation["summary"] if r["variant"] == "all"]
    arms = [
        {
            "arm": r["arm"],
            "teds": r["case_mean_final_teds"],
            "document_teds": r["document_mean_final_teds"],
            "teds_structure": r["case_mean_final_teds_structure"],
            "repairs": r["teds_nonmatching_fixed"],
            "regressions": r["teds_matching_regressions"],
            "changed": r["changed"],
            "valid_actions": r["actions_with_valid_schema"],
            "stops": r["stop_action"],
            "caps": r["hit_length_cap"],
            "generation_seconds": r["generation_seconds"],
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "render_calls": r["render_attempts"],
        }
        for r in summary
    ]
    artifact = root / "experiments/artifacts"
    formula = json.loads(
        (artifact / "second-parser-development-20261007/initial_summary.json").read_text()
    )
    teacher = json.loads(
        (artifact / "interactive-teacher-pilot-20261007/audit/evaluation.json").read_text()
    )["summary"]
    native = json.loads(
        (
            artifact / "native-table-development-20261007/compat-evaluation/evaluation.json"
        ).read_text()
    )["summary"]
    training = []
    for arm in ["all", "no_explicit_preservation"]:
        record = json.loads((args.results / "training" / arm / "run.json").read_text())
        if record["status"] != "completed" or record["completed_steps"] != 381:
            raise ValueError("Do not display a partially trained model as completed")
        training.append(
            {
                key: record[key]
                for key in [
                    "arm",
                    "completed_steps",
                    "optimizer_examples",
                    "total_supervised_tokens",
                    "training_seconds",
                    "peak_memory_allocated_gib",
                    "peak_memory_reserved_gib",
                    "adapter_tensors_changed",
                ]
            }
        )
    prompt = {
        "status": "not_started",
        "stage": None,
        "runs": [],
        "scope": "post-hoc fixed-checkpoint prompt ablation; not confirmatory",
    }
    if (args.prompt_results / "pipeline-status.json").exists():
        running = json.loads((args.prompt_results / "pipeline-status.json").read_text())
        prompt.update(status=running["status"], stage=running["stage"])
        for p in sorted(args.prompt_results.glob("*/inference/*/run.json")):
            record = json.loads(p.read_text())
            prompt["runs"].append(
                {
                    "prompt": p.parents[2].name,
                    "condition": record["condition"],
                    "status": record["status"],
                    "completed_calls": record["completed_calls"],
                }
            )
        for p in sorted(args.prompt_results.glob("*/evaluation/evaluation.json")):
            run = json.loads(p.with_name("run.json").read_text())
            if run["status"] == "completed" and run["evaluation_sha256"] == sha256(p):
                verify_prompt_stage(running, p.parent.parent.name, sha256(p.with_name("run.json")))
                if run["prediction_sha256"] != sha256(p.with_name("predictions.jsonl")):
                    raise ValueError("Prompt prediction evidence changed")
                complete = json.loads(p.read_text())
                verify_four_arm_coverage(complete)
                for arm, digest in run["inference_run_sha256"].items():
                    inference_folder = p.parent.parent / "inference" / arm
                    if sha256(inference_folder / "run.json") != digest:
                        raise ValueError("Prompt model call receipt changed after evaluation")
                    inference = json.loads((inference_folder / "run.json").read_text())
                    if (
                        inference["status"] != "completed"
                        or inference["completed_calls"] != 103
                        or sha256(inference_folder / "calls.jsonl") != inference["calls_sha256"]
                    ):
                        raise ValueError("Prompt calls lack complete sealed coverage")
                rows = complete["summary"]
                prompt.setdefault("completed_evaluations", {})[p.parent.parent.name] = [
                    {
                        "arm": r["arm"],
                        "teds": r["case_mean_final_teds"],
                        "repairs": r["teds_nonmatching_fixed"],
                        "regressions": r["teds_matching_regressions"],
                        "changed": r["changed"],
                        "valid_actions": r["actions_with_valid_schema"],
                        "caps": r["hit_length_cap"],
                        "generation_seconds": r["generation_seconds"],
                    }
                    for r in rows
                    if r["variant"] == "all"
                ]
    data = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "repository": "https://github.com/lifelonglearnerAdam/ocr-edr-extend",
        "goal_status": "active_incomplete",
        "table_screen": {
            "cases": 103,
            "documents": 32,
            "initial_matching": 32,
            "initial_nonmatching": 71,
            "arms": arms,
            "training": training,
            "contrasts": [r for r in stats["contrasts"] if r["variant"] == "all"],
            "source_audit_documents": 23,
            "placeholder_copies": {"all": 24, "no_explicit_preservation": 7},
            "interrupted_steps": 62,
            "interrupted_exposures": 248,
        },
        "senior": json.loads((args.site / "senior-provenance.json").read_text()),
        "formula_second_parser": formula,
        "teacher": teacher,
        "native_tables": native,
        "prompt_ablation": prompt,
        "evidence": {
            "main_evaluation_sha256": sha256(folder / "evaluation.json"),
            "main_stats_sha256": sha256(args.results / "statistics/report.json"),
            "verified_completion_sha256": sha256(args.results / "verified-completion.json"),
            "senior_provenance_sha256": sha256(args.site / "senior-provenance.json"),
        },
    }
    validate_snapshot(data)
    args.site.mkdir(parents=True, exist_ok=True)
    current = args.site / "research-data.json"
    if current.exists():
        previous = json.loads(current.read_text())
        if {k: v for k, v in previous.items() if k != "updated_at"} == {
            k: v for k, v in data.items() if k != "updated_at"
        }:
            data["updated_at"] = previous["updated_at"]
    (args.site / "research-data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    )
    if (args.site / "template.html").exists():
        assets = {p.name: p.read_bytes() for p in (args.site / "assets").glob("*.png")}
        (args.site / "index.html").write_text(
            render_dashboard(data, (args.site / "template.html").read_text(), assets)
        )
    print(
        json.dumps(
            {
                "updated_at": data["updated_at"],
                "prompt_status": prompt["status"],
                "main_results": "verified_complete",
            }
        )
    )


if __name__ == "__main__":
    main()
