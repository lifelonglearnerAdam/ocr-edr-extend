#!/usr/bin/env python3
"""Refresh public summary facts only from complete, content-hashed evidence."""

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.dashboard_evidence import (
    live_optimizer_progress,
    verify_completion_binding,
    verify_four_arm_coverage,
    verify_native_coverage,
    verify_prompt_stage,
    verify_source_probe_coverage,
    verify_walkthrough_cases,
)
from ocr_edr.native_table_repair import native_repair_inputs, verify_native_calls
from ocr_edr.native_tables import load_native_table_sources
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
    parser.add_argument(
        "--native-results", type=Path, default=Path("experiments/runs/native-table-sft-20261009")
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
            "page_template_sha256": sha256(args.site / "template.html"),
            "beginner_explanation_sha256": sha256(args.site / "beginner.fragment.html"),
            "briefing_explanation_sha256": sha256(args.site / "briefing.fragment.html"),
        },
    }
    if (args.site / "walkthrough-data.json").exists():
        cases = json.loads((args.site / "walkthrough-data.json").read_text())
        descriptive = args.prompt_results / "descriptive_schema/evaluation/evaluation.json"
        if cases["evaluation_original_sha256"] != sha256(descriptive):
            raise ValueError("Teaching cases must remain bound to the frozen actual experiment")
        verify_walkthrough_cases(
            cases["cases"],
            json.loads(descriptive.read_text()),
            [
                json.loads(line)
                for line in descriptive.with_name("predictions.jsonl").read_text().splitlines()
            ],
        )
        data["walkthroughs"] = cases["cases"]
    activity = {"status": "not_started", "stage": "准备输入", "runs": []}
    if (args.native_results / "pipeline-status.json").exists():
        record = json.loads((args.native_results / "pipeline-status.json").read_text())
        activity.update(status=record["status"], stage=record["stage"])
        for folder in sorted((args.native_results / "inference").glob("*")):
            if not (folder / "run.json").exists():
                continue
            r = json.loads((folder / "run.json").read_text())
            activity["runs"].append(
                {
                    "condition": r["condition"],
                    "status": r["status"],
                    "completed_calls": r["completed_calls"],
                    "target_calls": 32,
                }
            )
        evaluated = args.native_results / "evaluation"
        if (evaluated / "evaluation.json").exists() and record["status"] == "completed":
            r = json.loads((evaluated / "run.json").read_text())
            sealed = next(s for s in record["completed_stages"] if s["stage"] == "evaluation")
            if (
                sha256(evaluated / "run.json") != sealed["receipt_sha256"]
                or sha256(evaluated / "evaluation.json") != r["evaluation_sha256"]
                or sha256(evaluated / "predictions.jsonl") != r["predictions_sha256"]
            ):
                raise ValueError("Native diagnostic result drifted after completion")
            complete = json.loads((evaluated / "evaluation.json").read_text())
            verify_native_coverage(complete)
            native_examples = json.loads((args.site / "native-walkthrough-data.json").read_text())
            if native_examples["evaluation_original_sha256"] != sha256(
                evaluated / "evaluation.json"
            ):
                raise ValueError("Native walkthrough differs from its frozen evaluation")
            verify_walkthrough_cases(
                native_examples["cases"],
                complete,
                [
                    json.loads(line)
                    for line in (evaluated / "predictions.jsonl").read_text().splitlines()
                ],
            )
            data["native_walkthroughs"] = native_examples["cases"]
            dataset = root / "data/processed/pubtabnet-four-roles-20261007"
            native_inputs = args.native_results / "frozen-native-inputs/predictions.jsonl"
            if sha256(native_inputs) != r["native_prediction_sha256"]:
                raise ValueError("Frozen native source input changed")
            inputs = native_repair_inputs(
                [json.loads(line) for line in native_inputs.read_text().splitlines()],
                load_native_table_sources(dataset, role="model_dev"),
                role="model_dev",
            )
            for arm in ["base", "all", "no_explicit_preservation"]:
                inference = args.native_results / "inference" / arm
                receipt = json.loads((inference / "run.json").read_text())
                sealed = [s for s in record["completed_stages"] if s["stage"] == "inference_" + arm]
                if (
                    len(sealed) != 1
                    or sha256(inference / "run.json") != sealed[0]["receipt_sha256"]
                    or receipt["status"] != "completed"
                    or receipt["completed_calls"] != 32
                    or sha256(inference / "calls.jsonl") != receipt["calls_sha256"]
                ):
                    raise ValueError("Native model calls differ from the complete pipeline seal")
                verify_native_calls(
                    [
                        json.loads(line)
                        for line in (inference / "calls.jsonl").read_text().splitlines()
                    ],
                    inputs,
                    arm,
                )
                if arm != "base":
                    trained = json.loads((args.results / "training" / arm / "run.json").read_text())
                    if receipt["adapter_sha256"] != trained["checkpoint_sha256"]:
                        raise ValueError(
                            "Native adapter identity differs from the frozen trained checkpoint"
                        )
            summary = complete["summary"]
            original = next(
                s for s in summary if s["variant"] == "all" and s["arm"] == "unchanged_0"
            )
            activity.update(
                initial_matching=original["teds_initial_matching_n"],
                initial_nonmatching=original["teds_initial_nonmatching_n"],
                evaluation_sha256=sha256(evaluated / "evaluation.json"),
            )
            activity["results"] = [
                {
                    "arm": s["arm"],
                    "teds": s["case_mean_final_teds"],
                    "repairs": s["teds_nonmatching_fixed"],
                    "regressions": s["teds_matching_regressions"],
                    "changed": s["changed"],
                    "cases": s["cases"],
                    "normalized_changed": s["normalized_changed"],
                    "degraded": s["teds_degraded"],
                    "improved": s["teds_improved"],
                    "adapter_failures": s["adapter_failures"],
                    "caps": s["hit_length_cap"],
                }
                for s in summary
                if s["variant"] == "all"
            ]
    data["current_activity"] = activity
    diagnosis_root = root / "experiments/runs/table-diagnosis-20261009"
    training_unit = "ocr-edr-table-diagnosis-training-20261009.service"
    recovery = None
    location = diagnosis_root / "recovery-location.json"
    if location.exists():
        selected = json.loads(location.read_text())
        diagnosis_root = (root / selected["active_study"]).resolve()
        diagnosis_root.relative_to(root / "experiments/runs")
        recovery = json.loads((diagnosis_root / "recovery-receipt.json").read_text())
        if (
            sha256(diagnosis_root / "recovery-receipt.json") != selected["recovery_receipt_sha256"]
            or recovery["status"] != "original_interrupted"
            or not recovery["original_receipt_preserved"]
        ):
            raise ValueError("Diagnosis recovery location lacks explicit preserved evidence")
        training_unit = selected["training_unit"]
    if (diagnosis_root / "training/run.json").exists():
        training_record = json.loads((diagnosis_root / "training/run.json").read_text())
        if (
            training_record["study"] != "table_diagnosis_nf4_sft_20261009"
            or training_record["config"]["steps"] != 381
            or training_record["exposures"] != 1524
            or training_record["dev_optimizer_examples"] != 0
            or training_record["calibration_locked_optimizer_examples"] != 0
        ):
            raise ValueError("Diagnosis progress belongs to a different experiment")
        log = diagnosis_root / "training/training.jsonl"
        service = "unknown"
        if shutil.which("systemctl"):
            unit = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "is-active",
                    training_unit,
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            service = unit.stdout.strip() or "unknown"
        progress = live_optimizer_progress(
            training_record,
            log.read_text().splitlines() if log.exists() else [],
            service=service,
            idle_seconds=time.time() - log.stat().st_mtime if log.exists() else 0,
        )
        pipeline = diagnosis_root / "pipeline-status.json"
        current_stage = json.loads(pipeline.read_text()) if pipeline.exists() else {}
        data["diagnosis_component_progress"] = {
            "training_status": training_record["status"],
            **progress,
            "target_steps": 381,
            "training_service": service,
            "pipeline_stage": current_stage.get("stage", "training"),
            "pipeline_status": current_stage.get("status", "not_started"),
            "train_documents": 127,
            "train_records": 406,
            "exposures": 1524,
            "quality_results": "pending_review",
            "scope": "separate adapter; same base and weak controlled train sources",
            "retry_after_interruption": recovery is not None,
            "discarded_original_steps": recovery["discarded_steps"] if recovery else 0,
            "discarded_original_exposures": (
                recovery["discarded_optimizer_exposures"] if recovery else 0
            ),
        }
    source_probe = root / "experiments/runs/table-source-evidence-complete-20261009"
    if (source_probe / "completion.json").exists():
        completion = json.loads((source_probe / "completion.json").read_text())
        comparison = json.loads((source_probe / "comparison.json").read_text())
        if completion["status"] != "completed" or completion["model_calls"] != 64:
            raise ValueError("Source image probe is not complete")
        if sha256(source_probe / "comparison.json") != completion["comparison_sha256"]:
            raise ValueError("Source probe comparison differs from its completion seal")
        old = root / "experiments/runs/table-source-evidence-20261009"
        new = root / "experiments/runs/table-source-evidence-second-arm-20261009"
        paths = {
            "failed_run": old / "run.json",
            "saved_all_calls": old / "all/calls.jsonl",
            "recovery": old / "recovery.json",
            "recovered_comparison": old / "replayed-comparison.json",
            "second_run": new / "run.json",
            "second_calls": new / "no_explicit_preservation/calls.jsonl",
            "second_comparison": new / "comparison.json",
        }
        if any(sha256(p) != completion["evidence_sha256"][name] for name, p in paths.items()):
            raise ValueError("Source probe generating or recovery evidence changed")
        verify_source_probe_coverage(comparison)
        if comparison["summary"] != completion["summary"]:
            raise ValueError("Source probe displayed summary differs from paired evidence")
        generated = json.loads((new / "run.json").read_text())
        if generated["native_audit_sha256"] != sha256(args.native_results / "audit.json"):
            raise ValueError("Original source action audit changed after the probe")
        original_actions = json.loads((args.native_results / "audit.json").read_text())[
            "action_counts"
        ]
        data["source_evidence_probe"] = {
            "status": "completed",
            "cases": 32,
            "model_calls": 64,
            "summary": comparison["summary"],
            "comparison_sha256": completion["comparison_sha256"],
            "first_arm_reused_calls": 32,
            "scope": "post-hoc mechanism probe, no new quality/accuracy claim",
            "real_actions": {
                arm: original_actions[arm] for arm in ["all", "no_explicit_preservation"]
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
            render_dashboard(
                data,
                (args.site / "template.html").read_text(),
                assets,
                (
                    (args.site / "beginner.fragment.html").read_text()
                    if (args.site / "beginner.fragment.html").exists()
                    else ""
                )
                + (args.site / "briefing.fragment.html").read_text(),
            )
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
