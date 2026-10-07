#!/usr/bin/env python3
"""Verify a sealed train-only teacher cohort, then score and export paired views."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import _bootstrap  # noqa: F401

from ocr_edr.loop import Observation, digest
from ocr_edr.official_tables import (
    load_official_table_normalizer,
    load_official_teds,
    verify_official_table_sources,
)
from ocr_edr.sft import sha256
from ocr_edr.table_pilot import HTMLTableRenderer, parse_table, table_cell_map
from ocr_edr.table_sft_screen import _unique_object
from ocr_edr.teacher_audit import (
    paired_teacher_views,
    replay_teacher_episode,
    validate_teacher_mapping,
    verify_frozen_files,
)


def read(path):
    return json.loads(path.read_text(), object_pairs_hook=_unique_object)


def lines(path):
    return [
        json.loads(line, object_pairs_hook=_unique_object) for line in path.read_text().splitlines()
    ]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def require_hash(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Evidence hash mismatch: {path.name}")


def verify_packet(packet_root, seal_path, output):
    """No mapping or reference JSON is parsed by this mechanical stage."""
    seal = read(seal_path)
    verify_frozen_files(packet_root, seal["packet_files_sha256"])
    frozen_code = seal_path.parent / "sealed-code"
    verify_frozen_files(frozen_code, seal["sealed_code_sha256"])
    packet = read(packet_root / "packet.json")
    if (
        seal["sealed_code_sha256"]["scripts/prepare_teacher_table_pilot.py"]
        != packet["driver_sha256"]
    ):
        raise ValueError("Frozen preparation snapshot differs from the packet driver")
    require_hash(packet_root / "teacher-instructions.txt", packet["teacher_prompt_sha256"])
    helper = "src/ocr_edr/teacher_tables.py"
    executor = "scripts/execute_teacher_table_step.py"
    project = Path(__file__).resolve().parents[1]
    for relative in [helper, executor, "src/ocr_edr/table_pilot.py", "src/ocr_edr/loop.py"]:
        require_hash(project / relative, seal["sealed_code_sha256"][relative])
    ids = [row["case_id"] for row in packet["tasks"]]
    if (
        len(ids) != 12
        or len(set(ids)) != 12
        or set(ids) != {p.name for p in (packet_root / "cases").iterdir()}
        or any(not re.fullmatch(r"t_[a-f0-9]{12}", case) for case in ids)
        or seal["terminal_episodes"] != 12
        or seal["status"] != "all_episodes_terminal_before_offline_reference_check"
    ):
        raise ValueError("Complete frozen twelve-case cohort is required")
    renderer = HTMLTableRenderer(output / "fresh-renders")
    audits, episodes, decision_files = [], [], set()
    for expected in packet["tasks"]:
        case = expected["case_id"]
        work = packet_root / "cases" / case
        require_hash(work / "task.json", expected["task_sha256"])
        require_hash(work / "source.png", expected["source_sha256"])
        task, state = read(work / "task.json"), read(work / "state.json")
        if task["case_id"] != case or task["source_sha256"] != expected["source_sha256"]:
            raise ValueError("Teacher task/source identity mismatch")
        initial_render = None
        if task["current_render_error"] is None:
            require_hash(work / "current.png", task["current_render_sha256"])
            rendered = renderer.render(Observation(case, "table", "", task["current_html"]))
            require_hash(Path(rendered.path), task["current_render_sha256"])
            if task["current_cell_addresses"] != table_cell_map(task["current_html"]):
                raise ValueError("Frozen current-DOM addresses differ")
            initial_render = f"cases/{case}/current.png"
        elif task["current_render_sha256"] is not None:
            raise ValueError("Failed initial render carries successful image evidence")
        decisions, events, step_views = [], [], []
        current_html = task["current_html"]
        current_render, current_render_hash = initial_render, task["current_render_sha256"]
        previous_time = datetime.fromisoformat(packet["created_at"])
        event_files = set()
        for index, saved in enumerate(state["steps"]):
            filename = f"step-{index:02d}.json"
            if saved["file"] != filename:
                raise ValueError("Teacher event order differs")
            event_path = work / filename
            event_files.add(filename)
            require_hash(event_path, saved["sha256"])
            decision_path = packet_root / "decisions" / f"{case}-{index:02d}.json"
            decision_files.add(decision_path.name)
            event, decision = read(event_path), read(decision_path)
            require_hash(decision_path, event["teacher_decision_sha256"])
            if (
                event["source_sha256"] != task["source_sha256"]
                or event["executor_sha256"] != seal["sealed_code_sha256"][executor]
                or event["helper_sha256"] != seal["sealed_code_sha256"][helper]
                or event["teacher_model_label"] != packet["teacher_model_label"]
                or event["teacher_effort_label"] != packet["teacher_effort_label"]
            ):
                raise ValueError("Teacher execution/source provenance drift")
            recorded = datetime.fromisoformat(event["recorded_at"])
            if not previous_time <= recorded <= datetime.fromisoformat(seal["sealed_at"]):
                raise ValueError("Teacher event time is outside the sealed generation interval")
            previous_time = recorded
            observation = {
                "html": current_html,
                "html_sha256": digest(current_html or ""),
                "render_image": current_render,
                "render_image_sha256": current_render_hash,
            }
            tool = {
                k: event[k]
                for k in ["applied", "changed", "final_html", "action_error", "render_error"]
            }
            if event.get("render_path"):
                render_path = Path(event["render_path"]).resolve()
                current_render = render_path.relative_to(packet_root).as_posix()
                current_render_hash = event["render_image_sha256"]
                require_hash(render_path, current_render_hash)
                render_receipt = read(render_path.with_name("render.json"))
                if (
                    render_receipt["prediction_sha256"] != event["render_prediction_sha256"]
                    or render_receipt["png_sha256"] != current_render_hash
                    or render_receipt["backend"] != event["render_backend"]
                ):
                    raise ValueError("Frozen render receipt binding differs")
            tool.update(render_image=current_render, render_image_sha256=current_render_hash)
            step_views.append({"observation": observation, "decision": decision, "tool": tool})
            decisions.append(decision)
            events.append(event)
            current_html = event["final_html"]
        if event_files != {p.name for p in work.glob("step-*.json")}:
            raise ValueError("Unlogged teacher event")
        audit = replay_teacher_episode(
            task,
            decisions,
            events,
            state,
            max_edit_attempts=packet["max_edit_attempts_per_case"],
            renderer=renderer.render,
        )
        audit["initial_render_reproduced"] = initial_render is not None
        audits.append(audit)
        episodes.append(
            {
                "case_id": case,
                "source_image": f"cases/{case}/source.png",
                "source_sha256": task["source_sha256"],
                "initial_html": task["current_html"],
                "final_html": state["html"],
                "steps": step_views,
            }
        )
    if decision_files != {p.name for p in (packet_root / "decisions").iterdir()}:
        raise ValueError("Unlogged teacher decision")
    for field in ["edit_attempts"]:
        if sum(row[field] for row in audits) != seal[field]:
            raise ValueError("Cohort seal summary differs")
    write(output / "mechanical.json", {"completed_at": now(), "cases": audits})
    return packet, seal, episodes, audits


def load_train_references(args, packet, episodes):
    """Called only after every frozen episode has passed mechanical checks."""
    dataset = read(args.dataset / "dataset.json")
    require_hash(args.dataset / "dataset.json", packet["dataset_sha256"])
    admission_path = args.dataset / "admitted-supervision/admission.json"
    require_hash(admission_path, packet["admission_sha256"])
    admission = read(admission_path)
    if admission["status"] != "admitted_as_published_weak_supervision":
        raise ValueError("Training admission is unavailable")
    require_hash(args.native_run / "run.json", packet["native_run_sha256"])
    native_run = read(args.native_run / "run.json")
    require_hash(args.native_run / "predictions.jsonl", native_run["predictions_sha256"])
    if (
        native_run["status"] != "completed"
        or native_run["source_role"] != "train"
        or native_run["reference_access"] != "none"
    ):
        raise ValueError("Native train candidate provenance differs")
    for name in ["selected_sources.jsonl", "train-inputs.jsonl", "train-references.jsonl"]:
        require_hash(args.dataset / name, dataset["file_sha256"][name])
    mapping_path = args.packet / "private-mapping.json"
    require_hash(mapping_path, packet["private_mapping_sha256"])
    mapping = read(mapping_path)
    sources = lines(args.dataset / "selected_sources.jsonl")
    validate_teacher_mapping(mapping, sources, set(admission["excluded_families"]))
    by_case = {r["case_id"]: r for r in mapping}
    inventory = {r["family_id"]: r for r in sources}
    if set(by_case) != {r["case_id"] for r in episodes}:
        raise ValueError("Teacher mapping case coverage differs")
    families = [f"p{i:04d}" for i in range(9, 21)]

    def rank(label):
        return hashlib.sha256((packet["selection_seed"] + ":" + label).encode()).hexdigest()

    native_families = set(sorted(families, key=lambda f: rank("route:" + f))[:6])
    native = {r["family_id"]: r for r in lines(args.native_run / "predictions.jsonl")}
    controlled = lines(args.dataset / "train-inputs.jsonl")
    if {r["family_id"] for r in mapping} != set(families):
        raise ValueError("Teacher cohort selection differs")
    for episode in episodes:
        row = by_case[episode["case_id"]]
        family = row["family_id"]
        if family in native_families:
            candidate, origin, sample = native[family]["prediction"], "native", family + "-native"
        else:
            selected = min(
                (r for r in controlled if r["family_id"] == family),
                key=lambda r: rank("variant:" + r["sample_id"]),
            )
            candidate, origin, sample = selected["prediction"], "controlled", selected["sample_id"]
        if (
            episode["initial_html"] != candidate
            or episode["source_sha256"] != row["source_sha256"]
            or episode["case_id"] != "t_" + rank("opaque:" + family)[:12]
            or row["origin"] != origin
            or row["original_sample_id"] != sample
        ):
            raise ValueError("Teacher source/candidate selection is not reproducible")
    references = {}
    for row in lines(args.dataset / "train-references.jsonl"):
        if row["role"] != "train":
            raise ValueError("Held-out reference reached teacher verification")
        identity = (row["document_id"], row["reference"])
        if references.setdefault(row["family_id"], identity) != identity:
            raise ValueError("Inconsistent weak reference within a source family")
    for row in mapping:
        family = row["family_id"]
        if references[family][0] != inventory[family]["document_id"]:
            raise ValueError("Reference document identity differs")
    return by_case, references


def raw_shape(markup):
    if markup is None:
        return {"rows": 0, "cells": 0, "literal_format_tags": 0}
    try:
        table, _ = parse_table(markup)
    except ValueError:
        return {"rows": 0, "cells": 0, "literal_format_tags": 0}
    return {
        "rows": len(table.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr")),
        "cells": len(table.xpath(".//td|.//th")),
        "literal_format_tags": len(
            re.findall(r"</?(?:b|i|sup|sub|strong|em)\b[^>]*>", "".join(table.itertext()))
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["packet", "seal", "dataset", "native-run", "official-root", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.packet, args.output = args.packet.resolve(), args.output.resolve()
    if args.output == args.packet or args.packet in args.output.parents:
        raise ValueError("Audit output must be outside the sealed teacher packet")
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "initializing",
        "started_at": now(),
        "cohort_seal_sha256": sha256(args.seal),
        "independent_visual_judge": False,
        "label_scope": "published train weak-reference agreement; not visual gold",
        "admission_policy": "mechanically_valid + valid_final_html + low_terminal_uncertainty + both_official_scores_full_match_1e-12",
        "paired_views_scope": "same episodes, source, initial state and teacher final answer; not separately elicited direct-answer teacher",
        "token_budget_matched": False,
        "student_training_executed": False,
        "calibration_locked_reference_access": False,
        "code_sha256": {
            p.relative_to(Path(__file__).resolve().parents[1]).as_posix(): sha256(p)
            for p in [
                Path(__file__).resolve(),
                Path(__file__).resolve().parents[1] / "src/ocr_edr/teacher_audit.py",
            ]
        },
    }
    write(args.output / "run.json", receipt)
    try:
        packet, seal, episodes, mechanical = verify_packet(args.packet, args.seal, args.output)
        receipt["mechanical_completed_at"] = now()
        print(
            "All 12 frozen episodes mechanically verified; opening train-only mappings/references.",
            flush=True,
        )
        receipt["reference_stage_started_at"] = now()
        mapping, references = load_train_references(args, packet, episodes)
        receipt["reference_sha256"] = sha256(args.dataset / "train-references.jsonl")
        official = args.official_root.resolve()
        receipt["official_source"] = verify_official_table_sources(
            official, "f133a71e9e91c3621c7ce8994200a7b394a06eb3"
        )
        normalize = load_official_table_normalizer(official)
        teds = load_official_teds(official)
        metrics = {"teds": teds(), "teds_structure": teds(structure_only=True)}
        rows = []
        for episode, audit in zip(episodes, mechanical):
            source = mapping[episode["case_id"]]
            document, reference = references[source["family_id"]]
            ref_normalized = normalize(reference)
            row = {
                **audit,
                "family_id": source["family_id"],
                "document_id": document,
                "origin": source["origin"],
                "source_sha256": episode["source_sha256"],
                "initial_html_sha256": digest(episode["initial_html"] or ""),
            }
            for stage, key in [("before", "initial_html"), ("after", "final_html")]:
                shape = raw_shape(episode[key])
                row.update({stage + "_" + k: v for k, v in shape.items()})
                for name, metric in metrics.items():
                    if metric.evaluate(ref_normalized, ref_normalized) != 1:
                        raise ValueError("Published reference is not ready for official metric")
                    value = (
                        metric.evaluate(normalize(episode[key]), ref_normalized)
                        if shape["cells"]
                        else 0.0
                    )
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError("Invalid official teacher metric")
                    row[stage + "_" + name] = value
            row["provisionally_admitted"] = (
                audit["mechanically_valid"]
                and audit["final_html_valid"]
                and audit["terminal_assessment"] == "teacher_self_check"
                and all(row["after_" + name] >= 1 - 1e-12 for name in metrics)
            )
            rows.append(row)
        final, trajectory = paired_teacher_views(episodes, rows)
        outputs = {
            "episodes.jsonl": episodes,
            "final-only.jsonl": final,
            "trajectory.jsonl": trajectory,
            "review-required.jsonl": [r for r in rows if not r["provisionally_admitted"]],
        }
        for name, records in outputs.items():
            (args.output / name).write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
            )
        summary = {
            "source_documents": len(rows),
            "mechanically_valid": sum(r["mechanically_valid"] for r in rows),
            "changed_episodes": sum(r["changed_from_initial"] for r in rows),
            "teacher_self_check": sum(
                r["terminal_assessment"] == "teacher_self_check" for r in rows
            ),
            "teacher_abstention": sum(
                r["terminal_assessment"] == "teacher_abstention" for r in rows
            ),
            "provisionally_admitted": len(final),
            "review_required": len(rows) - len(final),
            **{
                key: sum(r[key] for r in rows)
                for key in [
                    "edit_attempts",
                    "global_edits",
                    "local_edits",
                    "action_failures",
                    "render_failures",
                ]
            },
        }
        for stage in ["before", "after"]:
            for name in metrics:
                key = stage + "_" + name
                summary[key + "_mean"] = mean(r[key] for r in rows)
                summary[key + "_full_match"] = sum(r[key] >= 1 - 1e-12 for r in rows)
        write(args.output / "evaluation.json", {"summary": summary, "cases": rows})
        with (args.output / "per_source.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        verify_frozen_files(args.packet, seal["packet_files_sha256"])
        require_hash(args.seal, receipt["cohort_seal_sha256"])
        receipt.update(
            status="completed",
            summary=summary,
            source_paths_relative_to="sealed packet root",
            outputs_sha256={
                name: sha256(args.output / name)
                for name in [*outputs, "evaluation.json", "per_source.csv", "mechanical.json"]
            },
            versions={
                name: importlib.metadata.version(name)
                for name in [
                    "lxml",
                    "weasyprint",
                    "Pillow",
                    "apted",
                    "Levenshtein",
                    "beautifulsoup4",
                ]
            },
        )
        print(json.dumps(summary, indent=2))
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error)[:700])
        raise
    finally:
        receipt["finished_at"] = now()
        write(args.output / "run.json", receipt)


if __name__ == "__main__":
    main()
