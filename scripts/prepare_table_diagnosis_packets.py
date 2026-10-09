#!/usr/bin/env python3
"""Freeze diagnosis projections only after all135 model diagnoses are frozen."""

import argparse
import hashlib
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from run_table_diagnosis import load_cohort

from ocr_edr.native_table_repair import verify_native_calls
from ocr_edr.sft import sha256
from ocr_edr.table_diagnosis import (
    bind_diagnosis,
    diagnosis_from_action,
    diagnosis_prompt,
    displace_region,
    parse_diagnosis,
)
from ocr_edr.table_training import load_table_supervision


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["dataset", "admission", "diagnoses", "native-run", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {}
    diagnosed = {}
    digests = {}
    for cohort, n in [("controlled", 103), ("native", 32)]:
        folder = args.diagnoses / cohort
        receipt = read(folder / "run.json")
        if (
            receipt["status"] != "completed"
            or receipt["completed_calls"] != n
            or receipt["reference_access"] != "none"
            or receipt["target_access"] != "none"
            or sha256(folder / "calls.jsonl") != receipt["calls_sha256"]
        ):
            raise ValueError("All135 reference-free diagnoses must be sealed first")
        inputs[cohort] = load_cohort(args.dataset, cohort, args.native_run)
        calls = rows(folder / "calls.jsonl")
        verify_native_calls(calls, inputs[cohort], "all")
        digests[cohort] = {
            "receipt": sha256(folder / "run.json"),
            "calls": sha256(folder / "calls.jsonl"),
        }
        result = []
        for item, call in zip(inputs[cohort], calls):
            if call["prompt"] != diagnosis_prompt(item["prediction"]):
                raise ValueError("Diagnosis prompt changed from current candidate")
            value = None
            error = None
            try:
                if call["output_tokens"] >= 192:
                    raise ValueError("token_cap")
                value = parse_diagnosis(item["prediction"], call["raw_output"])
            except ValueError as issue:
                error = str(issue)[:200]
            result.append(
                {
                    "sample_id": item["sample_id"],
                    "bound": bind_diagnosis(
                        item,
                        value,
                        producer="learned",
                        output_sha256=hashlib.sha256(call["raw_output"].encode()).hexdigest(),
                    ),
                    "parse_error": error,
                }
            )
        diagnosed[cohort] = result
    # First access to controlled gold/targets is after all diagnosis calls are frozen.
    gold = load_table_supervision(args.admission, args.dataset, role="model_dev")
    expected = {r["sample_id"]: r for r in gold}
    if set(expected) != {r["sample_id"] for r in inputs["controlled"]}:
        raise ValueError("Complete controlled diagnosis label coverage required")
    study = {
        "status": "completed",
        "diagnosis_evidence_sha256": digests,
        "oracle_label_access": "offline_controlled_location_projection_only",
        "replacement_text_or_restored_span_in_hints": False,
        "packets": {},
    }
    for cohort in ["controlled", "native"]:
        for condition in (
            ["learned", "displaced_region", "oracle_controlled"]
            if cohort == "controlled"
            else ["learned"]
        ):
            packet = []
            for item, learned in zip(inputs[cohort], diagnosed[cohort]):
                value = learned["bound"]["diagnosis"]
                producer = condition
                if condition == "displaced_region" and value is not None:
                    value = displace_region(
                        item["prediction"], value, identity=item["sample_id"], seed=20261007
                    )
                if condition == "oracle_controlled":
                    row = expected[item["sample_id"]]
                    if (
                        row["candidate"] != item["prediction"]
                        or row["source_sha256"] != item["source_sha256"]
                    ):
                        raise ValueError("Oracle source/candidate identity differs")
                    value = diagnosis_from_action(item["prediction"], json.loads(row["target"]))
                output_digest = (
                    hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
                    if condition == "oracle_controlled"
                    else learned["bound"]["output_sha256"]
                )
                bound = bind_diagnosis(item, value, producer=producer, output_sha256=output_digest)
                packet.append(
                    {
                        **item,
                        "diagnosis": bound,
                        "diagnosis_parse_error": (
                            learned["parse_error"] if condition != "oracle_controlled" else None
                        ),
                    }
                )
            filename = cohort + "-" + condition + ".jsonl"
            (out / filename).write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in packet)
            )
            study["packets"][filename] = {
                "sha256": sha256(out / filename),
                "cases": len(packet),
                "cohort": cohort,
                "condition": condition,
            }
    diagnosis_score = []
    for item, learned in zip(inputs["controlled"], diagnosed["controlled"]):
        target = diagnosis_from_action(
            item["prediction"], json.loads(expected[item["sample_id"]]["target"])
        )
        value = learned["bound"]["diagnosis"]
        diagnosis_score.append(
            {
                "sample_id": item["sample_id"],
                "family_id": item["family_id"],
                "document_id": expected[item["sample_id"]]["document_id"],
                "variant": expected[item["sample_id"]]["variant"],
                "prediction": value,
                "target": target,
                "valid_output": value is not None,
                "verdict_correct": value is not None and value["verdict"] == target["verdict"],
                "strict_joint": value == target,
                "parse_error": learned["parse_error"],
            }
        )
    (out / "diagnosis-scores.json").write_text(json.dumps(diagnosis_score, indent=2) + "\n")
    study["diagnosis_scores_sha256"] = sha256(out / "diagnosis-scores.json")
    (out / "run.json").write_text(json.dumps(study, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": 103,
                "valid_output": sum(r["valid_output"] for r in diagnosis_score),
                "verdict_correct": sum(r["verdict_correct"] for r in diagnosis_score),
                "strict_joint": sum(r["strict_joint"] for r in diagnosis_score),
            }
        )
    )


if __name__ == "__main__":
    main()
