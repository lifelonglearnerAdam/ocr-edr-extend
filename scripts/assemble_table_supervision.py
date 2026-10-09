#!/usr/bin/env python3
"""Prepare controlled train/model-dev targets without importing calibration/test roles."""

import argparse
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.sft import sha256
from ocr_edr.table_supervision import table_action_prompt, validate_table_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    receipt = json.loads((dataset / "dataset.json").read_text())
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    sources = {
        r["family_id"]: r
        for r in map(json.loads, (dataset / "selected_sources.jsonl").read_text().splitlines())
    }
    summary = {
        "target_provenance": "published annotation controlled targets, not native or teacher trajectories",
        "optimizer_steps": 0,
        "calibration_or_locked_targets_loaded": False,
        "roles": {},
    }
    for role in ["train", "model_dev"]:
        input_path = dataset / (role + "-inputs.jsonl")
        ref_path = dataset / (role + "-references.jsonl")
        for path in [input_path, ref_path]:
            if sha256(path) != receipt["file_sha256"][path.name]:
                raise ValueError("Frozen target/input file integrity mismatch")
        inputs = [json.loads(line) for line in input_path.read_text().splitlines()]
        references = {r["sample_id"]: r for r in map(json.loads, ref_path.read_text().splitlines())}
        if len(inputs) != len(references):
            raise ValueError("Supervision coverage mismatch")
        rows = []
        for case in inputs:
            if set(case) != {
                "sample_id",
                "family_id",
                "source_image",
                "source_sha256",
                "prediction",
            }:
                raise ValueError("Inference input must not contain target metadata")
            ref = references[case["sample_id"]]
            source = sources[case["family_id"]]
            if (
                ref["family_id"] != case["family_id"]
                or ref["role"] != role
                or source["role"] != role
            ):
                raise ValueError("Source/target role and family mismatch")
            image = (dataset / case["source_image"]).resolve()
            image.relative_to(dataset)
            if sha256(image) != case["source_sha256"]:
                raise ValueError("Image identity mismatch")
            target = json.dumps(ref["target_action"], ensure_ascii=False, separators=(",", ":"))
            validate_table_target(case["prediction"], target, ref["reference"])
            rows.append(
                {
                    "sample_id": case["sample_id"],
                    "family_id": case["family_id"],
                    "document_id": source["document_id"],
                    "role": role,
                    "modality": "table",
                    "source_image": str(image),
                    "source_sha256": case["source_sha256"],
                    "candidate": case["prediction"],
                    "prompt": table_action_prompt(case["prediction"]),
                    "target": target,
                    "variant": ref["variant"],
                    "target_provenance": ref["target_provenance"],
                }
            )
        path = root / (role + "-sft.jsonl")
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        summary["roles"][role] = {
            "records": len(rows),
            "documents": len({r["document_id"] for r in rows}),
            "variants": dict(Counter(r["variant"] for r in rows)),
            "sha256": sha256(path),
        }
    if {r["document_id"] for r in sources.values() if r["role"] == "train"} & {
        r["document_id"] for r in sources.values() if r["role"] == "model_dev"
    }:
        raise ValueError("Train/model-dev document overlap")
    summary["source_sha256"] = {
        str(path): sha256(path)
        for path in [
            Path(__file__),
            Path("src/ocr_edr/table_supervision.py"),
            Path("src/ocr_edr/table_pilot.py"),
        ]
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
