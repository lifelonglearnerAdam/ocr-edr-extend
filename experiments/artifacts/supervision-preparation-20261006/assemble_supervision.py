"""Join frozen native proposals offline and prepare direct-answer SFT records.

The recognizer has already completed without opening reference files. Targets
are released annotations or unchanged preservation candidates, not teacher traces.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))

from ocr_edr.formula_pilot import make_prompt, sha256_file, validate_pilot_inputs


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def conservative_key(text):
    return re.sub(r"\s+", "", text)


def main():
    data = ROOT / "data/processed/unimer-supervision-20261006"
    audit = ROOT / "experiments/runs/supervision-preflight-20261006"
    output = data / "supervision"
    output.mkdir(exist_ok=False)
    sources = read_rows(data / "sources.jsonl")
    summary = {"new_model_calls": 160, "training_steps": 0, "weights_changed": False, "splits": {}}
    for split, expected in [("train", 128), ("dev", 32)]:
        source_list = [row for row in sources if row["split"] == split]
        source_map = {row["family_id"]: row for row in source_list}
        path = audit / ("nougat-" + split)
        metadata = json.loads((path / "run.json").read_text())
        if metadata["status"] != "completed" or metadata["completed_sources"] != expected:
            raise ValueError("Native recognition is incomplete")
        if metadata["reference_access"] != "none":
            raise ValueError("Native inference must not read target references")
        if metadata["input_sha256"] != sha256_file(data / split / "source_inputs.jsonl"):
            raise ValueError("Recognizer input receipt mismatch")
        native = read_rows(path / "predictions.jsonl")
        if len(native) != expected or len({row["family_id"] for row in native}) != expected:
            raise ValueError("Missing/duplicate native predictions")
        if {row["family_id"] for row in native} != set(source_map):
            raise ValueError("Native recognition source coverage mismatch")
        native_inputs = []
        native_refs = []
        for row in native:
            source = source_map[row["family_id"]]
            if row["source_sha256"] != source["source_sha256"] or row["source_image"] != source["source_image"]:
                raise ValueError("Native source identity mismatch")
            native_inputs.append({
                "sample_id": row["family_id"] + "-native", "family_id": row["family_id"],
                "source_image": f"../{split}/{row['source_image']}", "source_sha256": row["source_sha256"],
                "prediction": row["prediction"],
            })
            native_refs.append({
                "sample_id": row["family_id"] + "-native", "family_id": row["family_id"],
                "reference": source["reference"], "variant": "native_parser_prediction",
                "source_kind": "UniMER-1M", "error_type": "untyped_native_output",
            })
        validate_pilot_inputs(native_inputs, output)
        write_rows(output / (split + "-native-inputs.jsonl"), native_inputs)
        write_rows(output / (split + "-native-references.jsonl"), native_refs)

        inputs = read_rows(data / split / "inputs.jsonl") + native_inputs
        refs = read_rows(data / split / "references.jsonl") + native_refs
        ref_map = {row["sample_id"]: row for row in refs}
        if len(ref_map) != len(inputs):
            raise ValueError("Supervision input/target coverage mismatch")
        records = []
        for row in inputs:
            ref = ref_map[row["sample_id"]]
            if row["family_id"] != ref["family_id"]:
                raise ValueError("Supervision family mismatch")
            target = row["prediction"] if ref["variant"] == "preservation" else ref["reference"]
            if ref["variant"] == "preservation" and row["prediction"] != ref["reference"]:
                raise ValueError("Preservation input must equal its released target bytes")
            records.append({
                "sample_id": row["sample_id"], "family_id": row["family_id"],
                "split": split, "modality": "formula",
                "source_image": f"../{split}/{source_map[row['family_id']]['source_image']}",
                "source_sha256": row["source_sha256"], "candidate": row["prediction"],
                "prompt": make_prompt(row["prediction"], False), "target": "<latex>" + target + "</latex>",
                "variant": ref["variant"], "target_provenance": "released_annotation_or_preservation_copy",
            })
        write_rows(output / (split + "-sft.jsonl"), records)
        lexical_matches = sum(
            conservative_key(row["prediction"]) == conservative_key(source_map[row["family_id"]]["reference"])
            for row in native
        )
        summary["splits"][split] = {
            "sources": expected, "sft_examples": len(records),
            "variants": dict(Counter(row["variant"] for row in records)),
            "native_records": len(native), "native_empty_outputs": sum(not row["prediction"].strip() for row in native),
            "native_length_cap_hits": sum(row["hit_length_cap"] for row in native),
            "native_whitespace_key_matches": lexical_matches,
            "native_whitespace_key_nonmatches": expected - lexical_matches,
            "native_generation_seconds": sum(row["generation_seconds"] for row in native),
            "native_output_tokens": sum(row["output_tokens"] for row in native),
            "recognizer_input_sha256": metadata["input_sha256"],
            "recognizer_predictions_sha256": sha256_file(path / "predictions.jsonl"),
            "sft_sha256": sha256_file(output / (split + "-sft.jsonl")),
        }
    summary.update({
        "model": "Norm/nougat-latex-base", "model_revision": "6c6f2afe62ae51d57d0621ea9f2e11e02fb8385a",
        "processor_revision": "d735d3a31bfd0cd48a020e01c5233a9154c6d4c2",
        "reference_access_during_inference": "none",
        "model_outputs_used_for_selection": False, "supervision_assembly_script_sha256": sha256_file(Path(__file__)),
        "native_match_metric": "whitespace-removed string equality only; not visual correctness, CDM or TEDS",
        "training_contract": "assistant target tokens only; dev records excluded from optimizer updates; trainer not executed",
        "targets_source_audited": "fixed first eight sources per split inspected; not all annotations validated",
    })
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (audit / "supervision_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
