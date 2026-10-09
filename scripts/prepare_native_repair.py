#!/usr/bin/env python3
"""Preserve every selected OCR output while separating native-repair labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.formula_pilot import sha256_file


def prepare(sources: list[dict], predictions: list[dict]):
    by_id = {r["family_id"]: r for r in predictions}
    if len(by_id) != len(predictions) or {r["family_id"] for r in sources} != set(by_id):
        raise ValueError("Native predictions must cover every selected source exactly once")
    inputs, references = [], []
    for index, source in enumerate(sources, start=1):
        prediction = by_id[source["family_id"]]
        if prediction["source_sha256"] != source["source_sha256"]:
            raise ValueError("Native predictions are for a different source image")
        sample_id = f"n{index:03}"
        inputs.append(
            {
                "sample_id": sample_id,
                "family_id": source["family_id"],
                "source_image": source["source_image"],
                "source_sha256": source["source_sha256"],
                "prediction": prediction["prediction"],
            }
        )
        references.append(
            {
                "sample_id": sample_id,
                "family_id": source["family_id"],
                "reference": source["reference"],
                "source_kind": source["split"],
                "variant": "native",
                "error_type": None,
            }
        )
    return inputs, references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    def read(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    inputs, references = prepare(read(args.sources), read(args.predictions))
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    # Relative source paths stay usable by linking the existing source-image directory.
    (root / "images").symlink_to(args.sources.resolve().parent / "images", target_is_directory=True)
    for name, rows in [("inputs.jsonl", inputs), ("references.jsonl", references)]:
        (root / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "role": "native OCR development; all selected sources retained",
                "sources_sha256": sha256_file(args.sources),
                "native_predictions_sha256": sha256_file(args.predictions),
                "sources": len(inputs),
                "removed_for_model_performance": 0,
                "input_sha256": sha256_file(root / "inputs.jsonl"),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Prepared {len(inputs)} native predictions without performance filtering")


if __name__ == "__main__":
    main()
