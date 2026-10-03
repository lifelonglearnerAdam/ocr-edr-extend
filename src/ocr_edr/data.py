"""Standalone region manifests; benchmark references stay outside policy observations."""

from __future__ import annotations

import json
from pathlib import Path

from .loop import Observation
from .metrics import file_sha256


def load_manifest(path: Path, source_root: Path | None = None) -> list[dict]:
    root = (source_root or path.parent).resolve()
    records = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for field in ["sample_id", "page_id", "benchmark", "source_image"]:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Line {line_number}: {field} must be a nonempty string")
        if row["sample_id"] in seen:
            raise ValueError("Duplicate sample identity")
        seen.add(row["sample_id"])
        if row.get("modality") not in {"formula", "table"}:
            raise ValueError("Only formula/table region records are supported")
        if row.get("split") not in {"train", "dev", "test"}:
            raise ValueError("Explicit train/dev/test split is required")
        if type(row.get("test_only")) is not bool:
            raise ValueError("Explicit boolean test_only is required")
        if row["split"] == "test" and not row["test_only"]:
            raise ValueError("Test records must be marked test_only=true")
        if row["split"] == "train" and row["test_only"]:
            raise ValueError("Test-only records cannot be assigned to training")
        if not isinstance(row.get("prediction"), str):
            raise ValueError("Prediction must be a string, including empty OCR outputs")
        if "reference" in row and not isinstance(row["reference"], str):
            raise ValueError("Reference must be a string when supplied")
        source = Path(row["source_image"])
        source = (source if source.is_absolute() else root / source).resolve()
        source.relative_to(root)
        if not source.is_file():
            raise FileNotFoundError(source)
        records.append(
            {**row, "source_image": str(source), "source_image_sha256": file_sha256(source)}
        )
    if not records:
        raise ValueError("Manifest is empty")
    return records


def observation(record: dict) -> Observation:
    """The policy boundary intentionally discards references and benchmark labels."""
    return Observation(
        record["sample_id"], record["modality"], record["source_image"], record["prediction"]
    )


def assert_trainable(records: list[dict], held_out: list[dict]) -> None:
    """Check explicit splits, parent page identities and exact image overlap.

    This is not a perceptual near-duplicate detector; review augmented/derived
    versions of held-out pages before finalizing a training dataset.
    """
    if not records or not held_out:
        raise ValueError("Training validation requires nonempty training and held-out manifests")
    if any(row["split"] == "train" for row in held_out):
        raise ValueError("Held-out manifest contains training records")
    pages = {(row["benchmark"], row["page_id"]) for row in held_out}
    hashes = {row["source_image_sha256"] for row in held_out}
    for row in records:
        if row["split"] != "train" or row["test_only"]:
            raise ValueError("Training requires split=train and test_only=false")
        if (row["benchmark"], row["page_id"]) in pages or row["source_image_sha256"] in hashes:
            raise ValueError("Training overlaps a held-out page or exact image")
