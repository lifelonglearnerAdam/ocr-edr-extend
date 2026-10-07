"""CPU-testable supervision boundaries for the bounded formula SFT screen."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from .formula_pilot import make_prompt

FIELDS = {
    "sample_id",
    "family_id",
    "split",
    "modality",
    "source_image",
    "source_sha256",
    "candidate",
    "prompt",
    "target",
    "variant",
    "target_provenance",
}
VARIANTS = {"preservation", "controlled_error", "native_parser_prediction"}
ARMS = {"all", "no_explicit_preservation"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return (
            hashlib.file_digest(stream, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else hashlib.sha256(stream.read()).hexdigest()
        )


def verify_model_files(snapshot: Path, receipt: dict) -> None:
    """Accept a plain pinned snapshot or its own standard HF-cache blob symlinks."""
    root = snapshot.resolve()
    if root.name != receipt["revision"] or not re.fullmatch(r"[a-f0-9]{40}", receipt["revision"]):
        raise ValueError("Pinned model revision mismatch")
    expected_cache = "models--" + receipt["model"].replace("/", "--")
    standard_cache = root.parent.name == "snapshots" and root.parent.parent.name == expected_cache
    blobs = root.parent.parent / "blobs"
    for name, expected in receipt["files"].items():
        if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Model receipt paths must be single safe file names")
        target = (root / name).resolve(strict=True)
        plain = target.parent == root
        cache_blob = (
            standard_cache
            and target.parent == blobs
            and bool(re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", target.name))
        )
        if not plain and not cache_blob:
            raise ValueError("Model file links outside its snapshot/cache blob boundary")
        if (
            not target.is_file()
            or target.stat().st_size != expected["bytes"]
            or sha256(target) != expected["sha256"]
        ):
            raise ValueError("Pinned model file integrity mismatch")


def load_supervision(
    path: Path,
    dataset_root: Path,
    *,
    split: str,
    expected_sha256: str,
) -> list[dict]:
    if split not in {"train", "dev"} or not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        raise ValueError("Invalid split or expected manifest hash")
    if sha256(path) != expected_sha256:
        raise ValueError("Supervision manifest hash mismatch")
    root = dataset_root.resolve()
    path.resolve().relative_to(root)
    rows = []
    seen = set()
    image_hashes = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError("Unexpected supervision fields")
        if row["split"] != split or row["modality"] != "formula" or row["variant"] not in VARIANTS:
            raise ValueError("Supervision split/modality/variant mismatch")
        if any(
            not isinstance(row[k], str) or not row[k]
            for k in ["sample_id", "family_id", "target", "source_image"]
        ):
            raise ValueError("Invalid supervision identity/target")
        if row["sample_id"] in seen:
            raise ValueError("Duplicate supervision sample identity")
        seen.add(row["sample_id"])
        if not isinstance(row["candidate"], str) or row["prompt"] != make_prompt(
            row["candidate"], False
        ):
            raise ValueError("Supervision prompt must use only source and current candidate")
        if not re.fullmatch(r"<latex>.+</latex>", row["target"], re.DOTALL):
            raise ValueError("Complete nonempty assistant LaTeX target required")
        if (
            row["variant"] == "preservation"
            and row["target"] != "<latex>" + row["candidate"] + "</latex>"
        ):
            raise ValueError("Preservation target must copy the original candidate exactly")
        if row["target_provenance"] != "released_annotation_or_preservation_copy":
            raise ValueError("Unknown target provenance; teacher data needs a separate protocol")
        image = (path.parent / row["source_image"]).resolve()
        image.relative_to(root)
        if image not in image_hashes:
            image_hashes[image] = sha256(image)
        if image_hashes[image] != row["source_sha256"]:
            raise ValueError("Supervision image hash mismatch")
        rows.append({**row, "resolved_source_image": str(image)})
    if not rows:
        raise ValueError("Empty supervision manifest")
    return rows


def validate_disjoint(train: list[dict], dev: list[dict]) -> None:
    if (
        not train
        or not dev
        or any(r["split"] != "train" for r in train)
        or any(r["split"] != "dev" for r in dev)
    ):
        raise ValueError("Nonempty separate train/dev manifests required")
    for field in ["sample_id", "family_id", "source_sha256"]:
        if {r[field] for r in train} & {r[field] for r in dev}:
            raise ValueError("Train/dev overlap: " + field)


def training_schedule(rows: list[dict], *, arm: str, exposures: int, seed: int) -> list[int]:
    if arm not in ARMS or type(exposures) is not int or exposures < 1:
        raise ValueError("Unknown training arm or invalid exposure budget")
    indices = [i for i, row in enumerate(rows) if arm == "all" or row["variant"] != "preservation"]
    if not indices or exposures % len(indices):
        raise ValueError("Exposure budget must cover an integer number of complete arm passes")
    rng = random.Random(seed)
    schedule = []
    for _ in range(exposures // len(indices)):
        order = indices.copy()
        rng.shuffle(order)
        schedule.extend(order)
    return schedule


def assistant_labels(prefix: list[int], complete: list[int], attention: list[int]) -> list[int]:
    if not prefix or len(complete) != len(attention) or complete[: len(prefix)] != prefix:
        raise ValueError("Chat prefix boundary or attention-mask mismatch")
    labels = [
        token if i >= len(prefix) and mask == 1 else -100
        for i, (token, mask) in enumerate(zip(complete, attention))
    ]
    if all(token == -100 for token in labels):
        raise ValueError("Assistant target has no supervised tokens")
    return labels


def proposal_contract(raw: str, *, hit_cap: bool) -> tuple[str | None, str]:
    """Reference-free format boundary; accepted syntax is not visual correctness."""
    from .formula_audit import normalize_outer_environment
    from .tex import validate_formula

    if hit_cap:
        return None, "token_cap"
    match = re.fullmatch(r"\s*<latex>(.*?)</latex>\s*", raw, re.DOTALL)
    if match is None or not match[1].strip() or "<latex>" in match[1] or "</latex>" in match[1]:
        return None, "invalid_output_contract"
    candidate, _ = normalize_outer_environment(match[1].strip())
    try:
        return validate_formula(candidate), "accepted_contract"
    except ValueError:
        return None, "unsafe_or_invalid_tex"
