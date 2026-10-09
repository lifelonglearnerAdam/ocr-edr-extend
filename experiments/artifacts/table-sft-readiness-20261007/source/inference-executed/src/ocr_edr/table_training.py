"""Training admission and document-balanced exposure for controlled table edits."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from .sft import sha256


def load_table_supervision(admission_root: Path, dataset_root: Path, *, role: str) -> list[dict]:
    """Resolve verified image identities portably; calibration/locked roles are forbidden."""
    if role not in {"train", "model_dev"}:
        raise ValueError("Only train/model-dev records can be loaded by the SFT interface")
    dataset = dataset_root.resolve()
    metadata = json.loads((dataset / "dataset.json").read_text())
    source_path = dataset / "selected_sources.jsonl"
    if sha256(source_path) != metadata["file_sha256"][source_path.name]:
        raise ValueError("Frozen source inventory hash mismatch")
    sources = {r["family_id"]: r for r in map(json.loads, source_path.read_text().splitlines())}
    admission = json.loads((admission_root / "admission.json").read_text())
    if admission["status"] != "admitted_as_published_weak_supervision":
        raise ValueError("Missing explicit training-admission status")
    path = admission_root / (role + "-sft.jsonl")
    if sha256(path) != admission["file_sha256"][path.name]:
        raise ValueError("Admitted supervision hash mismatch")
    fields = {
        "sample_id",
        "family_id",
        "document_id",
        "role",
        "modality",
        "source_image",
        "source_sha256",
        "candidate",
        "prompt",
        "target",
        "variant",
        "target_provenance",
    }
    from .table_pilot import apply_table_action
    from .table_supervision import table_action_prompt

    rows = []
    seen = set()
    image_hashes = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if set(row) != fields or row["role"] != role or row["modality"] != "table":
            raise ValueError("Table supervision schema/role mismatch")
        if row["sample_id"] in seen:
            raise ValueError("Duplicate admitted sample identity")
        seen.add(row["sample_id"])
        source = sources[row["family_id"]]
        if source["role"] != role or source["document_id"] != row["document_id"]:
            raise ValueError("Admitted source/document role mismatch")
        if role == "train" and row["family_id"] in admission["excluded_families"]:
            raise ValueError("Excluded training family reintroduced")
        if Path(row["source_image"]).name != Path(source["source_image"]).name:
            raise ValueError("Source filename identity changed during rebase")
        image = (dataset / source["source_image"]).resolve()
        image.relative_to(dataset)
        if image not in image_hashes:
            image_hashes[image] = sha256(image)
        if (
            image_hashes[image] != row["source_sha256"]
            or row["source_sha256"] != source["source_sha256"]
        ):
            raise ValueError("Table source-image hash mismatch")
        if row["prompt"] != table_action_prompt(row["candidate"]):
            raise ValueError("Prompt must derive solely from current HTML")
        _, action = apply_table_action(row["candidate"], row["target"])
        if row["variant"] == "preservation" and action != {"action": "stop"}:
            raise ValueError("Preservation supervision must target stop")
        if (
            row["target_provenance"]
            != "published_annotation_controlled_corruption_not_teacher_or_native"
        ):
            raise ValueError("Unknown table target provenance")
        rows.append({**row, "resolved_source_image": str(image)})
    if not rows:
        raise ValueError("Empty admitted supervision")
    return rows


def admit_training_records(
    rows: list[dict], excluded_families: dict[str, str]
) -> tuple[list[dict], list[dict]]:
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Nonempty unique training records required")
    if any(r["role"] != "train" for r in rows):
        raise ValueError("Only train-role records can enter optimizer admission")
    families = {r["family_id"] for r in rows}
    if set(excluded_families) - families or any(
        not isinstance(reason, str) or not reason.strip() for reason in excluded_families.values()
    ):
        raise ValueError("Review ledger names unknown families or has empty reasons")
    admitted, exclusions = [], []
    for row in rows:
        reason = excluded_families.get(row["family_id"])
        if reason is None:
            admitted.append(dict(row))
        else:
            exclusions.append(
                {
                    "sample_id": row["sample_id"],
                    "family_id": row["family_id"],
                    "document_id": row["document_id"],
                    "reason": reason,
                }
            )
    if not admitted:
        raise ValueError("No training examples survive the declared admission rule")
    return admitted, exclusions


def document_balanced_schedule(
    rows: list[dict], *, arm: str, passes_per_document: int, seed: int
) -> list[int]:
    if (
        arm not in {"all", "no_explicit_preservation"}
        or type(passes_per_document) is not int
        or passes_per_document < 1
    ):
        raise ValueError("Unknown arm or invalid document-exposure budget")
    documents = defaultdict(list)
    for index, row in enumerate(rows):
        if row["role"] != "train":
            raise ValueError("Non-training record in optimizer schedule")
        if arm == "all" or row["variant"] != "preservation":
            documents[row["document_id"]].append(index)
    if not documents or set(documents) != {r["document_id"] for r in rows}:
        raise ValueError("Every admitted document must retain training variants")
    if any(passes_per_document % len(indices) for indices in documents.values()):
        raise ValueError("Each document needs an integer number of complete variant cycles")
    rng = random.Random(seed)
    sequences = {}
    for doc in sorted(documents):
        sequence = []
        for _ in range(passes_per_document // len(documents[doc])):
            order = documents[doc].copy()
            rng.shuffle(order)
            sequence.extend(order)
        sequences[doc] = sequence
    schedule = []
    for exposure in range(passes_per_document):
        order = sorted(documents)
        rng.shuffle(order)
        schedule.extend(sequences[doc][exposure] for doc in order)
    return schedule
