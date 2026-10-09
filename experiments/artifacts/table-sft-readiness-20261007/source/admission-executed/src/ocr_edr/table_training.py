"""Training admission and document-balanced exposure for controlled table edits."""

from __future__ import annotations

import random
from collections import defaultdict


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
