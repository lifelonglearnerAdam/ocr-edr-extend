#!/usr/bin/env python3
"""Prepare bounded controlled table supervision with four disjoint document roles."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import tarfile
import time
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.pubtabnet import annotation_html, bounded_supervision, document_id, document_role
from ocr_edr.sft import sha256
from ocr_edr.table_pilot import parse_table

QUOTAS = {"train": 128, "model_dev": 32, "gate_calibration": 32, "locked_evaluation": 64}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.archive) != "90c55e733c85c98edf6d350b77f1e4c23767577555fc932e44ad723674de8d3e":
        raise ValueError("Official linked whole-archive identity mismatch")
    root = args.output.resolve()
    root.mkdir(exist_ok=False, parents=True)
    images_dir = root / "images"
    images_dir.mkdir()
    inventory = json.loads((args.preflight / "image_document_inventory.json").read_text())
    groups = inventory["document_ids"]
    test_docs = set(groups["test"])
    held_docs = test_docs | set(groups["val"])
    held_images = [
        json.loads(line)
        for line in (args.preflight / "held_out_image_keys.jsonl").read_text().splitlines()
    ]
    held_bytes = {r["bytes_sha256"] for r in held_images}
    held_pixels = {r["pixel_sha256"] for r in held_images}
    inspected_paths = [Path("experiments/runs/table-development-20261004/data-v2/sources.jsonl")]
    inspected_sha = set()
    for p in inspected_paths:
        for line in p.read_text().splitlines():
            inspected_sha.add(json.loads(line)["source_sha256"])
    annotations = args.preflight / "PubTabNet_2.0.0.jsonl"
    candidates = {role: [] for role in QUOTAS}
    counts = Counter()
    started = time.monotonic()
    with annotations.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            row = json.loads(line)
            doc = document_id(row["filename"])
            role = document_role(
                doc, row["split"], held_out_documents=held_docs, test_documents=test_docs
            )
            if role is None:
                counts["blocked_published_document"] += 1
                continue
            order = hashlib.sha256(
                ("20261007-pubtabnet-order:" + row["filename"]).encode()
            ).digest()
            candidates[role].append((order, offset, row["filename"], doc, row["split"]))
    pool = []
    eligibility = []
    for role, queue in candidates.items():
        queue.sort()
        documents = set()
        with annotations.open("rb") as stream:
            for _, offset, filename, doc, published in queue:
                if doc in documents:
                    continue
                stream.seek(offset)
                row = json.loads(stream.readline())
                try:
                    reference = annotation_html(row["html"])
                    table, _ = parse_table(reference)
                    rows = len(table.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr"))
                    cells = len(table.xpath(".//td|.//th"))
                    if not 2 <= rows <= 12 or not 4 <= cells <= 32 or len(reference) > 2000:
                        raise ValueError("outside_predeclared_table_size_limits")
                    variants = bounded_supervision(reference)
                except ValueError as error:
                    reason = str(error)
                    counts[reason] += 1
                    eligibility.append({"filename": filename, "role": role, "reason": reason})
                    continue
                documents.add(doc)
                pool.append(
                    {
                        "filename": filename,
                        "published_split": published,
                        "document_id": doc,
                        "role": role,
                        "archive_path": f"pubtabnet/{published}/{filename}",
                        "annotation_sha256": hashlib.sha256(
                            json.dumps(row, sort_keys=True, ensure_ascii=False).encode()
                        ).hexdigest(),
                        "reference": reference,
                        "rows": rows,
                        "cells": cells,
                        "variants": variants,
                        "bboxes": [cell.get("bbox") for cell in row["html"]["cells"]],
                    }
                )
                if len(documents) == 4 * QUOTAS[role]:
                    break
        if len(documents) != 4 * QUOTAS[role]:
            raise ValueError("Insufficient predeclared image-screening pool for " + role)
        print("Candidate pool frozen before image/model access:", role, len(documents), flush=True)
    write_rows(root / "candidate_pool.jsonl", pool)
    desired = {row["archive_path"]: row for row in pool}
    extracted = set()
    from PIL import Image

    last = time.monotonic()
    with tarfile.open(args.archive, "r|gz") as archive:
        for member in archive:
            if member.name not in desired:
                continue
            if not member.isfile():
                raise ValueError("Selected archive image is not a regular file")
            row = desired[member.name]
            data = archive.extractfile(member).read()
            with Image.open(io.BytesIO(data)) as image:
                rgb = image.convert("RGB")
                pixel = hashlib.sha256(struct.pack("<II", *rgb.size) + rgb.tobytes()).hexdigest()
                row["dimensions"] = list(rgb.size)
                if any(
                    box
                    and not (
                        0 <= box[0] <= box[2] <= rgb.width and 0 <= box[1] <= box[3] <= rgb.height
                    )
                    for box in row["bboxes"]
                ):
                    row["image_exclusion"] = "annotated_cell_bbox_outside_image"
            byte_sha = hashlib.sha256(data).hexdigest()
            row.update(source_sha256=byte_sha, pixel_sha256=pixel)
            if row["role"] in {"train", "model_dev"} and (
                byte_sha in held_bytes or pixel in held_pixels
            ):
                row["image_exclusion"] = "published_held_out_exact_image_or_pixels"
            if byte_sha in inspected_sha:
                row["image_exclusion"] = "previously_inspected_source_image"
            filename = row["filename"]
            (images_dir / filename).write_bytes(data)
            extracted.add(member.name)
            if time.monotonic() - last > 25:
                print("Extracted frozen pool images:", len(extracted), "/", len(pool), flush=True)
                last = time.monotonic()
    if extracted != desired.keys():
        raise ValueError("Some preselected annotation images are missing from archive")
    selected = []
    used_bytes, used_pixels, used_html = set(), set(), set()
    selected_count = Counter()
    for row in pool:
        role = row["role"]
        if selected_count[role] == QUOTAS[role]:
            continue
        reason = row.get("image_exclusion")
        content_key = hashlib.sha256(row["reference"].encode()).hexdigest()
        if (
            row["source_sha256"] in used_bytes
            or row["pixel_sha256"] in used_pixels
            or content_key in used_html
        ):
            reason = "selected_exact_image_pixels_or_content_duplicate"
        if reason:
            counts[reason] += 1
            eligibility.append({"filename": row["filename"], "role": role, "reason": reason})
            continue
        selected_count[role] += 1
        row["family_id"] = "p" + f"{len(selected)+1:04}"
        row["source_image"] = "images/" + row["filename"]
        selected.append(row)
        used_bytes.add(row["source_sha256"])
        used_pixels.add(row["pixel_sha256"])
        used_html.add(content_key)
    if dict(selected_count) != QUOTAS:
        raise ValueError("Independent image-screened quotas are not met")
    for left in QUOTAS:
        for right in QUOTAS:
            if left < right:
                if {r["document_id"] for r in selected if r["role"] == left} & {
                    r["document_id"] for r in selected if r["role"] == right
                }:
                    raise ValueError("Document leakage between final roles")
    role_cases = {}
    for role in QUOTAS:
        inputs, targets, source_only = [], [], []
        for row in selected:
            if row["role"] != role:
                continue
            source_only.append({k: row[k] for k in ["family_id", "source_image", "source_sha256"]})
            for variant in row["variants"]:
                sample = row["family_id"] + "-" + variant["variant"]
                inputs.append(
                    {
                        "sample_id": sample,
                        "family_id": row["family_id"],
                        "source_image": row["source_image"],
                        "source_sha256": row["source_sha256"],
                        "prediction": variant["prediction"],
                    }
                )
                targets.append(
                    {
                        "sample_id": sample,
                        "family_id": row["family_id"],
                        "role": role,
                        "document_id": row["document_id"],
                        "reference": row["reference"],
                        "variant": variant["variant"],
                        "target_action": variant["target_action"],
                        "target_provenance": "published_annotation_controlled_corruption_not_teacher_or_native",
                    }
                )
        write_rows(root / (role + "-inputs.jsonl"), inputs)
        write_rows(root / (role + "-references.jsonl"), targets)
        write_rows(root / (role + "-source-inputs.jsonl"), source_only)
        role_cases[role] = {
            "sources": len(source_only),
            "cases": len(inputs),
            "variants": dict(Counter(r["variant"] for r in targets)),
        }
    write_rows(
        root / "selected_sources.jsonl",
        [
            {k: v for k, v in row.items() if k not in {"reference", "variants", "bboxes"}}
            for row in selected
        ],
    )
    write_rows(root / "eligibility.jsonl", eligibility)
    files = [p for p in root.iterdir() if p.is_file()]
    selected_images = [root / row["source_image"] for row in selected]
    metadata = {
        "status": "prepared",
        "archive_sha256": sha256(args.archive),
        "annotation_file_sha256": sha256(annotations),
        "driver_sha256": sha256(Path(__file__)),
        "pubtabnet_module_sha256": sha256(Path("src/ocr_edr/pubtabnet.py")),
        "protocol_sha256": sha256(Path("docs/research/PUBTABNET_PREPARATION_20261007.md")),
        "roles": role_cases,
        "original_document_intersections": {"train_val": 6105, "train_test": 6124, "val_test": 216},
        "document_role_intersection_after_screening": 0,
        "model_outputs_used_for_selection": False,
        "optimizer_steps": 0,
        "locked_source_images_visually_inspected": False,
        "annotation_derived_controlled_errors_only": True,
        "native_table_parser_identity": None,
        "near_duplicate_scope": "article group, exact bytes, decoded pixels and identical canonical HTML; no general perceptual near-duplicate proof",
        "exclusions": dict(counts),
        "candidate_pool_sources": len(pool),
        "elapsed_seconds": time.monotonic() - started,
        "file_sha256": {
            str(p.relative_to(root)): sha256(p) for p in sorted(files + selected_images)
        },
    }
    (root / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: metadata[k]
                for k in [
                    "status",
                    "roles",
                    "document_role_intersection_after_screening",
                    "exclusions",
                ]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
