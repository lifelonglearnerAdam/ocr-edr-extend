#!/usr/bin/env python3
"""Freeze a bounded official-demo table subset without inspecting repair output."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.formula_pilot import sha256_file
from ocr_edr.table_pilot import parse_table


def main() -> None:
    from lxml import etree
    from PIL import Image, ImageOps

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory")
    root = args.official_root.resolve()
    revision = json.loads((root / "revision.json").read_text())["revision"]
    pages = json.loads((root / "demo_data/omnidocbench_demo/OmniDocBench_demo.json").read_text())
    by_page = {p["page_info"]["image_path"]: p for p in pages}
    records = json.loads((root / "result/end2end_quick_match_table_result.json").read_text())
    records.sort(
        key=lambda r: hashlib.sha256(
            f"20261004:{r['img_id']}:{r['gt_position'][0]}".encode()
        ).hexdigest()
    )
    selected, eligibility, used_pages = [], [], set()
    for record in records:
        table, serialized = parse_table(record["gt"])
        count = len(table.xpath(".//td|.//th"))
        reasons = []
        if len(record["norm_gt"]) > 850:
            reasons.append("reference_length")
        if count > 32:
            reasons.append("cell_count")
        if record["img_id"] in used_pages:
            reasons.append("repeated_parent_page")
        page = by_page[record["img_id"]]
        regions = [
            d
            for d in page["layout_dets"]
            if d["category_type"] == "table" and d["order"] == record["gt_position"][0]
        ]
        if len(regions) != 1:
            raise ValueError("Ambiguous official page/annotation mapping")
        chosen = not reasons and len(selected) < 4
        eligibility.append(
            {
                "img_id": record["img_id"],
                "gt_position": record["gt_position"],
                "cells": count,
                "reference_chars": len(record["norm_gt"]),
                "eligible": not reasons,
                "selected": chosen,
                "reasons": reasons or ([] if chosen else ["fixed_source_limit"]),
            }
        )
        if chosen:
            selected.append((record, table, serialized, page, regions[0]))
            used_pages.add(record["img_id"])
    if len(selected) != 4:
        raise ValueError("Insufficient sources under the fixed eligibility rules")
    args.output.mkdir(parents=True)
    (args.output / "images").mkdir()
    inputs, references, sources = [], [], []
    for index, (record, table, serialized, page, region) in enumerate(selected, 1):
        family = f"t{index:03d}"
        img_id = record["img_id"]
        relative = "demo_data/omnidocbench_demo/images/" + img_id
        path = root / relative
        if not path.exists():
            request = urllib.request.Request(
                f"https://raw.githubusercontent.com/opendatalab/OmniDocBench/{revision}/{relative}",
                headers={"User-Agent": "ocr-edr-research"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        polygon = region["poly"]
        bounds = [
            math.floor(min(polygon[::2])),
            math.floor(min(polygon[1::2])),
            math.ceil(max(polygon[::2])),
            math.ceil(max(polygon[1::2])),
        ]
        image_path = args.output / "images" / (family + ".png")
        with Image.open(path) as original:
            declared = [page["page_info"]["width"], page["page_info"]["height"]]
            actual = list(original.size)
            dimension_status = "matched"
            if actual != declared:
                if actual != declared[::-1]:
                    raise ValueError("Page dimensions disagree with annotations")
                dimension_status = "swapped_annotation_dimension_metadata"
            if not (
                0 <= bounds[0] < bounds[2] <= actual[0] and 0 <= bounds[1] < bounds[3] <= actual[1]
            ):
                raise ValueError("Annotation polygon does not fit the actual page image")
            ImageOps.expand(original.convert("RGB").crop(bounds), border=12, fill="white").save(
                image_path
            )
        source_hash = sha256_file(image_path)
        perturb = copy.deepcopy(table)
        changed = False
        for cell in perturb.xpath(".//td|.//th"):
            for element in cell.iter():
                if element.text:
                    for position, character in enumerate(element.text):
                        if character.isascii() and character.isdigit():
                            element.text = (
                                element.text[:position]
                                + str((int(character) + 1) % 10)
                                + element.text[position + 1 :]
                            )
                            changed = True
                            break
                if changed:
                    break
            if changed:
                break
        if not changed:
            raise ValueError("Selected source lacks a digit perturbation")
        variants = [
            ("reference_control", serialized),
            ("cell_perturbation", etree.tostring(perturb, encoding="unicode", method="html")),
            ("published_demo_prediction", record["pred"]),
        ]
        span_table = copy.deepcopy(table)
        for cell in span_table.xpath(".//td|.//th"):
            found = False
            for attribute in ["rowspan", "colspan"]:
                value = int(cell.get(attribute, "1"))
                if value > 1:
                    cell.set(attribute, str(value - 1))
                    variants.append(
                        (
                            "span_perturbation",
                            etree.tostring(span_table, encoding="unicode", method="html"),
                        )
                    )
                    found = True
                    break
            if found:
                break
        for variant, prediction in variants:
            sample_id = f"d{len(inputs) + 1:03d}"
            inputs.append(
                {
                    "sample_id": sample_id,
                    "family_id": family,
                    "source_image": f"images/{family}.png",
                    "source_sha256": source_hash,
                    "prediction": prediction,
                }
            )
            references.append(
                {
                    "sample_id": sample_id,
                    "family_id": family,
                    "parent_page": img_id,
                    "annotation_id": region["anno_id"],
                    "gt_position": record["gt_position"],
                    "variant": variant,
                    "reference": record["gt"],
                }
            )
        sources.append(
            {
                "family_id": family,
                "parent_page": img_id,
                "annotation_id": region["anno_id"],
                "gt_position": record["gt_position"],
                "page_sha256": sha256_file(path),
                "crop_bounds": bounds,
                "declared_dimensions": declared,
                "actual_dimensions": actual,
                "dimension_status": dimension_status,
                "padding": 12,
                "source_sha256": source_hash,
            }
        )
    for name, values in [
        ("inputs", inputs),
        ("references", references),
        ("sources", sources),
        ("eligibility", eligibility),
    ]:
        (args.output / (name + ".jsonl")).write_text(
            "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values)
        )
    (args.output / "dataset.json").write_text(
        json.dumps(
            {
                "repository": "opendatalab/OmniDocBench",
                "revision": revision,
                "source_records_sha256": sha256_file(
                    root / "result/end2end_quick_match_table_result.json"
                ),
                "role": "development; exclude source pages from future frozen tests",
                "selection": "first four eligible by sha256(20261004:img_id:gt_position), unique parent pages",
                "source_count": len(selected),
                "cases": len(inputs),
                "source_prediction_parser": "not identified in the published demo records",
                "model_outputs_used_for_selection": False,
                "input_sha256": sha256_file(args.output / "inputs.jsonl"),
                "reference_sha256": sha256_file(args.output / "references.jsonl"),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"sources": sources, "cases": len(inputs)}, indent=2))


if __name__ == "__main__":
    main()
