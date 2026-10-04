#!/usr/bin/env python3
"""Freeze an exploratory, renderer-supported subset from the pinned UniMER archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.formula_pilot import MathTextRenderer, pixel_signature, sha256_file
from ocr_edr.loop import Observation


def perturb(formula: str) -> tuple[str, str] | None:
    for symbol, name in [("^", "exponent"), ("_", "subscript")]:
        match = re.search(re.escape(symbol) + r"\s*\{\s*([0-9])\s*\}", formula)
        if match:
            start, end = match.span(1)
            digit = str((int(match[1]) + 1) % 10)
            return formula[:start] + digit + formula[end:], name
    match = re.search(r"(?<!\\)[+-]", formula)
    if match:
        symbol = "-" if match[0] == "+" else "+"
        return formula[: match.start()] + symbol + formula[match.end() :], "operator"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-split", type=int, default=8)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "images").mkdir()
    renderer = MathTextRenderer(root / "eligibility_renders")
    inputs, references, sources, eligibility = [], [], [], []
    source_count = 0
    with zipfile.ZipFile(args.archive) as archive:
        for split in ["spe", "sce"]:
            labels = archive.read(f"UniMER-Test/{split}.txt").decode().splitlines()
            names = [
                n
                for n in archive.namelist()
                if n.startswith(f"UniMER-Test/{split}/") and n.endswith(".png")
            ]
            names.sort(key=lambda n: hashlib.sha256(f"20261003:{n}".encode()).hexdigest())
            selected = 0
            for name in names:
                # Upstream Im2LatexDataset uses the numeric filename as its annotation index.
                annotation_index = int(Path(name).stem)
                if annotation_index >= len(labels):
                    raise ValueError("Archive annotation mapping is invalid")
                gold = labels[annotation_index]
                info = {"archive_path": name, "annotation_index": annotation_index, "split": split}
                if not gold.strip() or len(gold) > 200:
                    eligibility.append({**info, "reason": "empty_or_over_200_characters"})
                    continue
                corrupted = perturb(gold)
                if corrupted is None:
                    eligibility.append({**info, "reason": "no_defined_single_symbol_perturbation"})
                    continue
                error, error_type = corrupted
                try:
                    render_good = renderer.render(Observation("eligibility", "formula", "", gold))
                    render_bad = renderer.render(Observation("eligibility", "formula", "", error))
                    different = pixel_signature(Path(render_good.path)) != pixel_signature(
                        Path(render_bad.path)
                    )
                except Exception as exc:
                    eligibility.append(
                        {**info, "reason": "unsupported_mathtext", "error_type": type(exc).__name__}
                    )
                    continue
                if not different:
                    eligibility.append({**info, "reason": "no_exact_raster_change"})
                    continue
                source_count += 1
                family = f"u{source_count:03}"
                relative = f"images/{family}.png"
                image = root / relative
                image.write_bytes(archive.read(name))
                image_hash = sha256_file(image)
                sources.append(
                    {
                        **info,
                        "family_id": family,
                        "source_image": relative,
                        "source_sha256": image_hash,
                        "reference": gold,
                    }
                )
                eligibility.append({**info, "reason": "selected", "family_id": family})
                for variant, prediction in [("correct", gold), ("incorrect", error)]:
                    sample_id = f"r{len(inputs)+1:03}"
                    inputs.append(
                        {
                            "sample_id": sample_id,
                            "family_id": family,
                            "source_image": relative,
                            "source_sha256": image_hash,
                            "prediction": prediction,
                        }
                    )
                    references.append(
                        {
                            "sample_id": sample_id,
                            "family_id": family,
                            "reference": gold,
                            "variant": variant,
                            "source_kind": split,
                            "error_type": error_type if variant == "incorrect" else None,
                        }
                    )
                selected += 1
                if selected == args.per_split:
                    break
            if selected != args.per_split:
                raise ValueError("Insufficient eligible formulas")
    for name, rows in [
        ("inputs.jsonl", inputs),
        ("references.jsonl", references),
        ("sources.jsonl", sources),
        (
            "source_inputs.jsonl",
            [
                {k: row[k] for k in ["family_id", "source_image", "source_sha256"]}
                for row in sources
            ],
        ),
        ("eligibility.jsonl", eligibility),
    ]:
        (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    metadata = {
        "dataset": "wanderkid/UniMER_Dataset",
        "dataset_revision": "2343ddd963290469da36ca83e3a56c66e068add9",
        "archive_sha256": sha256_file(args.archive),
        "annotation_mapping_source_revision": "5a2c80d96b1d2dba447ff18d873e5fb73ba03c35",
        "mapping": "numeric image basename indexes the split annotation file",
        "selection": "sha256(20261003:archive_path) order; first eligible per split",
        "per_split": args.per_split,
        "cases": len(inputs),
        "sources": len(sources),
        "reference_access_for_selection": True,
        "model_outputs_used_for_selection": False,
        "role": "exploratory development; permanently exclude from future frozen test claims",
        "metric": "restricted same-renderer raster proxy; no CDM or VisFix",
        "limits": [
            "MathText-supported and <=200-character labels only",
            "controlled one-symbol errors, not native parser predictions",
            "published benchmark sources may have appeared in model pretraining",
        ],
        "input_sha256": sha256_file(root / "inputs.jsonl"),
        "reference_sha256": sha256_file(root / "references.jsonl"),
    }
    (root / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
