#!/usr/bin/env python3
"""Generate a held-out synthetic diagnostic; keep reference fields out of inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.formula_pilot import FORMULAS, MathTextRenderer, pixel_signature, sha256_file
from ocr_edr.loop import Observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    renderer = MathTextRenderer(root / "images")
    cases, references = [], []
    # Two source directions for four formula types expose prior-driven correction.
    formulas = [(gold, error, error_type, "base") for gold, error, error_type in FORMULAS[:4]]
    formulas += [
        (error, gold, error_type, "counterfactual") for gold, error, error_type in FORMULAS[:4]
    ]
    for index, (gold, error, error_type, source_kind) in enumerate(formulas, start=1):
        family = f"f{index:02}"
        source = Path(renderer.render(Observation(family, "formula", "", gold)).path)
        variants = [("correct", gold), ("incorrect", error)]
        # Same source and different exact string, with raster equality checked below.
        if index <= 4:
            variants.append(("equivalent", f" {gold} "))
        for variant, prediction in variants:
            sample_id = f"c{len(cases) + 1:03}"
            initial = Path(
                renderer.render(Observation(sample_id, "formula", str(source), prediction)).path
            )
            same_raster = pixel_signature(source) == pixel_signature(initial)
            if same_raster != (variant != "incorrect"):
                raise ValueError(f"Unexpected rendering relation in {family}/{variant}")
            cases.append(
                {
                    "sample_id": sample_id,
                    "family_id": family,
                    "source_image": str(source.relative_to(root)),
                    "source_sha256": sha256_file(source),
                    "prediction": prediction,
                }
            )
            references.append(
                {
                    "sample_id": sample_id,
                    "family_id": family,
                    "reference": gold,
                    "variant": variant,
                    "source_kind": source_kind,
                    "error_type": error_type if variant == "incorrect" else None,
                }
            )
    for name, rows in [("inputs.jsonl", cases), ("references.jsonl", references)]:
        (root / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "date": "2026-10-02",
        "dataset": "controlled_mathtext_formula_diagnostic_v1",
        "cases": len(cases),
        "source_families": len(formulas),
        "references_separate": True,
        "held_out_from_training": True,
        "renderer": "matplotlib.mathtext:cm:160dpi:22pt:16px-border",
        "input_sha256": sha256_file(root / "inputs.jsonl"),
        "reference_sha256": sha256_file(root / "references.jsonl"),
        "limitations": [
            "synthetic",
            "single font",
            "hand-authored perturbations",
            "no real parser errors",
            "equivalent controls are whitespace-only",
        ],
    }
    (root / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
