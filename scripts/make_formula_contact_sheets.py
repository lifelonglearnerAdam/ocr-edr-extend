#!/usr/bin/env python3
"""Create private offline source/reference/initial/final inspection sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.loop import Observation
from ocr_edr.tex import TectonicRenderer


def main() -> None:
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="", help="Optional post-hoc audit policy suffix")
    args = parser.parse_args()
    predictions = [json.loads(s) for s in (args.run / "predictions.jsonl").read_text().splitlines()]
    inputs = [json.loads(s) for s in (args.data / "inputs.jsonl").read_text().splitlines()]
    references = {
        r["sample_id"]: r
        for r in map(json.loads, (args.data / "references.jsonl").read_text().splitlines())
    }
    index = {(r["sample_id"], r["arm"]): r for r in predictions}
    args.output.mkdir(parents=True, exist_ok=True)
    renderer = TectonicRenderer(args.output / "renders")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    arms = ["source_only", "source_first", "source_last"]
    suffix = "__" + args.policy if args.policy else ""
    width, row_height, cell_width = 1800, 200, 300
    for start in range(0, len(inputs), 4):
        cases = inputs[start : start + 4]
        sheet = Image.new("RGB", (width, 55 + row_height * len(cases)), "white")
        draw = ImageDraw.Draw(sheet)
        for col, label in enumerate(["source image", "reference (offline)", "initial", *arms]):
            draw.text((col * cell_width + 8, 10), label, font=font, fill="black")
        for row, case in enumerate(cases):
            y = 55 + row * row_height
            sample_id = case["sample_id"]
            ref = references[sample_id]
            draw.text((8, y), f"{sample_id}  {ref['family_id']}  {ref['variant']}", font=small)
            slots = [
                case["prediction"],
                *[index[(sample_id, a + suffix)]["final_prediction"] for a in arms],
            ]
            paths = [args.data / case["source_image"]]
            descriptions = ["published source", "offline reference", "initial"]
            for label, value in zip(["reference", "initial", *arms], [ref["reference"], *slots]):
                try:
                    p = Path(renderer.render(Observation(sample_id, "formula", "", value)).path)
                    paths.append(p)
                except Exception:
                    paths.append(None)
            for arm in arms:
                r = index[(sample_id, arm + suffix)]
                descriptions.append(
                    "changed" if r["final_prediction"] != r["initial_prediction"] else "preserved"
                )
            for col, (path, desc) in enumerate(zip(paths, descriptions)):
                x = col * cell_width
                draw.rectangle(
                    (x, y + 20, x + cell_width - 2, y + row_height - 2), outline="#cccccc"
                )
                draw.text((x + 8, y + row_height - 27), desc, font=small, fill="#333333")
                if path is None:
                    draw.text((x + 15, y + 85), "RENDER FAILED", font=font, fill="#a00000")
                else:
                    with Image.open(path) as original:
                        image = ImageOps.contain(original.convert("RGB"), (cell_width - 18, 135))
                    sheet.paste(
                        image,
                        (x + (cell_width - image.width) // 2, y + 25 + (135 - image.height) // 2),
                    )
        path = args.output / f"cases-{start + 1:02d}-{start + len(cases):02d}.png"
        sheet.save(path)
        print(path)


if __name__ == "__main__":
    main()
