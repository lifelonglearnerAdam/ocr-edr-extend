#!/usr/bin/env python3
"""Private, offline source/reference/initial/final table inspection panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.loop import Observation
from ocr_edr.table_pilot import HTMLTableRenderer

ARMS = ["source_only_rewrite", "source_first_rewrite", "source_last_rewrite", "source_only_patch"]


def main() -> None:
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = [json.loads(s) for s in (args.data / "inputs.jsonl").read_text().splitlines()]
    refs = {
        r["sample_id"]: r
        for r in map(json.loads, (args.data / "references.jsonl").read_text().splitlines())
    }
    results = {
        (r["sample_id"], r["arm"]): r
        for r in map(json.loads, (args.run / "predictions.jsonl").read_text().splitlines())
    }
    evaluation = json.loads(args.evaluation.read_text())
    scores = {(r["sample_id"], r["arm"]): r for r in evaluation["cases"]}
    args.output.mkdir(parents=True, exist_ok=True)
    renderer = HTMLTableRenderer(args.output / "renders")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    for case in inputs:
        sid = case["sample_id"]
        ref = refs[sid]
        sheet = Image.new("RGB", (2400, 1410), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (15, 10),
            f"{sid} / {case['family_id']} / {ref['variant']} - private development inspection",
            font=font,
            fill="black",
        )
        values = [
            ("source image", None),
            ("released reference (offline)", ref["reference"]),
            ("initial prediction", case["prediction"]),
        ]
        values += [(arm, results[(sid, arm)]["final_prediction"]) for arm in ARMS]
        for slot, (label, markup) in enumerate(values):
            top = slot < 3
            col = slot if top else slot - 3
            cell_width = 800 if top else 600
            x, y = col * cell_width, 55 if top else 730
            draw.rectangle((x + 3, y, x + cell_width - 3, y + 650), outline="#bbbbbb")
            draw.text((x + 15, y + 8), label, font=font, fill="black")
            if slot == 0:
                path = args.data / case["source_image"]
            else:
                path = Path(renderer.render(Observation(sid, "table", "", markup)).path)
            with Image.open(path) as original:
                image = ImageOps.contain(original.convert("RGB"), (cell_width - 24, 565))
            sheet.paste(
                image, (x + (cell_width - image.width) // 2, y + 43 + (565 - image.height) // 2)
            )
            if not top:
                row = scores[(sid, label)]
                caption = f"TEDS {row['initial_teds']:.4f} -> {row['final_teds']:.4f}; S {row['final_teds_structure']:.4f}"
                draw.text((x + 12, y + 606), caption, font=small, fill="black")
                detail = f"changed={row['changed']} rejected={row['adapter_failures']+row['render_rejections']} cap={row['hit_length_cap']}"
                draw.text((x + 12, y + 628), detail, font=small, fill="black")
        path = args.output / (sid + ".png")
        sheet.save(path)
        print(path, flush=True)


if __name__ == "__main__":
    main()
