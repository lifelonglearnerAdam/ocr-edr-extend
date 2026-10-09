"""Private source/reference panels; no model predictions are loaded."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo / "src"))
from ocr_edr.loop import Observation
from ocr_edr.sft import sha256
from ocr_edr.table_pilot import HTMLTableRenderer
from ocr_edr.table_sft_screen import load_table_screen_inputs

data = repo / "data/processed/pubtabnet-four-roles-20261007"
out = Path(__file__).parent / "model-dev-reference-audit"
out.mkdir(exist_ok=False)
inputs = {r["sample_id"]: r for r in load_table_screen_inputs(data)}
path = data / "model_dev-references.jsonl"
metadata = json.loads((data / "dataset.json").read_text())
assert sha256(path) == metadata["file_sha256"][path.name]
refs = [json.loads(l) for l in path.read_text().splitlines()]
refs = [r for r in refs if r["variant"] == "preservation"]
assert len(refs) == 32 and all(r["role"] == "model_dev" for r in refs)
receipt = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "scope": "all 32 model-dev source/released-reference pairs; no prediction output access",
    "selection": "one preservation variant for every frozen model-dev family",
    "method": "assistant visual inspection; not an independent human character-level audit",
    "images": 32,
    "calibration_locked_images_loaded": False,
    "reference_sha256": sha256(path),
    "driver_sha256": sha256(Path(__file__)),
    "reference_edits": False,
    "exclude_cases_based_on_audit": False,
    "purpose": "flag clear discrepancies and uncertainty; retain all 103 cases for scoring",
}
(out / "protocol.json").write_text(json.dumps(receipt, indent=2) + "\n")
renderer = HTMLTableRenderer(out / "renders")
font_path = subprocess.check_output(["fc-match", "-f", "%{file}", "sans-serif"], text=True)
font = ImageFont.truetype(font_path, 26)
panels = []
for ref in refs:
    case = inputs[ref["sample_id"]]
    source = data / case["source_image"]
    render = Path(renderer.render(Observation(ref["sample_id"], "table", "", ref["reference"])).path)
    images = []
    for path in [source, render]:
        with Image.open(path) as image:
            images.append(ImageOps.contain(image.convert("RGB"), (960, 840)))
    height = max(im.height for im in images) + 120
    panel = Image.new("RGB", (2000, height), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((20, 10), ref["family_id"] + " / " + source.name + " / MODEL-DEV", font=font, fill="black")
    for i, (title, image) in enumerate(zip(["SOURCE IMAGE", "PUBLISHED REFERENCE (generic CSS)"], images)):
        draw.text((20 + 1000 * i, 48), title, font=font, fill="black")
        panel.paste(image, (20 + 1000 * i, 98))
    dest = out / (ref["family_id"] + ".png")
    panel.save(dest)
    panels.append({"family_id": ref["family_id"], "document_id": ref["document_id"],
                   "source_sha256": case["source_sha256"], "panel_sha256": sha256(dest)})
    print(ref["family_id"], flush=True)
(out / "panels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in panels))
