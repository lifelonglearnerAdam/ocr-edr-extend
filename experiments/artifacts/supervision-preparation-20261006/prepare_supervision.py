"""Frozen data-preparation procedure for the 2026-10-06 supervision stage.

This experiment script uses annotations to prepare targets and filters, never
model outputs. All raw data and prepared targets remain outside public artifacts.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import struct
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_unimer_development import perturb

from ocr_edr.formula_pilot import FORMULAS, pixel_signature, sha256_file
from ocr_edr.loop import Observation
from ocr_edr.tex import TectonicRenderer


def label_key(text):
    return hashlib.sha256(re.sub(r"\s+", "", text).encode()).hexdigest()


def image_keys(data):
    with Image.open(io.BytesIO(data)) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    pixel_key = hashlib.sha256(struct.pack("<II", width, height) + image.tobytes()).hexdigest()
    tiny = list(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    difference = 0
    for y in range(8):
        for x in range(8):
            difference = (difference << 1) | (tiny[9 * y + x] > tiny[9 * y + x + 1])
    return {
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "pixel_sha256": pixel_key,
        "dhash64": f"{difference:016x}",
        "width": width, "height": height,
    }


def near_match(keys, candidates):
    target = int(keys["dhash64"], 16)
    ratio = keys["width"] / keys["height"]
    for other in candidates:
        if abs(ratio / (other["width"] / other["height"]) - 1) <= 0.02:
            if (target ^ int(other["dhash64"], 16)).bit_count() <= 2:
                return True
    return False


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main():
    output = ROOT / "data/processed/unimer-supervision-20261006"
    output.mkdir(parents=True, exist_ok=False)
    audit_root = ROOT / "experiments/runs/supervision-preflight-20261006"
    protocol = ROOT / "docs/research/SUPERVISION_PROTOCOL_20261006.md"
    archive_path = ROOT / "data/raw/unimer-official/UniMER-1M.zip"
    test_path = ROOT / "data/raw/unimer-official/UniMER-Test.zip"
    expected = "c2563aba157f470a9d7d2084aac58fb32607c58ce0992eb4a89d767d23be44dd"
    with archive_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError("Official training archive hash mismatch")
    with test_path.open("rb") as stream:
        test_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if test_sha != "9bf370b8cac868fee84835f40dec26c477430611253e3feb681356a8149a3a90":
        raise ValueError("Official benchmark archive hash mismatch")

    held_images = []
    blocked_labels = {label_key(text) for item in FORMULAS for text in item[:2]}
    blocked_bytes = set()
    blocked_pixels = set()
    with zipfile.ZipFile(test_path) as test:
        for split in ["spe", "cpe", "sce", "hwe"]:
            labels = test.read(f"UniMER-Test/{split}.txt").decode().splitlines()
            names = sorted(n for n in test.namelist() if n.startswith(f"UniMER-Test/{split}/") and n.endswith(".png"))
            for name in names:
                index = int(Path(name).stem)
                if not 0 <= index < len(labels):
                    raise ValueError("Held-out image/annotation index mismatch")
                key = label_key(labels[index])
                blocked_labels.add(key)
                keys = image_keys(test.read(name))
                held_images.append({"benchmark": "UniMER-Test", "archive_path": name, "reference_key": key, **keys})
                blocked_bytes.add(keys["source_sha256"])
                blocked_pixels.add(keys["pixel_sha256"])
            print("Indexed complete held-out split:", split, len(names), flush=True)

    inspected = []
    for name in [
        "experiments/runs/unimer-development-20261003/data/sources.jsonl",
        "experiments/runs/table-development-20261004/data-v2/sources.jsonl",
    ]:
        path = ROOT / name
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in rows:
            blocked_bytes.add(row["source_sha256"])
            if "reference" in row:
                blocked_labels.add(label_key(row["reference"]))
        inspected.append({"file": name, "sha256": sha256_file(path), "sources": len(rows)})
    write_rows(output / "held_out_blocklist.jsonl", held_images)

    renderer = TectonicRenderer(audit_root / "eligibility_renders")
    quotas = {"train": 128, "dev": 32}
    counts = Counter()
    reasons = Counter()
    selected = []
    selected_keys = set()
    selected_images = []
    source_inputs = {split: [] for split in quotas}
    inputs = {split: [] for split in quotas}
    references = {split: [] for split in quotas}
    manifests = {split: [] for split in quotas}
    for split in quotas:
        (output / split / "images").mkdir(parents=True)

    exclusions = (output / "selection_log.jsonl").open("w")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            labels = archive.read("UniMER-1M/train.txt").decode().splitlines()
            names = [n for n in archive.namelist() if n.startswith("UniMER-1M/images/") and n.endswith(".png")]
            inventory = {"images": len(names), "annotation_lines": len(labels), "nonblank_annotations": sum(bool(x.strip()) for x in labels)}
            names.sort(key=lambda name: hashlib.sha256(f"20261006-supervision:{name}".encode()).digest())
            for name in names:
                index = int(Path(name).stem)
                if not 0 <= index < len(labels):
                    raise ValueError("Training image/annotation index mismatch")
                gold = labels[index]
                ref_key = label_key(gold)
                split = "dev" if int(hashlib.sha256(f"20261006-split:{ref_key}".encode()).hexdigest(), 16) % 5 == 0 else "train"
                reason = None
                bad = None
                if counts[split] == quotas[split]:
                    reason = "split_quota_full"
                elif not 8 <= len(gold.strip()) <= 200:
                    reason = "outside_label_length_8_200"
                elif ref_key in blocked_labels:
                    reason = "held_out_or_inspected_formula_key"
                elif ref_key in selected_keys:
                    reason = "selected_formula_key"
                else:
                    bad = perturb(gold)
                    if bad is None:
                        reason = "no_defined_visible_symbol_perturbation"
                if reason is None:
                    corrupted, error_type = bad
                    bad_key = label_key(corrupted)
                    if bad_key in selected_keys:
                        reason = "controlled_input_formula_collision"
                    elif bad_key in blocked_labels:
                        reason = "controlled_input_held_out_formula_collision"
                if reason is None:
                    data = archive.read(name)  # ZipFile also checks the member CRC.
                    keys = image_keys(data)
                    if keys["source_sha256"] in blocked_bytes or keys["pixel_sha256"] in blocked_pixels:
                        reason = "held_out_or_selected_exact_image"
                    elif near_match(keys, held_images):
                        reason = "held_out_near_image_proxy"
                    elif near_match(keys, selected_images):
                        reason = "selected_near_image_proxy"
                if reason is None:
                    try:
                        correct_render = renderer.render(Observation("eligibility", "formula", "", gold))
                        wrong_render = renderer.render(Observation("eligibility", "formula", "", corrupted))
                        different = pixel_signature(Path(correct_render.path)) != pixel_signature(Path(wrong_render.path))
                        if not different:
                            reason = "no_exact_render_change"
                    except (ValueError, FileNotFoundError, TimeoutError) as error:
                        reason = "render_failure:" + type(error).__name__
                entry = {"archive_path": name, "annotation_index": index, "split": split, "reason": reason or "selected"}
                exclusions.write(json.dumps(entry) + "\n")
                exclusions.flush()
                if reason is not None:
                    reasons[reason] += 1
                    continue
                family = f"s{len(selected)+1:04}"
                image_relative = f"images/{family}.png"
                (output / split / image_relative).write_bytes(data)
                counts[split] += 1
                selected_keys.update([ref_key, bad_key])
                blocked_bytes.add(keys["source_sha256"])
                blocked_pixels.add(keys["pixel_sha256"])
                selected_images.append(keys)
                selected.append({**entry, "family_id": family, "source_image": image_relative, "reference": gold, "reference_key": ref_key, "controlled_input_key": bad_key, "error_type": error_type, **keys})
                source_inputs[split].append({"family_id": family, "source_image": image_relative, "source_sha256": keys["source_sha256"]})
                for variant, prediction in [("preservation", gold), ("controlled_error", corrupted)]:
                    sample = family + ("-keep" if variant == "preservation" else "-repair")
                    inputs[split].append({"sample_id": sample, "family_id": family, "source_image": image_relative, "source_sha256": keys["source_sha256"], "prediction": prediction})
                    references[split].append({"sample_id": sample, "family_id": family, "reference": gold, "variant": variant, "source_kind": "UniMER-1M", "error_type": None if variant == "preservation" else error_type})
                    manifests[split].append({"sample_id": sample, "page_id": "unimer1m-image:" + str(index), "benchmark": "UniMER-1M", "modality": "formula", "split": split, "test_only": False, "source_image": f"{split}/{image_relative}", "prediction": prediction, "reference": gold})
                if sum(counts.values()) % 16 == 0:
                    print("Frozen sources:", dict(counts), flush=True)
                if all(counts[split] == quotas[split] for split in quotas):
                    break
            if any(counts[split] != quotas[split] for split in quotas):
                raise ValueError("Not enough eligible independent training-archive sources")
    finally:
        exclusions.close()

    write_rows(output / "sources.jsonl", selected)
    for split in quotas:
        write_rows(output / split / "source_inputs.jsonl", source_inputs[split])
        write_rows(output / split / "inputs.jsonl", inputs[split])
        write_rows(output / split / "references.jsonl", references[split])
        write_rows(output / f"{split}.jsonl", manifests[split])
    paths = [p for p in output.rglob("*") if p.is_file()]
    hashes = {str(path.relative_to(output)): sha256_file(path) for path in sorted(paths)}
    receipt = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "wanderkid/UniMER_Dataset", "revision": "2343ddd963290469da36ca83e3a56c66e068add9",
        "training_archive_sha256": expected, "test_archive_sha256": test_sha,
        "available_archive_inventory": inventory,
        "protocol_sha256": sha256_file(protocol), "preparation_script_sha256": sha256_file(Path(__file__)),
        "train_sources": counts["train"], "dev_sources": counts["dev"],
        "train_cases": len(inputs["train"]), "dev_cases": len(inputs["dev"]),
        "benchmark_images_blocked": len(held_images), "unique_held_out_formula_keys": len(blocked_labels),
        "previously_inspected_sources": inspected, "exclusion_counts": dict(reasons),
        "model_outputs_used_for_selection": False, "new_model_calls": 0, "weights_changed": False,
        "target_provenance": "released annotations; not teacher trajectories",
        "original_document_grouping": "unavailable; archive image identity and formula/image grouping only",
        "model_pretraining_overlap": "unknown",
        "near_duplicate_test": "64-bit dHash <=2 and image aspect ratio within 2%; screening proxy only",
        "file_sha256": hashes,
    }
    (output / "dataset.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (audit_root / "dataset_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ["train_sources", "dev_sources", "train_cases", "dev_cases", "benchmark_images_blocked", "exclusion_counts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
