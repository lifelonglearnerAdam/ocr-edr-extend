"""Import the existing Note diagnostic set; preserve evaluation-only provenance."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .metrics import MODALITIES, compare_results, file_sha256, read_json, result_path, sample_key


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def formula_body(text: str) -> str:
    text = text.strip()
    for left, right in [("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$")]:
        if text.startswith(left) and text.endswith(right):
            return text[len(left) : -len(right)].strip()
    return text


def assert_trainable(records: list[dict]) -> None:
    """Metadata guard only; training also needs page/image hash and near-duplicate checks."""
    for row in records:
        if row.get("split") != "train" or row.get("test_only") is not False:
            raise ValueError(
                "Training requires explicit split=train and test_only=false on every record"
            )


def prepare_note(root: Path, output: Path, remote_root: Path = Path("/data/hzhang")) -> dict:
    root = root.resolve()
    annotation_path = root / "OmniDocBench_note/OmniDocBench_note.json"
    pages = read_json(annotation_path)
    page_map = {Path(p["page_info"]["image_path"]).name: p for p in pages}
    if len(page_map) != len(pages):
        raise ValueError("Duplicate Note page identity")
    if any(p["page_info"]["page_attribute"]["data_source"] != "note" for p in pages):
        raise ValueError("Expected a Note-only annotation snapshot")
    results_dir = root / "baselines/monkeyocrv2_b_note/omnidocbench_results_cdm"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    matches = []
    inputs = [
        {"file": str(annotation_path.relative_to(root)), "sha256": file_sha256(annotation_path)}
    ]
    alignment = Counter()
    image_hashes: dict[str, str] = {}

    def checked_path(path: Path) -> Path:
        path = path.resolve()
        path.relative_to(root)  # Reject paths escaping the supplied mirror/server root.
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def image_hash(path: Path) -> str:
        if str(path) not in image_hashes:
            image_hashes[str(path)] = file_sha256(path)
        return image_hashes[str(path)]

    for page_id in page_map:
        image_hash(checked_path(root / "OmniDocBench_note/images" / page_id))

    for modality, (category, _) in MODALITIES.items():
        manifest_path = (
            root / f"diagnostics/OmniDocBench_note_test_only/{modality}/manifest_test_only.jsonl"
        )
        manifest = read_jsonl(manifest_path)
        official_path = result_path(results_dir, f"{category}_result")
        official = read_json(official_path)
        if any(row["img_id"] not in page_map for row in official):
            raise ValueError("Official match references a page outside the Note snapshot")
        for path in (manifest_path, official_path):
            inputs.append({"file": str(path.relative_to(root)), "sha256": file_sha256(path)})
        for row in official:
            matches.append(
                {
                    "sample_id": f"{modality}:{sample_key(row)}",
                    "modality": modality,
                    "split": "test",
                    "test_only": True,
                    "official_match": row,
                }
            )
        for row in manifest:
            if row.get("split") != "test" or row.get("test_only") is not True:
                raise ValueError("Note diagnostic import requires explicit test-only provenance")
            page_id = Path(row["image_path"]).name
            page = page_map[page_id]
            annos = [a for a in page["layout_dets"] if a["anno_id"] == row["anno_id"]]
            if len(annos) != 1:
                raise ValueError("Missing or ambiguous annotation identity")
            order = annos[0].get("order")
            candidates = [
                r for r in official if r["img_id"] == page_id and order in r["gt_position"]
            ]
            reference = row.get("raw_latex", row["ground_truth"])
            one_to_one = len(candidates) == 1 and candidates[0]["gt_position"] == [order]
            if one_to_one:
                a, b = reference.strip(), candidates[0]["gt"].strip()
                if modality == "formula":
                    a, b = formula_body(a), formula_body(b)
                one_to_one = a == b
            status = (
                "one_to_one" if one_to_one else ("split_or_merged" if candidates else "unmatched")
            )
            alignment[f"{modality}:{status}"] += 1
            crop = Path(row["crop"])
            if crop.is_absolute():
                crop = root / crop.relative_to(remote_root)
            else:
                crop = root / crop
            crop = checked_path(crop)
            source_page = checked_path(root / "OmniDocBench_note/images" / page_id)
            records.append(
                {
                    "sample_id": f"{modality}:{page_id}:{row['anno_id']}",
                    "modality": modality,
                    "page_id": page_id,
                    "annotation_id": row["anno_id"],
                    "split": "test",
                    "test_only": True,
                    "bbox": row["bbox"],
                    "source_image": str(crop),
                    "source_image_sha256": image_hash(crop),
                    "source_page_sha256": image_hash(source_page),
                    "reference": reference,
                    "alignment": status,
                    "initial_prediction": candidates[0]["pred"] if one_to_one else None,
                    "official_match_ids": [f"{modality}:{sample_key(r)}" for r in candidates],
                }
            )
    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Duplicate diagnostic element identity")
    write_jsonl(output / "elements_test_only.jsonl", records)
    write_jsonl(output / "official_matches_test_only.jsonl", matches)
    # Block every page and crop, including pages without formula/table elements.
    write_jsonl(
        output / "image_sha256_blocklist.jsonl",
        [{"sha256": value, "test_only": True} for value in sorted(set(image_hashes.values()))],
    )
    summary = {
        "schema_version": 1,
        "split": "test",
        "test_only": True,
        "note_pages": len(pages),
        "diagnostic_elements": dict(Counter(r["modality"] for r in records)),
        "official_matches": dict(Counter(r["modality"] for r in matches)),
        "alignment": dict(sorted(alignment.items())),
        "blocked_image_hashes": len(set(image_hashes.values())),
        "source_artifacts": inputs,
        "baseline": compare_results(results_dir),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
