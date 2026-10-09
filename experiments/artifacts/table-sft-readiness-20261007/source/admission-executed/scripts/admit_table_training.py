#!/usr/bin/env python3
"""Create a separate admitted training version; leave all four original roles frozen."""

import argparse
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from ocr_edr.sft import sha256
from ocr_edr.table_training import admit_training_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--review-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = yaml.safe_load(args.policy.read_text())
    root = args.dataset.resolve()
    summary = json.loads((root / "supervision/summary.json").read_text())
    train_path = root / "supervision/train-sft.jsonl"
    dev_path = root / "supervision/model_dev-sft.jsonl"
    for path, role in [(train_path, "train"), (dev_path, "model_dev")]:
        if sha256(path) != summary["roles"][role]["sha256"]:
            raise ValueError("Frozen supervision identity mismatch")
    rows = [json.loads(line) for line in train_path.read_text().splitlines()]
    if (
        len(rows) != policy["expected_original_records"]
        or len({r["document_id"] for r in rows}) != policy["expected_original_documents"]
    ):
        raise ValueError("Original training coverage changed")
    reviews = [json.loads(line) for line in args.review_ledger.read_text().splitlines()]
    reviewed = {r["family_id"] for r in reviews if r["role"] == "train"}
    if set(policy["excluded_families"]) - reviewed:
        raise ValueError("Declared exclusion has no source-review ledger entry")
    admitted, excluded = admit_training_records(rows, policy["excluded_families"])
    if (
        len(admitted) != policy["expected_admitted_records"]
        or len({r["document_id"] for r in admitted}) != policy["expected_admitted_documents"]
    ):
        raise ValueError("Admitted coverage differs from frozen policy")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "train-sft.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in admitted)
    )
    (output / "exclusions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in excluded))
    (output / "model_dev-sft.jsonl").write_bytes(dev_path.read_bytes())
    receipt = {
        "status": "admitted_as_published_weak_supervision",
        "dataset": policy["dataset"],
        "policy_sha256": sha256(args.policy),
        "review_ledger_sha256": sha256(args.review_ledger),
        "original_train_sha256": sha256(train_path),
        "original_dev_sha256": sha256(dev_path),
        "records": len(admitted),
        "documents": len({r["document_id"] for r in admitted}),
        "variants": dict(Counter(r["variant"] for r in admitted)),
        "excluded_records": len(excluded),
        "excluded_families": policy["excluded_families"],
        "reviewed_training_sources": len(reviewed),
        "all_targets_independently_verified": False,
        "unreviewed_label_noise_unresolved": True,
        "reference_relabeling": False,
        "model_outputs_used_for_admission": False,
        "optimizer_steps": 0,
        "calibration_locked_targets_loaded": False,
        "file_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
        "source_sha256": {
            str(p): sha256(p) for p in [Path(__file__), Path("src/ocr_edr/table_training.py")]
        },
    }
    (output / "admission.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
