#!/usr/bin/env python3
"""Validate this project's formula/table region manifest without a GPU."""

import argparse
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.data import assert_trainable, load_manifest
from ocr_edr.metrics import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--purpose", choices=["evaluation", "training"], default="evaluation")
    parser.add_argument("--held-out", type=Path)
    parser.add_argument("--held-out-root", type=Path)
    args = parser.parse_args()
    records = load_manifest(args.manifest, args.source_root)
    if args.purpose == "training":
        if args.held_out is None:
            parser.error("Training validation requires --held-out")
        assert_trainable(records, load_manifest(args.held_out, args.held_out_root))
    elif any(row["split"] == "train" for row in records):
        parser.error("Evaluation input contains training records")
    print(
        json.dumps(
            {
                "samples": len(records),
                "modalities": dict(Counter(r["modality"] for r in records)),
                "benchmarks": dict(Counter(r["benchmark"] for r in records)),
                "splits": dict(Counter(r["split"] for r in records)),
                "manifest_sha256": file_sha256(args.manifest),
                "purpose": args.purpose,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
