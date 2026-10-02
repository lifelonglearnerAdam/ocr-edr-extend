#!/usr/bin/env python3
"""Import existing Note artifacts; no download, inference, or training."""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.data import prepare_note


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--remote-root", default=Path("/data/hzhang"), type=Path)
    parser.add_argument("--output", default=Path("data/raw/note_eval"), type=Path)
    parser.add_argument(
        "--public-summary", type=Path, help="Optional aggregate-only report for Git"
    )
    args = parser.parse_args()
    summary = prepare_note(args.source_root, args.output, args.remote_root)
    if args.public_summary:
        args.public_summary.parent.mkdir(parents=True, exist_ok=True)
        args.public_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ["note_pages", "diagnostic_elements", "official_matches", "alignment"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
