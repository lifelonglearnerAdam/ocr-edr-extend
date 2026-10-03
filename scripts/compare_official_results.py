#!/usr/bin/env python3
"""Compare official artifacts; fail if reference matches differ."""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.metrics import compare_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--before-prefix")
    parser.add_argument("--after-prefix")
    parser.add_argument("--good-threshold", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_results(
        args.before, args.after, args.before_prefix, args.after_prefix, args.good_threshold
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["modalities"], indent=2))


if __name__ == "__main__":
    main()
