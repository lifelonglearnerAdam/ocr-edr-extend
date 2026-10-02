#!/usr/bin/env python
"""Show data acquisition instructions; automatic downloads are not implemented."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench", default="omnidocbench", choices=["omnidocbench", "ocerrbench", "unimer"]
    )
    parser.add_argument("--show-instructions", action="store_true")
    args = parser.parse_args()
    if not args.show_instructions:
        parser.error(
            "No automatic downloader is configured. Use --show-instructions or import existing data with prepare_note_eval.py."
        )
    print(f"Data acquisition: {args.bench}; see data/README.md for sources and permissions.")
    if args.bench == "omnidocbench":
        print("Official source: https://github.com/opendatalab/OmniDocBench")
        print("Existing Note diagnostics can be imported with scripts/prepare_note_eval.py.")
    else:
        print("Confirm the official release and license before adding this benchmark.")


if __name__ == "__main__":
    main()
