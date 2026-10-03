#!/usr/bin/env python3
"""Print official acquisition guidance; automatic downloads are not configured."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench", choices=["omnidocbench", "ocrerrbench", "unimer"], default="omnidocbench"
    )
    parser.add_argument("--show-instructions", action="store_true")
    args = parser.parse_args()
    if not args.show_instructions:
        parser.error(
            "No automatic downloader is configured. Use --show-instructions and data/README.md."
        )
    print(f"Prepare this project's {args.bench} manifest following docs/DATA_PROTOCOL.md.")
    if args.bench == "omnidocbench":
        print("Official source: https://github.com/opendatalab/OmniDocBench")
    else:
        print("Confirm the official release and license before preparing data.")


if __name__ == "__main__":
    main()
