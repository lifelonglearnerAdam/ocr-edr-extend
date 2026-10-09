#!/usr/bin/env python3
"""Fetch only pinned official table code/demo metadata; no model or page-image download."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.official_tables import verify_official_table_sources

DEFAULT_REVISION = "f133a71e9e91c3621c7ce8994200a7b394a06eb3"
FILES = [
    "LICENSE",
    "src/metrics/table_metric.py",
    "src/core/preprocess/data_preprocess.py",
    "src/core/preprocess/table_postprocess.py",
    "src/core/preprocess/table_utils.py",
    "src/core/preprocess/text_postprocess.py",
    "demo_data/omnidocbench_demo/OmniDocBench_demo.json",
    "result/end2end_quick_match_table_result.json",
]


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ocr-edr-research"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    if not re.fullmatch("[a-f0-9]{40}", args.revision):
        raise ValueError("An exact Git commit hash is required")
    root = args.output.resolve()
    revision_path = root / "revision.json"
    if (
        revision_path.exists()
        and json.loads(revision_path.read_text())["revision"] != args.revision
    ):
        raise ValueError("Output already belongs to a different official revision")
    inventory_path = root / "inventory.json"
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
    else:
        inventory = json.loads(
            fetch(
                f"https://api.github.com/repos/opendatalab/OmniDocBench/git/trees/{args.revision}?recursive=1"
            )
        )
    if inventory["sha"] != args.revision or inventory.get("truncated"):
        raise ValueError("Official Git inventory is incomplete or has the wrong revision")
    blobs = {r["path"]: r["sha"] for r in inventory["tree"] if r["type"] == "blob"}
    root.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
    revision_path.write_text(
        json.dumps({"repo": "opendatalab/OmniDocBench", "revision": args.revision}, indent=2) + "\n"
    )
    for relative in FILES:
        path = root / relative
        content = (
            path.read_bytes()
            if path.exists()
            else fetch(
                f"https://raw.githubusercontent.com/opendatalab/OmniDocBench/{args.revision}/{relative}"
            )
        )
        git_sha = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if git_sha != blobs.get(relative):
            raise ValueError(
                f"Downloaded or cached file differs from official Git blob: {relative}"
            )
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        print(relative, flush=True)
    receipt = verify_official_table_sources(root, args.revision)
    (root / "table_source_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
