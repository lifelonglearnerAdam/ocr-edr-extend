#!/usr/bin/env python3
"""Fetch a pinned, hash-verified CDM source subtree without models or data images."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401
from fetch_official_table_sources import DEFAULT_REVISION, fetch

from ocr_edr.official_cdm import cdm_source_paths, verify_official_cdm_sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{40}", args.revision):
        raise ValueError("An exact Git commit hash is required")
    root = args.output.resolve()
    revision_path = root / "revision.json"
    if (
        revision_path.exists()
        and json.loads(revision_path.read_text())["revision"] != args.revision
    ):
        raise ValueError("Output belongs to a different official revision")
    inventory_path = root / "inventory.json"
    inventory = (
        json.loads(inventory_path.read_text())
        if inventory_path.exists()
        else json.loads(
            fetch(
                f"https://api.github.com/repos/opendatalab/OmniDocBench/git/trees/{args.revision}?recursive=1"
            )
        )
    )
    if inventory["sha"] != args.revision or inventory.get("truncated"):
        raise ValueError("Official Git inventory is incomplete or has the wrong revision")
    paths = ["LICENSE"] + cdm_source_paths(inventory)
    blobs = {e["path"]: e["sha"] for e in inventory["tree"] if e["type"] == "blob"}
    root.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
    revision_path.write_text(
        json.dumps({"repo": "opendatalab/OmniDocBench", "revision": args.revision}, indent=2) + "\n"
    )
    for relative in paths:
        path = root / relative
        content = (
            path.read_bytes()
            if path.exists()
            else fetch(
                f"https://raw.githubusercontent.com/opendatalab/OmniDocBench/{args.revision}/{relative}"
            )
        )
        blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if blob != blobs[relative]:
            raise ValueError(f"Downloaded or existing bytes differ from Git blob: {relative}")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        print(relative, flush=True)
    receipt = verify_official_cdm_sources(root, args.revision)
    (root / "cdm_source_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Verified {len(receipt['files'])} upstream Python files")


if __name__ == "__main__":
    main()
