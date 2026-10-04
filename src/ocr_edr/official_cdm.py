"""Isolated loading of unmodified, pinned official OmniDocBench CDM code."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
from pathlib import Path
from types import ModuleType

PREFIX = "src/metrics/cdm/"
ENTRYPOINT = PREFIX + "cdm.py"


def cdm_source_paths(inventory: dict) -> list[str]:
    paths = sorted(
        entry["path"]
        for entry in inventory["tree"]
        if entry["type"] == "blob"
        and entry["path"].startswith(PREFIX)
        and entry["path"].endswith(".py")
    )
    if ENTRYPOINT not in paths or any(".." in Path(p).parts for p in paths):
        raise ValueError("Official CDM inventory has no valid entrypoint")
    return paths


def verify_official_cdm_sources(source_root: Path, expected_revision: str) -> dict:
    """Verify every Python file that can enter the isolated upstream package."""
    root = source_root.resolve()
    inventory = json.loads((root / "inventory.json").read_text())
    revision = json.loads((root / "revision.json").read_text())["revision"]
    if (
        not re.fullmatch(r"[a-f0-9]{40}", expected_revision)
        or revision != expected_revision
        or inventory["sha"] != expected_revision
        or inventory.get("truncated")
    ):
        raise ValueError("Official CDM revision/inventory mismatch")
    paths = cdm_source_paths(inventory)
    actual = {p.relative_to(root).as_posix() for p in (root / PREFIX).rglob("*.py")}
    if actual != set(paths):
        raise ValueError("Official CDM Python file inventory mismatch")
    blobs = {e["path"]: e["sha"] for e in inventory["tree"] if e["type"] == "blob"}
    hashes = {}
    for relative in paths:
        content = (root / relative).read_bytes()
        blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if blob != blobs[relative]:
            raise ValueError(f"Modified official CDM source: {relative}")
        hashes[relative] = {
            "git_blob_sha1": blob,
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    return {"repository": "opendatalab/OmniDocBench", "revision": revision, "files": hashes}


def load_official_cdm(source_root: Path, expected_revision: str):
    """Return the verified core callable, avoiding unrelated upstream imports."""
    root = source_root.resolve()
    receipt = verify_official_cdm_sources(root, expected_revision)
    identity = hashlib.sha256(
        (str(root) + json.dumps(receipt, sort_keys=True)).encode()
    ).hexdigest()
    namespace = "ocr_edr_upstream_cdm_" + identity
    if namespace not in sys.modules:
        package = ModuleType(namespace)
        package.__path__ = [str(root / PREFIX)]
        sys.modules[namespace] = package
    module = importlib.import_module(namespace + ".cdm")
    return module.cdm_metrics, receipt
