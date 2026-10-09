"""Load pinned upstream TEDS and HTML normalization without modifying their code."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_official_teds(source_root: Path):
    path = source_root / "src/metrics/table_metric.py"
    spec = importlib.util.spec_from_file_location("ocr_edr_upstream_teds", path)
    if spec is None or spec.loader is None:
        raise ValueError("Official TEDS source unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TEDS


def load_official_table_normalizer(source_root: Path):
    # Avoid the upstream package's unrelated formula/CDM imports; source files are unchanged.
    for name, path in [
        ("src", source_root / "src"),
        ("src.core", source_root / "src/core"),
        ("src.core.preprocess", source_root / "src/core/preprocess"),
    ]:
        module = sys.modules.get(name)
        if module is not None and list(getattr(module, "__path__", [])) != [str(path)]:
            raise ValueError("Conflicting upstream package namespace")
        if module is None:
            module = ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
    from src.core.preprocess.data_preprocess import normalized_html_table
    from src.core.preprocess.table_postprocess import table_content_post_process

    def normalize(markup: str) -> str:
        return table_content_post_process(normalized_html_table(markup))

    return normalize


def verify_official_table_sources(source_root: Path, expected_revision: str) -> dict:
    """Check downloaded bytes against the pinned Git tree's blob IDs before use."""
    import hashlib
    import json

    from .formula_pilot import sha256_file

    root = source_root.resolve()
    revision = json.loads((root / "revision.json").read_text())["revision"]
    inventory = json.loads((root / "inventory.json").read_text())
    if (
        revision != expected_revision
        or inventory["sha"] != expected_revision
        or inventory.get("truncated")
    ):
        raise ValueError("Official source revision/inventory mismatch")
    paths = [
        "src/metrics/table_metric.py",
        "src/core/preprocess/data_preprocess.py",
        "src/core/preprocess/table_postprocess.py",
        "src/core/preprocess/table_utils.py",
        "src/core/preprocess/text_postprocess.py",
    ]
    blobs = {entry["path"]: entry["sha"] for entry in inventory["tree"] if entry["type"] == "blob"}
    hashes = {}
    for relative in paths:
        path = root / relative
        content = path.read_bytes()
        git_sha = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if git_sha != blobs.get(relative):
            raise ValueError(f"Modified or missing official source: {relative}")
        hashes[relative] = {"git_blob_sha1": git_sha, "sha256": sha256_file(path)}
    return {"repository": "opendatalab/OmniDocBench", "revision": revision, "files": hashes}
