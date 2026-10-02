#!/usr/bin/env python3
"""Run the official OmniDocBench evaluator in an isolated experiment directory."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from ocr_edr.metrics import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline", help="Configured OCR system, unless --pred-dir is supplied")
    parser.add_argument("--pred-dir", type=Path)
    parser.add_argument("--gt", type=Path)
    parser.add_argument("--official-repo", type=Path)
    parser.add_argument("--python", dest="python_executable")
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path, default=Path("experiments/runs/official"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    official = cfg.get("official", {})
    repo = (args.official_repo or Path(official.get("repo", "../omnidocbench-eval"))).resolve()
    entrypoint = repo / "pdf_validation.py"
    if not entrypoint.is_file():
        parser.error(f"Official evaluator entrypoint not found: {entrypoint}")
    baselines = [b for b in cfg["baselines"] if b["name"] == args.baseline]
    if args.pred_dir is None and len(baselines) != 1:
        parser.error(f"Unknown or duplicate baseline: {args.baseline}")
    pred = (args.pred_dir or Path(baselines[0]["pred_dir"])).resolve()
    gt = (
        args.gt
        or Path(cfg["benchmark"]["root"]) / cfg["benchmark"].get("annotation", "OmniDocBench.json")
    ).resolve()
    if not pred.is_dir() or not gt.is_file():
        parser.error("Prediction directory and annotation file must exist")
    run_id = (
        args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", run_id):
        parser.error("Run ID must use letters, numbers, underscores or hyphens")
    run_dir = args.output_root.resolve() / run_id
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=False)
    native = {
        "end2end_eval": {
            "metrics": {
                "text_block": {"metric": ["Edit_dist"]},
                "display_formula": {
                    "metric": ["Edit_dist", "CDM"],
                    "cdm_workers": official.get("workers", 4),
                },
                "table": {
                    "metric": ["TEDS", "Edit_dist"],
                    "teds_workers": official.get("workers", 4),
                },
                "reading_order": {"metric": ["Edit_dist"]},
            },
            "dataset": {
                "dataset_name": "end2end_dataset",
                "ground_truth": {"data_path": str(gt)},
                "prediction": {"data_path": str(pred)},
                "match_method": official.get("match_method", "quick_match"),
                "match_workers": official.get("workers", 4),
            },
        },
    }
    if cfg["benchmark"].get("filter"):
        native["end2end_eval"]["dataset"]["filter"] = cfg["benchmark"]["filter"]
    native_path = run_dir / "native.yaml"
    native_path.write_text(yaml.safe_dump(native, sort_keys=False), encoding="utf-8")
    command = [
        args.python_executable or official.get("python", sys.executable),
        str(entrypoint),
        "--config",
        str(native_path),
    ]
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    metadata = {
        "command": command,
        "cwd": str(run_dir),
        "dry_run": args.dry_run,
        "official_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "config_sha256": file_sha256(args.config),
        "native_config_sha256": file_sha256(native_path),
        "annotation_sha256": file_sha256(gt),
        "baseline": args.baseline or "custom",
    }
    metadata_path = run_dir / "run.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    if not args.dry_run:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        completed = subprocess.run(command, cwd=run_dir, env=env)
        metadata["returncode"] = completed.returncode
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
