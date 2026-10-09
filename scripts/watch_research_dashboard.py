#!/usr/bin/env python3
"""Update verified public snapshots; explicitly publish only the two generated files."""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import _bootstrap  # noqa: F401

from ocr_edr.sft import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("experiments/runs/table-nf4-screen-recovery-20261009")
    )
    parser.add_argument(
        "--prompt-results",
        type=Path,
        default=Path("experiments/runs/table-prompt-ablation-20261009"),
    )
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.interval < 30:
        parser.error("Use an interval of at least 30 seconds")
    project = Path(__file__).resolve().parents[1]
    files = ["docs/site/index.html", "docs/site/research-data.json"]
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "driver_sha256": sha256(Path(__file__)),
        "publication_requested": args.publish,
        "successful_updates": 0,
    }

    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=project, text=True).strip()

    try:
        if args.publish:
            remote = git("config", "--get", "remote.origin.url")
            repository = (
                remote.split(":", 1)[1]
                if remote.startswith("git@github.com:")
                else urlsplit(remote).path.lstrip("/")
            )
            if (
                repository.removesuffix(".git") != "lifelonglearnerAdam/ocr-edr-extend"
                or git("branch", "--show-current") != "feat/image-role-research"
            ):
                raise ValueError(
                    "Publication requires the authorized repository and research branch"
                )
        while True:
            before = {name: sha256(project / name) for name in files}
            subprocess.run(
                [
                    sys.executable,
                    str(project / "scripts/update_research_dashboard.py"),
                    "--results",
                    str(args.results),
                    "--prompt-results",
                    str(args.prompt_results),
                ],
                cwd=project,
                check=True,
            )
            after = {name: sha256(project / name) for name in files}
            changed = before != after
            if changed and args.publish:
                staged = set(filter(None, git("diff", "--cached", "--name-only").splitlines()))
                if staged - set(files):
                    raise ValueError(
                        "Other staged work exists; refuse to include it in a dashboard commit"
                    )
                subprocess.run(["git", "add", "--", *files], cwd=project, check=True)
                subprocess.run(
                    ["git", "commit", "-m", "Refresh shared research evidence snapshot"],
                    cwd=project,
                    check=True,
                )
                subprocess.run(
                    ["git", "push", "origin", "HEAD:refs/heads/feat/image-role-research"],
                    cwd=project,
                    check=True,
                )
                state["successful_updates"] += 1
                state["last_published_commit"] = git("rev-parse", "HEAD")
            state.update(
                last_checked_at=datetime.now(timezone.utc).isoformat(),
                public_files_sha256=after,
                semantic_or_template_change=changed,
            )
            args.receipt.write_text(json.dumps(state, indent=2) + "\n")
            if args.once:
                state["status"] = "completed"
                break
            time.sleep(args.interval)
    except Exception as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error)[:700])
        raise
    finally:
        args.receipt.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
