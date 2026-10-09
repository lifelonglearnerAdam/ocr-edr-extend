#!/usr/bin/env python3
"""Build an offline/shareable research page from the sanitized evidence snapshot."""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.research_dashboard import render_dashboard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=Path("docs/site"))
    args = parser.parse_args()
    data = json.loads((args.site / "research-data.json").read_text())
    template = (args.site / "template.html").read_text()
    assets = {p.name: p.read_bytes() for p in (args.site / "assets").glob("*.png")}
    output = args.site / "index.html"
    explainer = (
        (args.site / "beginner.fragment.html").read_text()
        if (args.site / "beginner.fragment.html").exists()
        else None
    )
    if (args.site / "briefing.fragment.html").exists():
        explainer = (explainer or "") + (args.site / "briefing.fragment.html").read_text()
    output.write_text(render_dashboard(data, template, assets, explainer))
    print(
        json.dumps(
            {
                "output": str(output),
                "updated_at": data["updated_at"],
                "bytes": output.stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    main()
