#!/usr/bin/env python3
"""Freeze syntax-only outcomes; this command never reads reference annotations."""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.formula_pilot import validate_pilot_inputs
from ocr_edr.sft import sha256
from ocr_edr.sft_screen import adapt_screen_calls
from ocr_edr.tex import TectonicRenderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    inputs = [json.loads(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    validate_pilot_inputs(inputs, args.inputs.parent)
    calls = {}
    source_hashes = {}
    for path in args.run:
        receipt = json.loads((path / "run.json").read_text())
        if (
            receipt["status"] != "completed"
            or receipt["reference_access"] != "none"
            or receipt["completed_calls"] != len(inputs)
        ):
            raise ValueError("Inference run is incomplete or violates the reference boundary")
        call_path = path / "calls.jsonl"
        if sha256(call_path) != receipt["calls_sha256"] or receipt["condition"] in calls:
            raise ValueError("Call hashes or unique condition mismatch")
        if receipt["config"]["inference"]["max_new_tokens"] != args.max_new_tokens:
            raise ValueError("Inference cap changed")
        calls[receipt["condition"]] = [
            json.loads(line) for line in call_path.read_text().splitlines()
        ]
        source_hashes[str(path)] = receipt["calls_sha256"]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    renderer = TectonicRenderer(root / "renders")
    outcomes = adapt_screen_calls(
        inputs, calls, renderer=renderer, max_new_tokens=args.max_new_tokens
    )
    predictions = root / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in outcomes))
    (root / "run.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "reference_access": "none",
                "new_model_calls": 0,
                "adapter": "strict full latex response; declared cap rejection; syntax/render rollback only",
                "input_sha256": sha256(args.inputs),
                "call_sha256": source_hashes,
                "prediction_sha256": sha256(predictions),
                "records": len(outcomes),
                "cases": len(inputs),
                "source_sha256": {
                    str(p): sha256(p)
                    for p in [
                        Path(__file__),
                        Path("src/ocr_edr/sft.py"),
                        Path("src/ocr_edr/sft_screen.py"),
                    ]
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Frozen {len(outcomes)} outcomes for {len(inputs)} fixed inputs, without reference access."
    )


if __name__ == "__main__":
    main()
