#!/usr/bin/env python3
"""Reference-free table proposals with fixed image-role and bounded-edit controls."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr.formula_pilot import sha256_file, validate_pilot_inputs
from ocr_edr.loop import Observation
from ocr_edr.qwen import QwenFormulaProposer
from ocr_edr.table_pilot import HTMLTableRenderer, apply_table_action, extract_table, table_cell_map

MODES = ["source_only_rewrite", "source_first_rewrite", "source_last_rewrite", "source_only_patch"]
PATCH_MODES = {"source_only_patch", "source_only_patch_indexed"}
ALL_MODES = [*MODES, "source_only_patch_indexed"]


def message(source: Path, initial: str, rendered: Path | None, mode: str):
    if mode not in ALL_MODES:
        raise ValueError("Unknown table evidence mode")
    roles = "Image 1 is the source table."
    paths = [source]
    if mode in {"source_first_rewrite", "source_last_rewrite"}:
        if rendered is None:
            raise ValueError("Two-image arm requires initial rendering")
        if mode == "source_first_rewrite":
            paths = [source, rendered]
            roles = "Image 1 is the source table. Image 2 is a rendering of the current OCR table."
        else:
            paths = [rendered, source]
            roles = "Image 1 is a rendering of the current OCR table. Image 2 is the source table."
    prompt = (
        roles + "\nCompare the source with the current OCR table. Correct only differences visible "
        "in the source. Preserve correct cell text, row/column order, rowspan and colspan. "
        "If the table is already correct, preserve it.\nCurrent OCR table:\n" + initial + "\n"
    )
    if mode == "source_only_patch_indexed":
        prompt += (
            "Cell address map computed from the current HTML (zero-based row/cell indices):\n"
            "<cell_map>\n"
            + json.dumps(table_cell_map(initial), ensure_ascii=False, separators=(",", ":"))
            + "\n</cell_map>\n"
        )
    if mode in PATCH_MODES:
        prompt += (
            'Return exactly one JSON object using one of these actions: {"action":"stop"}; '
            '{"action":"replace_cell","row":0,"cell":0,"text":"correct plain cell text"}; '
            '{"action":"set_span","row":0,"cell":0,"rowspan":1,"colspan":1}; '
            '{"action":"delete_row","row":0}. Row and cell indices are zero-based in the current '
            "HTML document order. Choose the appropriate indices yourself. Change at most one "
            "cell or row. Do not return a rewritten table or surrounding explanation."
        )
    else:
        prompt += "Return only the complete corrected HTML table, <table>...</table>, without explanation."
    return (
        paths,
        [
            {
                "role": "user",
                "content": [*[{"type": "image"} for _ in paths], {"type": "text", "text": prompt}],
            }
        ],
        prompt,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=ALL_MODES, default=MODES)
    parser.add_argument("--min-pixels", type=int, default=100352)
    parser.add_argument("--max-pixels", type=int, default=200704)
    args = parser.parse_args(argv)
    if len(set(args.modes)) != len(args.modes):
        parser.error("Each proposal mode must be selected only once")
    if args.min_pixels <= 0 or args.max_pixels < args.min_pixels:
        parser.error("Positive ordered image pixel limits required")
    return args


def main() -> None:
    args = parse_args()
    import torch

    if args.output.exists():
        raise ValueError("Use a new output directory")
    inputs = list(map(json.loads, args.inputs.read_text().splitlines()))
    validate_pilot_inputs(inputs, args.inputs.parent)
    torch.set_num_threads(8)
    torch.manual_seed(20261004)
    args.output.mkdir(parents=True)
    renderer = HTMLTableRenderer(args.output / "renders")
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": "Qwen/Qwen2-VL-2B-Instruct",
        "model_revision": args.model.name,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "source_hashes": {
            p: sha256_file(Path(p))
            for p in [
                "scripts/run_table_pilot.py",
                "src/ocr_edr/qwen.py",
                "src/ocr_edr/table_pilot.py",
            ]
        },
        "input_sha256": sha256_file(args.inputs),
        "cases": len(inputs),
        "reference_access": "none",
        "device": "cpu",
        "dtype": "bfloat16",
        "cpu_threads": 8,
        "max_new_tokens": 512,
        "seed": 20261004,
        "do_sample": False,
        "image_pixels": {"min": args.min_pixels, "max": args.max_pixels},
        "renderer": "WeasyPrint fixed CSS",
        "renderer_fonts": renderer.fonts,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["torch", "transformers", "weasyprint", "lxml", "Pillow", "pydyf", "cffi"]
        },
        "status": "running",
        "calls": 0,
        "arms": ["unchanged_0", *args.modes],
        "cell_map_source": (
            "initial_html_only" if "source_only_patch_indexed" in args.modes else "none"
        ),
        "gate": "rollback only for adapter/schema/render failure; no learned visual judge",
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        proposer = QwenFormulaProposer(
            args.model,
            device="cpu",
            max_new_tokens=512,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        with (args.output / "predictions.jsonl").open("w") as sink:
            for case in inputs:
                initial = case["prediction"]
                source = args.inputs.parent / case["source_image"]
                rendered, initial_error = None, None
                try:
                    rendered = Path(
                        renderer.render(Observation(case["sample_id"], "table", "", initial)).path
                    )
                except Exception as exc:
                    initial_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                for mode in ["unchanged_0", *args.modes]:
                    result = {
                        "sample_id": case["sample_id"],
                        "family_id": case["family_id"],
                        "arm": mode,
                        "initial_prediction": initial,
                        "final_prediction": initial,
                        "initial_render_error": initial_error,
                        "trace": [],
                    }
                    if mode != "unchanged_0":
                        if rendered is None and mode in {
                            "source_first_rewrite",
                            "source_last_rewrite",
                        }:
                            result["skip_reason"] = "initial_render_failure"
                        else:
                            paths, messages, prompt = message(source, initial, rendered, mode)
                            call = proposer.generate(paths, messages, prompt)
                            metadata["calls"] += 1
                            candidate, adapter_error, render_error, action = None, None, None, None
                            try:
                                if mode in PATCH_MODES:
                                    candidate, action = apply_table_action(
                                        initial, call["raw_output"]
                                    )
                                    extraction = "json_action"
                                else:
                                    candidate, extraction = extract_table(call["raw_output"])
                            except Exception as exc:
                                extraction = "adapter_failure"
                                adapter_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                            if candidate is not None:
                                try:
                                    renderer.render(
                                        Observation(case["sample_id"], "table", "", candidate)
                                    )
                                    result["final_prediction"] = candidate
                                except Exception as exc:
                                    render_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                            call.update(
                                {
                                    "candidate": candidate,
                                    "action": action,
                                    "extraction": extraction,
                                    "adapter_error": adapter_error,
                                    "render_error": render_error,
                                    "hit_length_cap": call["output_tokens"] >= 512,
                                }
                            )
                            result["trace"] = [call]
                    sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                    sink.flush()
                print(
                    f"{case['sample_id']}: {len(args.modes)} proposal arms; {metadata['calls']} model calls",
                    flush=True,
                )
        metadata["status"] = "completed"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        raise
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
