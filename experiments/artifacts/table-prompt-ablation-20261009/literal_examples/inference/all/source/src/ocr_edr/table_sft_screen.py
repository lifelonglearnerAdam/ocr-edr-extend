"""Reference-free input, checkpoint and bounded-action contracts for table SFT."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from .formula_pilot import validate_pilot_inputs
from .loop import Observation
from .sft import sha256
from .table_pilot import apply_table_action, parse_table
from .table_supervision import table_action_prompt

PROMPT_FORMATS = ("descriptive_schema", "literal_examples")
LITERAL_ACTION_EXAMPLES = (
    "\nExact response-shape examples (indices and replacement text are placeholders; "
    "choose your own from the current HTML and source image, and emit exactly one object):\n"
    '{"action":"stop"}\n'
    '{"action":"replace_cell","row":0,"cell":0,"text":"correct plain cell text"}\n'
    '{"action":"set_span","row":0,"cell":0,"rowspan":1,"colspan":1}\n'
    '{"action":"delete_row","row":0}'
)


def load_table_screen_inputs(dataset_root: Path) -> list[dict]:
    """Load all model-dev inputs; never open a target/reference or held-out image."""
    root = dataset_root.resolve()
    metadata = json.loads((root / "dataset.json").read_text())
    paths = [root / "model_dev-inputs.jsonl", root / "selected_sources.jsonl"]
    for path in paths:
        if sha256(path) != metadata["file_sha256"][path.name]:
            raise ValueError("Frozen table input/inventory hash mismatch")
    cases, sources = [
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for path in paths
    ]
    inventory = {s["family_id"]: s for s in sources}
    if len(inventory) != len(sources):
        raise ValueError("Duplicate source family in inventory")
    role = metadata["roles"]["model_dev"]
    if not cases or len(cases) != role["cases"]:
        raise ValueError("Every frozen model-dev case must be retained")
    documents = set()
    for case in cases:
        source = inventory.get(case["family_id"])
        if source is None or source["role"] != "model_dev":
            raise ValueError("Model-dev inference source role mismatch")
        image = (root / case["source_image"]).resolve()
        image.relative_to(root)
        if (
            image != (root / source["source_image"]).resolve()
            or case["source_sha256"] != source["source_sha256"]
        ):
            raise ValueError("Inference source identity changed")
        parse_table(case["prediction"])
        documents.add(source["document_id"])
    if (
        len({case["family_id"] for case in cases}) != role["sources"]
        or len(documents) != role["sources"]
    ):
        raise ValueError("Expected one source per frozen model-dev document")
    validate_pilot_inputs(cases, root)
    return cases


def validate_table_adapter(
    run_root: Path,
    *,
    condition: str,
    config: dict,
    config_sha256: str,
    admission_sha256: str,
) -> dict:
    """Reject cross-study, incomplete or modified adapters before model loading."""
    run = json.loads((run_root / "run.json").read_text())
    if (
        condition not in {"all", "no_explicit_preservation"}
        or run.get("arm") != condition
        or run.get("status") != "completed"
        or run.get("study") != config["study"]
        or run.get("config") != config
        or run.get("config_sha256") != config_sha256
        or run.get("completed_steps") != config["steps"]
        or run.get("dev_optimizer_examples") != 0
        or run.get("calibration_locked_optimizer_examples") != 0
        or run.get("admission_sha256") != admission_sha256
    ):
        raise ValueError("Table adapter completion/protocol/admission mismatch")
    hashes = run.get("checkpoint_sha256", {})
    if not {"adapter_config.json", "adapter_model.safetensors"} <= hashes.keys():
        raise ValueError("Missing adapter checkpoint integrity records")
    root = (run_root / "checkpoint").resolve()
    if {p.name for p in root.iterdir()} != set(hashes):
        raise ValueError("Unexpected or missing checkpoint files")
    for name, expected in hashes.items():
        if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Unsafe checkpoint file name")
        path = (root / name).resolve(strict=True)
        if path.parent != root or not path.is_file() or sha256(path) != expected:
            raise ValueError("Table adapter checkpoint integrity mismatch")
    return hashes


def table_messages(
    initial: str, *, prompt_format: str = "descriptive_schema"
) -> tuple[list[dict], str]:
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError("Unknown table prompt format")
    prompt = table_action_prompt(initial)
    if prompt_format == "literal_examples":
        prompt += LITERAL_ACTION_EXAMPLES
    return [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    ], prompt


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON action key")
        value[key] = item
    return value


def adapt_table_call(
    case: dict,
    call: dict,
    *,
    renderer,
    max_new_tokens: int,
    prompt_format: str = "descriptive_schema",
) -> dict:
    """Apply at most one strict action, rolling back failures without erasing cost.

    The renderer is a callable taking an Observation. Neither it nor this function
    receives a reference, known error location, dataset variant or target action.
    """
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("Positive output-token cap required")
    messages, prompt = table_messages(case["prediction"], prompt_format=prompt_format)
    if (
        any(call.get(key) != value for key, value in case.items())
        or call.get("prompt") != prompt
        or call.get("messages") != messages
        or call.get("ordered_image_sha256") != [case["source_sha256"]]
        or call.get("condition") not in {"base", "all", "no_explicit_preservation"}
    ):
        raise ValueError("Frozen call input/evidence/condition mismatch")
    for name in ["input_tokens", "output_tokens"]:
        if type(call.get(name)) is not int or call[name] < 0:
            raise ValueError("Invalid generation token accounting")
    if (
        not isinstance(call.get("generation_seconds"), (int, float))
        or not math.isfinite(call["generation_seconds"])
        or call["generation_seconds"] < 0
    ):
        raise ValueError("Invalid generation time accounting")
    cap = call["output_tokens"] >= max_new_tokens
    trace = {
        **call,
        "candidate": None,
        "action": None,
        "adapter_error": None,
        "render_error": None,
        "hit_length_cap": cap,
        "contract": "token_cap" if cap else "invalid_output_contract",
        "render_attempts": 0,
        "render_seconds": 0.0,
    }
    result = {
        "sample_id": case["sample_id"],
        "family_id": case["family_id"],
        "arm": call["condition"],
        "initial_prediction": case["prediction"],
        "final_prediction": case["prediction"],
        "trace": [trace],
    }
    if cap:
        trace["adapter_error"] = "token_cap"
        return result
    try:
        action = json.loads(call["raw_output"], object_pairs_hook=_unique_object)
        if not isinstance(action, dict):
            raise ValueError("Expected one JSON action object")
        candidate, action = apply_table_action(case["prediction"], json.dumps(action))
    except (ValueError, TypeError) as error:
        trace["adapter_error"] = f"{type(error).__name__}: {str(error)[:200]}"
        return result
    trace.update(
        candidate=candidate, action=action, contract="accepted_contract", render_attempts=1
    )
    start = time.perf_counter()
    try:
        renderer(Observation(case["sample_id"], "table", "", candidate))
        result["final_prediction"] = candidate
    except Exception as error:
        # External rendering failures are measured rejections, never dropped cases.
        trace["render_error"] = f"{type(error).__name__}: {str(error)[:200]}"
    finally:
        trace["render_seconds"] = time.perf_counter() - start
    return result
