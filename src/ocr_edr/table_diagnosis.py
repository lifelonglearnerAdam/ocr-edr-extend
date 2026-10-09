"""Candidate-bound diagnosis targets and advisory hints, never corrected answers."""

import hashlib
import json

from .table_pilot import apply_table_action, table_cell_map


def verify_guided_runtime(guided, baseline):
    fields = [
        "config_sha256",
        "versions",
        "gpu",
        "cuda_runtime",
        "quantization",
        "adapter_sha256",
        "model_receipt_sha256",
    ]
    if any(k not in guided or k not in baseline or guided[k] != baseline[k] for k in fields):
        raise ValueError("Guided and unhinted model/runtime conditions differ")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate diagnosis JSON key")
        result[key] = value
    return result


def parse_diagnosis(initial, raw):
    try:
        value = json.loads(raw, object_pairs_hook=_object)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Expected one strict diagnosis object") from error
    if not isinstance(value, dict) or set(value) != {"verdict", "error", "region"}:
        raise ValueError("Diagnosis requires exactly verdict/error/region")
    if value["verdict"] == "valid":
        if value["error"] is not None or value["region"] is not None:
            raise ValueError("Valid verdict cannot name an error or region")
        return value
    if (
        value["verdict"] != "invalid"
        or not isinstance(value["error"], str)
        or value["error"] not in {"content", "extra", "structure"}
    ):
        raise ValueError("Unknown diagnosis verdict or error")
    region = value["region"]
    if (
        not isinstance(region, dict)
        or not isinstance(region.get("unit"), str)
        or region["unit"] not in {"cell", "row"}
    ):
        raise ValueError("Diagnosis region must name a current cell or row")
    fields = {"unit", "row", "cell"} if region["unit"] == "cell" else {"unit", "row"}
    if set(region) != fields or any(
        type(region[k]) is not int or region[k] < 0 for k in fields - {"unit"}
    ):
        raise ValueError("Diagnosis indices must be nonnegative integers")
    cells = table_cell_map(initial)
    candidates = {(r["row"], r["cell"]) for r in cells}
    if (region["unit"] == "cell" and (region["row"], region["cell"]) not in candidates) or (
        region["unit"] == "row" and region["row"] not in {r["row"] for r in cells}
    ):
        raise ValueError("Diagnosis region absent from current candidate")
    return value


def diagnosis_from_action(initial, action):
    """Offline controlled-label projection, omitting replacement text/restored spans."""
    _, action = apply_table_action(initial, json.dumps(action))
    verb = action["action"]
    if verb == "stop":
        value = {"verdict": "valid", "error": None, "region": None}
    elif verb in {"replace_cell", "set_span"}:
        value = {
            "verdict": "invalid",
            "error": "content" if verb == "replace_cell" else "structure",
            "region": {"unit": "cell", "row": action["row"], "cell": action["cell"]},
        }
    elif verb == "delete_row":
        value = {
            "verdict": "invalid",
            "error": "extra",
            "region": {"unit": "row", "row": action["row"]},
        }
    else:
        raise ValueError("Unknown controlled diagnostic provenance")
    return parse_diagnosis(initial, json.dumps(value))


def diagnosis_prompt(initial):
    cells = json.dumps(table_cell_map(initial), ensure_ascii=False, separators=(",", ":"))
    return (
        "Image 1 is the source table. Check the current OCR candidate against the image. "
        "Return a diagnosis only, without editing or giving corrected text. A valid verdict means "
        "no visible difference is supported. Otherwise identify one visible content, extra-row or structure error "
        "and its scope inside the current candidate. Candidate indices are not image coordinates.\n"
        "Current HTML:\n"
        + initial
        + "\nCurrent zero-based cell addresses:\n<cell_map>"
        + cells
        + "</cell_map>\n"
        "Return exactly one JSON object with verdict,error,region, no additional fields or prose. "
        "verdict is valid or invalid. For valid, error and region must both be null. For invalid, "
        "error is content, extra or structure. region is an object: cell scope has unit=cell,row,cell; "
        "row scope has unit=row,row. Every row/cell index is an existing zero-based integer from the current DOM. "
        "Do not include replacement text, a repaired HTML table, restored spans or external reference."
    )


def displace_region(initial, value, *, identity, seed):
    value = parse_diagnosis(initial, json.dumps(value))
    if value["verdict"] == "valid":
        return value
    original = value["region"]
    cells = table_cell_map(initial)
    if original["unit"] == "row":
        options = [
            {"unit": "row", "row": r}
            for r in sorted({c["row"] for c in cells})
            if r != original["row"]
        ]
    else:
        options = [
            {"unit": "cell", "row": c["row"], "cell": c["cell"]}
            for c in cells
            if (c["row"], c["cell"]) != (original["row"], original["cell"])
        ]
    if not options:
        return value
    index = int(hashlib.sha256((str(seed) + ":" + identity).encode()).hexdigest(), 16) % len(
        options
    )
    return {**value, "region": options[index]}


def bind_diagnosis(case, value, *, producer, output_sha256):
    if value is not None:
        value = parse_diagnosis(case["prediction"], json.dumps(value))
    return {
        "sample_id": case["sample_id"],
        "source_sha256": case["source_sha256"],
        "candidate_sha256": hashlib.sha256(case["prediction"].encode()).hexdigest(),
        "producer": producer,
        "output_sha256": output_sha256,
        "diagnosis": value,
    }


def verify_diagnosis_binding(case, bound):
    if (
        bound.get("sample_id") != case["sample_id"]
        or bound.get("source_sha256") != case["source_sha256"]
        or bound.get("candidate_sha256") != hashlib.sha256(case["prediction"].encode()).hexdigest()
    ):
        raise ValueError("Diagnosis is stale or from another source/candidate")
    if bound.get("diagnosis") is not None:
        parse_diagnosis(case["prediction"], json.dumps(bound["diagnosis"]))


def diagnosis_advice(case, bound):
    verify_diagnosis_binding(case, bound)
    # Producer metadata stays outside the model prompt: oracle and learned hints
    # share a format; no teacher/model authority cue can substitute for evidence.
    safe = {k: bound[k] for k in ["candidate_sha256", "diagnosis"]}
    return (
        "\nAn advisory diagnosis is attached below. It may be wrong; verify its claim against Image 1. "
        "The scope is a current DOM address, not corrected content. Preserve unaffected content; "
        "if the source supports no edit, return stop. Do not invent corrected text from the diagnosis.\n"
        "<advisory_diagnosis>" + json.dumps(safe, separators=(",", ":")) + "</advisory_diagnosis>"
    )
