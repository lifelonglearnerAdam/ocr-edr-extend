"""Complete source-matched native candidates, without labels in student input."""

from .table_pilot import parse_table


def native_model_identity(run, receipt_hash, legacy_provenance=None):
    """Old receipts stay immutable; replay needs an explicitly bound later audit."""
    if run.get("model_receipt_sha256"):
        return run["model_receipt_sha256"]
    audit = legacy_provenance or {}
    if (
        audit.get("status") != "audit_completed"
        or audit.get("historical_base_receipt_hash_missing") is not True
        or audit.get("current_model_files_and_controller_path_verified") is not True
        or not audit.get("current_model_receipt_sha256")
        or audit.get("evidence_sha256", {}).get(run["condition"] + "_receipt") != receipt_hash
    ):
        raise ValueError(
            "Historical native replay requires an explicit hash-bound provenance audit"
        )
    return audit["current_model_receipt_sha256"]


def verify_native_calls(calls, inputs, condition):
    """Require one frozen call per input, with the original source and candidate."""
    if len(calls) != len(inputs) or not inputs:
        raise ValueError("Incomplete native model calls")
    if len({r["sample_id"] for r in calls}) != len(calls):
        raise ValueError("Duplicate native calls")
    for call, item in zip(calls, inputs):
        if (
            any(call.get(k) != v for k, v in item.items())
            or call.get("condition") != condition
            or call.get("ordered_image_sha256") != [item["source_sha256"]]
        ):
            raise ValueError("Native call order, condition, image or candidate identity changed")


def native_repair_inputs(predictions, sources, *, role):
    if role != "model_dev":
        raise ValueError("This diagnostic allows only model-dev sources")
    expected = {r["family_id"]: r for r in sources}
    actual = {r["family_id"]: r for r in predictions}
    if (
        not expected
        or len(expected) != len(sources)
        or len(actual) != len(predictions)
        or expected.keys() != actual.keys()
    ):
        raise ValueError("Complete unique native source/candidate coverage required")
    rows = []
    for source in sources:
        candidate = actual[source["family_id"]]
        if (
            any(candidate.get(k) != v for k, v in source.items())
            or candidate["parser_error"]
            or not isinstance(candidate["prediction"], str)
        ):
            raise ValueError("Native identity differs or parser failure needs a separate protocol")
        parse_table(candidate["prediction"])
        rows.append(
            {
                "sample_id": source["family_id"] + "-native",
                **source,
                "prediction": candidate["prediction"],
            }
        )
    return rows
