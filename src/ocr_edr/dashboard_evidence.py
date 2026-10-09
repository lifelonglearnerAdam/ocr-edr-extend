"""Bind public result updates to sealed complete experiment evidence."""

import hashlib
import json


def live_optimizer_progress(run, lines, *, service, idle_seconds):
    """Only completed optimizer log rows show progress; service activity is separate."""
    steps = run["completed_steps"]
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        steps = max(steps, entry["step"])
        break
    if type(steps) is not int or not 0 <= steps <= 381:
        raise ValueError("Invalid recorded diagnosis optimizer progress")
    return {
        "observed_steps": steps,
        "stalled": service == "active"
        and run["status"] == "running"
        and steps < 381
        and idle_seconds >= 300,
    }


def verify_source_probe_coverage(comparison):
    """Two full32-source cohorts; displayed agreement is a count, not accuracy."""
    arms = {"all", "no_explicit_preservation"}
    rows = comparison["cases"]
    if len(rows) != 64 or len({(r["arm"], r["sample_id"]) for r in rows}) != 64:
        raise ValueError("Source probe needs all64 unique paired results")
    summary = comparison["summary"]
    if len(summary) != 2 or {r["arm"] for r in summary} != arms:
        raise ValueError("Both source probe conditions required")
    identities = None
    for arm in arms:
        group = [r for r in rows if r["arm"] == arm]
        ids = {r["sample_id"] for r in group}
        if len(ids) != 32 or (identities is not None and identities != ids):
            raise ValueError("Source probe source pairing differs")
        identities = ids
        total = next(r for r in summary if r["arm"] == arm)
        if total["cases"] != 32:
            raise ValueError("Source probe denominator differs")
        for key in [
            "raw_output_equal",
            "action_equal",
            "final_html_equal",
            "normalized_html_equal",
            "hit_length_cap",
        ]:
            if (
                any(type(r[key]) is not bool for r in group)
                or type(total[key]) is not int
                or total[key] != sum(r[key] for r in group)
            ):
                raise ValueError("Source agreement summary differs from all paired cases")


def verify_completion_binding(completion, observed_hashes):
    if (
        completion.get("status") != "completed"
        or completion.get("scientific_outputs_verified") is not True
    ):
        raise ValueError("The full experiment requires independent terminal verification")
    pinned = completion.get("evidence_sha256", {})
    if not observed_hashes or any(
        pinned.get(name) != value for name, value in observed_hashes.items()
    ):
        raise ValueError("Current result file differs from the independent completion seal")


def verify_four_arm_coverage(evaluation):
    arms = {"unchanged_0", "base", "all", "no_explicit_preservation"}
    rows = evaluation["cases"]
    indexed = {(row["arm"], row["sample_id"]): row for row in rows}
    if len(indexed) != len(rows) or len(rows) != 4 * 103 or {r["arm"] for r in rows} != arms:
        raise ValueError("Keep complete unique four-arm 103-case coverage")
    identities = {}
    for arm in arms:
        cases = {r["sample_id"]: r["document_id"] for r in rows if r["arm"] == arm}
        if len(cases) != 103 or len(set(cases.values())) != 32:
            raise ValueError("Each model condition requires all32 documents and103 cases")
        if identities and cases != identities:
            raise ValueError("Prompt evaluation source/document pairing differs")
        identities = cases
    summary = [r for r in evaluation["summary"] if r["variant"] == "all"]
    if (
        len(summary) != 4
        or {r["arm"] for r in summary} != arms
        or any(r["cases"] != 103 or r["documents"] != 32 for r in summary)
    ):
        raise ValueError("Complete four-arm aggregate summaries are required")


def verify_prompt_stage(pipeline, prompt, receipt_hash):
    candidates = [
        r for r in pipeline.get("completed_stages", []) if r["stage"] == prompt + "_evaluation"
    ]
    if len(candidates) != 1 or candidates[0]["receipt_sha256"] != receipt_hash:
        raise ValueError("Prompt evaluation differs from its completed pipeline stage seal")


def verify_native_coverage(evaluation):
    """Native diagnostic has one candidate per document, all32 in each condition."""
    arms = {"unchanged_0", "base", "all", "no_explicit_preservation"}
    rows = evaluation["cases"]
    if len(rows) != 128 or len({(r["arm"], r["sample_id"]) for r in rows}) != 128:
        raise ValueError("Native diagnostic needs complete unique128 result rows")
    identities = None
    for arm in arms:
        cases = {r["sample_id"]: r["parent_page"] for r in rows if r["arm"] == arm}
        if len(cases) != 32 or len(set(cases.values())) != 32:
            raise ValueError("Each native condition requires the same32 source documents")
        if identities is not None and cases != identities:
            raise ValueError("Native result source pairing differs")
        identities = cases
    summary = [r for r in evaluation["summary"] if r["variant"] == "all"]
    if len(summary) != 4 or {r["arm"] for r in summary} != arms:
        raise ValueError("All four native summaries required")
    denominator = None
    for row in summary:
        counts = (row["teds_initial_matching_n"], row["teds_initial_nonmatching_n"])
        if (
            row["cases"] != 32
            or row["pages"] != 32
            or sum(counts) != 32
            or any(type(n) is not int or n < 0 for n in counts)
            or (denominator is not None and counts != denominator)
        ):
            raise ValueError("Native denominators must match the complete paired cohort")
        denominator = counts


def verify_walkthrough_cases(projection, evaluation, predictions):
    """Bind displayed actions, scores and HTML digests to actual frozen outputs."""

    def digest(markup):
        return hashlib.sha256(markup.encode()).hexdigest()

    scores = {(r["sample_id"], r["arm"]): r for r in evaluation["cases"]}
    outputs = {(r["sample_id"], r["arm"]): r for r in predictions}
    if not projection or len({r["sample_id"] for r in projection}) != len(projection):
        raise ValueError("Empty or duplicate walkthrough cases")
    for case in projection:
        if not case["outcomes"] or len({r["arm"] for r in case["outcomes"]}) != len(
            case["outcomes"]
        ):
            raise ValueError("Walkthrough model outcomes must be distinct")
        for outcome in case["outcomes"]:
            key = (case["sample_id"], outcome["arm"])
            if key not in scores or key not in outputs:
                raise ValueError("Walkthrough output is absent from the frozen experiment")
            actual, output = scores[key], outputs[key]
            if (
                case["source_document"] != actual.get("document_id", actual.get("parent_page"))
                or case["input_sha256"] != digest(output["initial_prediction"])
                or outcome["output_sha256"] != digest(output["final_prediction"])
                or outcome["action"]
                != actual.get("proposed_action", (output.get("trace") or [{}])[0].get("action"))
                or any(outcome[k] != actual[k] for k in ["initial_teds", "final_teds"])
            ):
                raise ValueError("Displayed case facts differ from the actual frozen output")
