"""Bind public result updates to sealed complete experiment evidence."""


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
