"""Output-only, post-hoc replay controls; never infer correctness from rendering."""

from __future__ import annotations

import re

from .formula_pilot import extract_formula

ENVIRONMENT = re.compile(
    r"\s*\\begin\s*\{(equation\*?|displaymath|align\*?|gather\*?)\}" r"(.*?)\\end\s*\{\1\}\s*",
    re.DOTALL,
)


def normalize_outer_environment(candidate: str) -> tuple[str, str]:
    """Normalize only a whole display wrapper, preserving every body token.

    This is an exploratory adapter control. It does not remove prose, labels,
    document commands, or unmatched environments, or select a better candidate.
    """
    match = ENVIRONMENT.fullmatch(candidate)
    if match is None:
        return candidate, "identity"
    environment, body = match.groups()
    body = body.strip()
    if environment.startswith("align"):
        return r"\begin{aligned}" + body + r"\end{aligned}", "align_to_aligned"
    if environment.startswith("gather"):
        return r"\begin{gathered}" + body + r"\end{gathered}", "gather_to_gathered"
    return body, "strip_display_environment"


def audit_candidate(raw_output: str, policy: str) -> tuple[str | None, str]:
    """Decide using generated output only, before offline labels are loaded."""
    candidate, extraction = extract_formula(raw_output)
    if policy == "require_latex_tags" and extraction != "latex_tags":
        return None, "missing_latex_contract"
    if policy not in {"normalize_environment", "require_latex_tags"}:
        raise ValueError("Unknown replay policy")
    return normalize_outer_environment(candidate)
