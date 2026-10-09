"""Validate public research facts and render a self-contained HTML snapshot."""

from __future__ import annotations

import base64
import json
import math
import re
from datetime import datetime


def validate_snapshot(snapshot):
    if snapshot.get("schema_version") != 1:
        raise ValueError("Unknown research snapshot schema")
    datetime.fromisoformat(snapshot["updated_at"])
    table = snapshot["table_screen"]
    if table["cases"] != 103 or table["documents"] != 32:
        raise ValueError("Incomplete frozen table development coverage")
    rows = table["arms"]
    if len(rows) != 4 or {r["arm"] for r in rows} != {
        "unchanged_0",
        "base",
        "all",
        "no_explicit_preservation",
    }:
        raise ValueError("All four table conditions must remain visible")
    for row in rows:
        if (
            type(row["teds"]) not in {int, float}
            or not math.isfinite(row["teds"])
            or not 0 <= row["teds"] <= 1
        ):
            raise ValueError("Invalid finite table quality")
        for name, bound in [("repairs", 71), ("regressions", 32), ("changed", 103)]:
            if type(row[name]) is not int or not 0 <= row[name] <= bound:
                raise ValueError("Table result count differs from its denominator")
    peer = snapshot["senior"]["evaluation"]
    if peer["n"] != 1800 or not peer["metrics"]:
        raise ValueError("Preserve the supplied verifier evaluation denominator")
    for row in peer["metrics"]:
        for metric in ["verdict_percent", "strict_joint_percent", "admissible_joint_percent"]:
            if (
                not isinstance(row[metric], (int, float))
                or not math.isfinite(row[metric])
                or not 0 <= row[metric] <= 100
            ):
                raise ValueError("Invalid collaborator-reported percentage")
    encoded = json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
    forbidden = r"/home/|/run/media/|C:[/\\]Users[/\\]|-----BEGIN [A-Z ]*PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{20}|github_pat_[A-Za-z0-9_]{20}|passwd\s*:"
    if re.search(forbidden, encoded, flags=re.IGNORECASE):
        raise ValueError("Private paths or authentication patterns cannot enter the public report")
    return snapshot


def render_dashboard(snapshot, template, assets):
    validate_snapshot(snapshot)
    if template.count("{{DATA}}") != 1:
        raise ValueError("Template needs exactly one embedded research-data slot")
    encoded = (
        json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    rendered = template.replace("{{DATA}}", encoded)

    def embed(match):
        name = match[1]
        if (
            not re.fullmatch(r"[a-z0-9_-]+\.png", name)
            or name not in assets
            or not assets[name].startswith(b"\x89PNG\r\n\x1a\n")
        ):
            raise ValueError("Missing or unvetted local PNG asset")
        return "data:image/png;base64," + base64.b64encode(assets[name]).decode()

    return re.sub(r"\{\{ASSET:([^}]+)\}\}", embed, rendered)
