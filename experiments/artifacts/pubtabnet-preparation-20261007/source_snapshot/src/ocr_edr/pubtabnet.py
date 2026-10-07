"""PubTabNet annotation conversion and document-isolated supervised local edits."""

from __future__ import annotations

import copy
import hashlib
import html
import json
import re

from .table_pilot import apply_table_action, parse_table


def document_id(filename: str) -> str:
    match = re.fullmatch(r"(PMC\d+)_\d+_\d+\.png", filename)
    if match is None:
        raise ValueError("Published filename has no safe original-document identity")
    return match[1]


def document_role(
    document: str,
    published_split: str,
    *,
    held_out_documents: set[str],
    test_documents: set[str] | None = None,
) -> str | None:
    if re.fullmatch(r"PMC\d+", document) is None or published_split not in {"train", "val", "test"}:
        raise ValueError("Invalid document identity or published split")
    number = int(hashlib.sha256(("20261007-pubtabnet-role:" + document).encode()).hexdigest(), 16)
    if published_split == "train":
        return (
            None if document in held_out_documents else "model_dev" if number % 5 == 0 else "train"
        )
    if published_split == "val":
        return (
            None
            if document in (test_documents or set())
            else "gate_calibration" if number % 3 == 0 else "locked_evaluation"
        )
    return None


def annotation_html(annotation: dict) -> str:
    structure = "".join(annotation["structure"]["tokens"])
    cells = annotation["cells"]
    if len(re.findall(r"</t[dh]>", structure)) != len(cells):
        raise ValueError("Structure/annotated-cell coverage mismatch")
    inline = {"<b>", "</b>", "<i>", "</i>", "<sup>", "</sup>", "<sub>", "</sub>", "<br>", "<br/>"}
    index = 0

    def fill(match):
        nonlocal index
        tokens = cells[index]["tokens"]
        index += 1
        body = []
        for token in tokens:
            if not isinstance(token, str):
                raise ValueError("Annotation text tokens must be strings")
            if (
                token.startswith("<")
                and token.endswith(">")
                and len(token) > 1
                and token not in inline
            ):
                raise ValueError("Unsupported annotated inline tag")
            body.append(token if token in inline else html.escape(token, quote=False))
        return "".join(body) + match[0]

    markup = "<table>" + re.sub(r"</t[dh]>", fill, structure) + "</table>"
    table, canonical = parse_table(markup)
    if len(table.xpath(".//td|.//th")) != len(cells):
        raise ValueError("Parser changed annotated cell coverage")
    return canonical


def bounded_supervision(reference: str) -> list[dict]:
    """Build reversible annotation-derived errors, never call them native trajectories."""
    from lxml import etree

    table, canonical = parse_table(reference)
    result = [
        {"variant": "preservation", "prediction": canonical, "target_action": {"action": "stop"}}
    ]
    digit = copy.deepcopy(table)
    changed = False
    for row_index, row in enumerate(digit.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr")):
        for cell_index, cell in enumerate(row.xpath("./td|./th")):
            if len(cell) or not cell.text:
                continue
            match = re.search(r"[0-9]", cell.text)
            if match:
                original = cell.text
                cell.text = (
                    original[: match.start()]
                    + str((int(match[0]) + 1) % 10)
                    + original[match.end() :]
                )
                result.append(
                    {
                        "variant": "cell_perturbation",
                        "prediction": etree.tostring(digit, encoding="unicode", method="html"),
                        "target_action": {
                            "action": "replace_cell",
                            "row": row_index,
                            "cell": cell_index,
                            "text": original,
                        },
                    }
                )
                changed = True
                break
        if changed:
            break
    if not changed:
        raise ValueError("No plain-text digit cell eligible for a reversible edit")
    span = copy.deepcopy(table)
    found = False
    for row_index, row in enumerate(span.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr")):
        for cell_index, cell in enumerate(row.xpath("./td|./th")):
            for attribute in ["rowspan", "colspan"]:
                value = int(cell.get(attribute, "1"))
                if value > 1:
                    action = {
                        "action": "set_span",
                        "row": row_index,
                        "cell": cell_index,
                        "rowspan": int(cell.get("rowspan", "1")),
                        "colspan": int(cell.get("colspan", "1")),
                    }
                    cell.set(attribute, str(value - 1))
                    result.append(
                        {
                            "variant": "span_perturbation",
                            "prediction": etree.tostring(span, encoding="unicode", method="html"),
                            "target_action": action,
                        }
                    )
                    found = True
                    break
            if found:
                break
        if found:
            break
    extra = copy.deepcopy(table)
    rows = extra.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr")
    if all(int(cell.get("rowspan", "1")) == 1 for cell in rows[-1].xpath("./td|./th")):
        rows[-1].addnext(copy.deepcopy(rows[-1]))
        result.append(
            {
                "variant": "extra_row",
                "prediction": etree.tostring(extra, encoding="unicode", method="html"),
                "target_action": {"action": "delete_row", "row": len(rows)},
            }
        )
    for row in result:
        final, _ = apply_table_action(row["prediction"], json.dumps(row["target_action"]))
        if final != canonical:
            raise ValueError("Annotation-derived target does not exactly restore the reference")
    return result
