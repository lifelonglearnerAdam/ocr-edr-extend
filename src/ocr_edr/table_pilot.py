"""A strict HTML-table adapter and bounded edits for development diagnostics."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path

from .formula_pilot import sha256_file
from .loop import Observation, Rendered, digest

ALLOWED_TAGS = {
    "table",
    "thead",
    "tbody",
    "tfoot",
    "tr",
    "td",
    "th",
    "b",
    "strong",
    "i",
    "em",
    "sub",
    "sup",
    "br",
}
ALLOWED_ATTRIBUTES = {"rowspan", "colspan", "border"}


def parse_table(markup: str):
    from lxml import etree, html

    value = markup.strip()
    if not value or len(value) > 50000:
        raise ValueError("Empty or oversized table")
    if re.search(r"<!DOCTYPE|<!ENTITY|<!--", value, flags=re.IGNORECASE):
        raise ValueError("Unsupported table declaration")
    if not re.fullmatch(
        r"(?:<html>\s*<body>\s*)?<table\b.*</table>(?:\s*</body>\s*</html>)?",
        value,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        raise ValueError("Expected one complete HTML table without surrounding prose")
    document = html.fromstring(value)
    tables = document.xpath("descendant-or-self::table")
    if len(tables) != 1:
        raise ValueError("Expected exactly one table")
    table = tables[0]
    for element in table.iter():
        if element.tag not in ALLOWED_TAGS or set(element.attrib) - ALLOWED_ATTRIBUTES:
            raise ValueError("Unsupported table element or attribute")
        for name in ["rowspan", "colspan"]:
            if name in element.attrib:
                raw = element.attrib[name]
                if not raw.isdecimal() or not 1 <= int(raw) <= 100:
                    raise ValueError("Invalid positive table span")
        if element.tag in {"td", "th"} and element.getparent().tag != "tr":
            raise ValueError("Cell must belong to a row")
    rows = table.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr")
    if not rows or len(rows) > 100 or not table.xpath(".//td|.//th"):
        raise ValueError("Empty or oversized table structure")
    if len(table.xpath(".//td|.//th")) > 400:
        raise ValueError("Oversized table cell count")
    # Parsing must not silently drop a truncation or change cell/row cardinality.
    if len(re.findall(r"<tr\b", value, re.IGNORECASE)) != len(rows):
        raise ValueError("Ambiguous or nested rows")
    if len(re.findall(r"<t[dh]\b", value, re.IGNORECASE)) != len(table.xpath(".//td|.//th")):
        raise ValueError("Ambiguous or nested cells")
    return table, etree.tostring(table, encoding="unicode", method="html")


def extract_table(raw: str) -> tuple[str, str]:
    match = re.fullmatch(r"\s*```(?:html)?\s*\n(.*?)\n```\s*", raw, flags=re.DOTALL)
    return (match[1].strip(), "code_fence") if match else (raw.strip(), "unwrapped")


def apply_table_action(initial: str, raw_output: str) -> tuple[str, dict]:
    """One validated, atomic operation addressed in the initial row/cell order."""
    text = raw_output.strip()
    match = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
    action = json.loads(match[1] if match else text)
    if not isinstance(action, dict):
        raise ValueError("Expected one JSON action object")
    kind = action.get("action")
    schemas = {
        "stop": {"action"},
        "replace_cell": {"action", "row", "cell", "text"},
        "set_span": {"action", "row", "cell", "rowspan", "colspan"},
        "delete_row": {"action", "row"},
    }
    if kind not in schemas or set(action) != schemas[kind]:
        raise ValueError("Unexpected bounded-edit schema")
    table, _ = parse_table(initial)
    if kind == "stop":
        return initial, action
    table = copy.deepcopy(table)
    rows = table.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr")
    row = action["row"]
    if type(row) is not int or not 0 <= row < len(rows):
        raise ValueError("Row index out of range")
    target_row = rows[row]
    if kind == "delete_row":
        target_row.getparent().remove(target_row)
    else:
        cells = target_row.xpath("./td|./th")
        cell = action["cell"]
        if type(cell) is not int or not 0 <= cell < len(cells):
            raise ValueError("Cell index out of range")
        target = cells[cell]
        if kind == "replace_cell":
            content = action["text"]
            if not isinstance(content, str) or len(content) > 1000:
                raise ValueError("Invalid replacement cell text")
            for child in list(target):
                target.remove(child)
            target.text = content  # Plain text is escaped on serialization.
        else:
            for attribute in ["rowspan", "colspan"]:
                span = action[attribute]
                if type(span) is not int or not 1 <= span <= 100:
                    raise ValueError("Invalid positive span")
                if span == 1:
                    target.attrib.pop(attribute, None)
                else:
                    target.set(attribute, str(span))
    from lxml import etree

    candidate = etree.tostring(table, encoding="unicode", method="html")
    parse_table(candidate)
    return candidate, action


CSS = """
@page { size: 1200px 1800px; margin: 18px; }
body { font-family: "Noto Sans CJK SC", sans-serif; font-size: 20px; }
table { border-collapse: collapse; width: auto; max-width: 1150px; }
td, th { border: 1px solid black; padding: 5px 8px; font-weight: normal; }
"""


class HTMLTableRenderer:
    """Fixed CSS HTML rendering. Successful compilation is never a correctness gate."""

    def __init__(self, output_root: Path, *, dpi: int = 120):
        self.output_root = output_root.resolve()
        if dpi <= 0:
            raise ValueError("Positive DPI required")
        self.dpi = dpi
        self.version = importlib.metadata.version("weasyprint")
        self.fonts = {}
        for family in ["Noto Sans CJK SC", "sans-serif"]:
            location = subprocess.check_output(
                ["fc-match", "-f", "%{file}", family], text=True
            ).strip()
            if not location or not Path(location).is_file():
                raise ValueError("Configured table font is unavailable")
            self.fonts[family] = {"path": location, "sha256": sha256_file(Path(location))}

    def render(self, observation: Observation) -> Rendered:
        from PIL import Image, ImageChops, ImageOps
        from weasyprint import CSS as WeasyCSS
        from weasyprint import HTML

        if observation.modality != "table":
            raise ValueError("HTML table renderer supports tables only")
        _, markup = parse_table(observation.prediction)
        font_hash = digest(json.dumps(self.fonts, sort_keys=True))
        backend = f"weasyprint:{self.version}:{digest(CSS)}:{font_hash}:{self.dpi}"
        work = self.output_root / digest(backend + ":" + observation.prediction)
        work.mkdir(parents=True, exist_ok=True)
        path = work / "render.png"
        receipt = work / "render.json"
        if path.exists() and receipt.exists():
            cached = json.loads(receipt.read_text())
            if cached.get("png_sha256") == sha256_file(path):
                return Rendered(str(path), digest(observation.prediction), backend)

        def no_fetch(url, *args, **kwargs):
            raise ValueError("Table markup cannot request external content")

        html = "<html><body>" + markup + "</body></html>"
        (work / "input.html").write_text(html)
        document = HTML(string=html, url_fetcher=no_fetch).render(
            stylesheets=[WeasyCSS(string=CSS)]
        )
        if len(document.pages) != 1:
            raise ValueError("Table exceeds the fixed one-page render budget")
        pdf = work / "render.pdf"
        document.write_pdf(pdf)
        subprocess.run(
            ["pdftoppm", "-singlefile", "-r", str(self.dpi), "-png", str(pdf), str(work / "page")],
            check=True,
            capture_output=True,
            timeout=45,
        )
        with Image.open(work / "page.png") as original:
            image = original.convert("RGB")
        bounds = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
        if bounds is None:
            raise ValueError("Empty rendered table")
        image = ImageOps.expand(image.crop(bounds), border=16, fill="white")
        image.save(path)
        receipt.write_text(
            json.dumps(
                {
                    "backend": backend,
                    "prediction_sha256": digest(observation.prediction),
                    "png_sha256": sha256_file(path),
                    "dimensions": list(image.size),
                    "fonts": self.fonts,
                },
                indent=2,
            )
            + "\n"
        )
        return Rendered(str(path), digest(observation.prediction), backend)
