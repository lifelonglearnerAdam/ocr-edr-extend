"""Full-TeX formula rendering with the bundled, untrusted Tectonic runtime.

This renderer supports real LaTeX constructs. Exact raster equality remains a
restricted proxy, not CDM or visual equivalence on arbitrary source images.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from .formula_pilot import sha256_file, strip_math_wrappers
from .loop import Observation, Rendered, digest

TEMPLATE = r"""\documentclass[border=4pt]{standalone}
\usepackage{amsmath,amssymb,mathtools,mathrsfs,bm}
\begin{document}
$\displaystyle FORMULA_PLACEHOLDER$
\end{document}
"""


def find_tectonic() -> Path:
    candidates = [shutil.which("tectonic"), "/usr/lib/chatgpt/resources/tectonic/tectonic"]
    for value in candidates:
        if value and Path(value).is_file():
            return Path(value).resolve()
    raise FileNotFoundError("Tectonic not found; supply its executable path")


def validate_formula(markup: str) -> str:
    formula = strip_math_wrappers(markup)
    forbidden = re.compile(
        r"\\(?:input|include|includegraphics|openin|openout|read|write|catcode|csname|"
        r"def|edef|gdef|xdef|let|special|usepackage|documentclass|newcommand|renewcommand|"
        r"newenvironment)(?![A-Za-z])"
    )
    if not formula or "$" in formula or forbidden.search(formula):
        raise ValueError("Formula contains empty/nested delimiters or document-control commands")
    if re.search(r"\\(?:begin|end)\s*\{\s*document\s*\}", formula):
        raise ValueError("Formula cannot change the document boundary")
    return formula


class TectonicRenderer:
    def __init__(
        self,
        output_root: Path,
        *,
        executable: Path | None = None,
        dpi: int = 160,
        timeout: float = 45,
    ):
        if dpi <= 0 or timeout <= 0:
            raise ValueError("Positive DPI and timeout required")
        self.output_root = output_root.resolve()
        self.executable = (executable or find_tectonic()).resolve()
        self.dpi = dpi
        self.timeout = timeout
        self.binary_sha256 = sha256_file(self.executable)

    def render(self, observation: Observation) -> Rendered:
        from PIL import Image, ImageChops, ImageOps

        if observation.modality != "formula":
            raise ValueError("Tectonic formula renderer does not support tables")
        formula = validate_formula(observation.prediction)
        backend_key = f"tectonic:{self.binary_sha256}:{digest(TEMPLATE)}:{self.dpi}"
        key = digest(backend_key + ":" + observation.prediction)
        work = self.output_root / key
        work.mkdir(parents=True, exist_ok=True)
        path = work / "render.png"
        receipt = work / "render.json"
        if path.exists() and receipt.exists():
            cached = json.loads(receipt.read_text())
            if cached.get("png_sha256") == sha256_file(path):
                return Rendered(str(path), digest(observation.prediction), backend_key)
        source = work / "formula.tex"
        source.write_text(TEMPLATE.replace("FORMULA_PLACEHOLDER", formula))
        command = [
            str(self.executable),
            "-X",
            "compile",
            "--untrusted",
            "--keep-logs",
            "--outdir",
            str(work),
            str(source),
        ]
        process = subprocess.run(
            command, cwd=work, capture_output=True, text=True, timeout=self.timeout
        )
        (work / "compiler.log").write_text(process.stdout + process.stderr)
        if process.returncode != 0 or not (work / "formula.pdf").exists():
            raise ValueError(f"TeX compile failed with exit {process.returncode}")
        convert = [
            "pdftoppm",
            "-f",
            "1",
            "-singlefile",
            "-r",
            str(self.dpi),
            "-png",
            str(work / "formula.pdf"),
            str(work / "page"),
        ]
        subprocess.run(convert, cwd=work, check=True, capture_output=True, timeout=self.timeout)
        with Image.open(work / "page.png") as original:
            image = original.convert("RGB")
        bbox = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
        if bbox is None:
            raise ValueError("TeX rendered an empty formula")
        image = ImageOps.expand(image.crop(bbox), border=16, fill="white")
        image.save(path)
        receipt.write_text(
            json.dumps(
                {
                    "prediction_sha256": digest(observation.prediction),
                    "backend": backend_key,
                    "tex_sha256": sha256_file(source),
                    "png_sha256": sha256_file(path),
                    "command": command,
                    "dimensions": list(image.size),
                },
                indent=2,
            )
            + "\n"
        )
        return Rendered(str(path), digest(observation.prediction), backend_key)
