"""A controlled MathText diagnostic, with references confined to offline evaluation.

This is a supported LaTeX subset, not a general TeX renderer or the CDM metric.
The inference harness tests proposals before adding a learned acceptance gate.
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

from .loop import Observation, Rendered, digest

FORMULAS = (
    (r"E=mc^{2}", r"E=mc^{3}", "exponent"),
    (r"\frac{a+b}{c}", r"\frac{a-b}{c}", "operator"),
    (r"\sum_{i=1}^{n}i=\frac{n(n+1)}{2}", r"\sum_{i=0}^{n}i=\frac{n(n+1)}{2}", "lower_limit"),
    (r"\int_{0}^{1}x^{2}\,dx=\frac{1}{3}", r"\int_{0}^{2}x^{2}\,dx=\frac{1}{3}", "upper_limit"),
    (r"x=\frac{-b+\sqrt{b^{2}-4ac}}{2a}", r"x=\frac{-b+\sqrt{b^{2}+4ac}}{2a}", "radicand"),
    (r"f(x)=\frac{1}{1+e^{-x}}", r"f(x)=\frac{1}{1+e^{x}}", "missing_minus"),
    (
        r"\|x\|_{2}^{2}=\sum_{k=1}^{d}x_{k}^{2}",
        r"\|x\|_{2}=\sum_{k=1}^{d}x_{k}^{2}",
        "missing_exponent",
    ),
    (
        r"P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}",
        r"P(A\mid B)=\frac{P(B\mid A)P(A)}{P(A)}",
        "denominator",
    ),
    (r"\alpha_{i}+\beta_{j}=\gamma", r"\alpha_{i}+\beta_{i}=\gamma", "subscript"),
    (r"\prod_{j=1}^{m}(1+p_{j})", r"\prod_{j=1}^{m}(1-p_{j})", "product_term"),
    (r"a^{2}+b^{2}\geq 2ab", r"a^{2}+b^{2}\leq 2ab", "relation"),
    (r"\binom{n}{k}=\frac{n!}{k!(n-k)!}", r"\binom{n}{k}=\frac{n!}{k!(n+k)!}", "factorial_term"),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_pilot_inputs(cases: list[dict], input_root: Path) -> None:
    allowed = {"sample_id", "family_id", "source_image", "source_sha256", "prediction"}
    seen = set()
    for case in cases:
        if set(case) != allowed:
            raise ValueError("Unexpected input fields: reference/label/score metadata is forbidden")
        if case["sample_id"] in seen:
            raise ValueError("Duplicate input sample ID")
        seen.add(case["sample_id"])
        source = input_root / case["source_image"]
        if sha256_file(source) != case["source_sha256"]:
            raise ValueError("Source image hash mismatch")


def strip_math_wrappers(markup: str) -> str:
    markup = markup.strip()
    for left, right in [(r"\[", r"\]"), (r"\(", r"\)"), ("$$", "$$"), ("$", "$")]:
        if markup.startswith(left) and markup.endswith(right) and len(markup) >= len(left + right):
            return markup[len(left) : -len(right)].strip()
    return markup


def extract_formula(raw: str) -> tuple[str, str]:
    """Extract a whole response; do not silently select a formula from prose."""
    match = re.fullmatch(r"\s*<latex>(.*?)</latex>\s*", raw, flags=re.DOTALL)
    if match:
        return strip_math_wrappers(match[1]), "latex_tags"
    match = re.fullmatch(r"\s*```(?:latex|tex)?\s*\n(.*?)\n```\s*", raw, flags=re.DOTALL)
    if match:
        return strip_math_wrappers(match[1]), "code_fence"
    return strip_math_wrappers(raw), "unwrapped"


class MathTextRenderer:
    """Deterministic rasterization for the synthetic pilot's restricted syntax."""

    def __init__(self, output_root: Path, *, fontset: str = "cm", dpi: int = 160):
        self.output_root = output_root.resolve()
        self.fontset = fontset
        self.dpi = dpi

    def render(self, observation: Observation) -> Rendered:
        from matplotlib import mathtext, rc_context
        from PIL import Image, ImageChops, ImageOps

        if observation.modality != "formula":
            raise ValueError("MathText pilot renderer supports formulas only")
        formula = strip_math_wrappers(observation.prediction)
        if not formula or "$" in formula:
            raise ValueError("Empty or nested math delimiters")
        buffer = io.BytesIO()
        with rc_context({"mathtext.fontset": self.fontset, "font.size": 22}):
            mathtext.math_to_image(
                f"${formula}$", buffer, dpi=self.dpi, format="png", color="black"
            )
        image = Image.open(buffer).convert("RGB")
        bbox = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
        if bbox is None:
            raise ValueError("Empty rendered formula")
        image = ImageOps.expand(image.crop(bbox), border=16, fill="white")
        self.output_root.mkdir(parents=True, exist_ok=True)
        key = digest(f"{self.fontset}:{self.dpi}:{observation.prediction}")
        path = self.output_root / f"{key}.png"
        image.save(path)
        return Rendered(
            str(path), digest(observation.prediction), f"mathtext:{self.fontset}:{self.dpi}"
        )


def pixel_signature(path: Path) -> tuple[tuple[int, int], str]:
    """Evaluation-only exact raster proxy; not visual equivalence or CDM."""
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        return image.size, hashlib.sha256(image.tobytes()).hexdigest()


def make_prompt(prediction: str, with_render: bool) -> str:
    context = "Image 1 is the source formula."
    if with_render:
        context += " Image 2 is a rendering of the current OCR prediction."
    return (
        f"{context}\n"
        "Compare the source with the current OCR prediction. Correct only differences visible "
        "in the source image. If the current prediction is already correct, copy it unchanged. "
        "Preserve symbols, subscripts, superscripts, limits, and equation structure. "
        "Do not solve or simplify the equation.\n"
        f"Current OCR prediction:\n{prediction}\n"
        "Return only the complete LaTeX formula inside <latex>...</latex>."
    )
