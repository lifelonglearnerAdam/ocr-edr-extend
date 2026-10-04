import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ocr_edr.formula_pilot import pixel_signature
from ocr_edr.loop import Observation
from ocr_edr.tex import TectonicRenderer, validate_formula


class FormulaBoundaryTests(unittest.TestCase):
    def test_formula_input_cannot_change_the_tex_document_or_io(self):
        for text in [
            r"\input{private}",
            r"\write18{command}",
            r"\csname input\endcsname",
            r"\end{document}",
            r"\documentclass{article}",
            "",
        ]:
            with self.assertRaises(ValueError):
                validate_formula(text)
        self.assertEqual(validate_formula(r"\left(x\right)"), r"\left(x\right)")
        self.assertEqual(validate_formula("$x^2$"), "x^2")


@unittest.skipUnless(
    os.environ.get("OCR_EDR_TEST_TECTONIC") == "1", "explicit TeX integration check"
)
class TectonicIntegrationTests(unittest.TestCase):
    def test_real_matrix_support_and_pixel_equivalent_bracing(self):
        with tempfile.TemporaryDirectory() as tmp:
            renderer = TectonicRenderer(Path(tmp))
            a = renderer.render(Observation("a", "formula", "", "E=mc^2"))
            b = renderer.render(Observation("b", "formula", "", "E=mc^{2}"))
            self.assertEqual(pixel_signature(Path(a.path)), pixel_signature(Path(b.path)))
            matrix = renderer.render(
                Observation("m", "formula", "", r"\begin{pmatrix}a&b\\c&d\end{pmatrix}")
            )
            self.assertTrue(Path(matrix.path).is_file())
            self.assertEqual(renderer.render(Observation("a", "formula", "", "E=mc^2")), a)


if __name__ == "__main__":
    unittest.main()
