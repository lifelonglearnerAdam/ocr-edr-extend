import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_edr.formula_audit import audit_candidate, normalize_outer_environment


class FormulaAuditTests(unittest.TestCase):
    def test_whole_equation_wrapper_preserves_body(self):
        value, decision = normalize_outer_environment(
            r"\begin{equation} x_{1}=\frac{a+b}{c}\label{eq:1} \end{equation}"
        )
        self.assertEqual(value, r"x_{1}=\frac{a+b}{c}\label{eq:1}")
        self.assertEqual(decision, "strip_display_environment")

    def test_multiline_alignment_keeps_every_editable_token(self):
        value, _ = normalize_outer_environment(r"\begin{align*}a&=b\\c&=d\end{align*}")
        self.assertEqual(value, r"\begin{aligned}a&=b\\c&=d\end{aligned}")

    def test_partial_or_mismatched_wrappers_remain_unchanged(self):
        for value in [
            r"Here is \begin{equation}x=2\end{equation}",
            r"\begin{equation}x=2\end{align}",
            r"\begin{equation}x=2",
            r"\begin{document}x=2\end{document}",
        ]:
            with self.subTest(value=value):
                self.assertEqual(normalize_outer_environment(value), (value, "identity"))

    def test_strict_contract_rejects_renderable_coordinate_response(self):
        self.assertEqual(
            audit_candidate("(10,10),(988,988)", "require_latex_tags"),
            (None, "missing_latex_contract"),
        )

    def test_strict_contract_does_not_guess_from_prose(self):
        self.assertEqual(
            audit_candidate("Answer: <latex>x=1</latex>", "require_latex_tags"),
            (None, "missing_latex_contract"),
        )
        self.assertEqual(audit_candidate("<latex>x=1</latex>", "require_latex_tags")[0], "x=1")

    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):
            audit_candidate("x=1", "reference_selected")


if __name__ == "__main__":
    unittest.main()
