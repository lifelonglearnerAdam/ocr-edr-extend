import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_formula_pilot import evaluate


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "optional pilot dependencies")
class PilotEvaluationTests(unittest.TestCase):
    @staticmethod
    def result(sample_id, arm, initial, final):
        return {
            "sample_id": sample_id,
            "arm": arm,
            "initial_prediction": initial,
            "final_prediction": final,
            "trace": [],
        }

    def test_repairs_and_correct_input_regressions_are_reported_together(self):
        references = [
            {
                "sample_id": "bad",
                "family_id": "f1",
                "reference": "x^2",
                "variant": "incorrect",
                "source_kind": "base",
                "error_type": "exponent",
            },
            {
                "sample_id": "good",
                "family_id": "f2",
                "reference": "y^2",
                "variant": "correct",
                "source_kind": "base",
                "error_type": None,
            },
        ]
        results = [
            self.result("bad", "repair", "x^3", "x^2"),
            self.result("good", "repair", "y^2", "y^3"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            _, summary = evaluate(results, references, Path(tmp))
        self.assertEqual(summary[0]["bad_fixed_proxy"], 1)
        self.assertEqual(summary[0]["good_regressions_proxy"], 1)
        self.assertEqual(summary[0]["full_set_raster_exact_proxy"], 0.5)

    def test_unpaired_and_duplicate_condition_outputs_are_rejected(self):
        references = [
            {
                "sample_id": name,
                "family_id": name,
                "reference": "x^2",
                "variant": "correct",
                "source_kind": "base",
                "error_type": None,
            }
            for name in ["a", "b"]
        ]
        results = [
            self.result("a", "source", "x^2", "x^2"),
            self.result("b", "source", "x^2", "x^2"),
            self.result("a", "render", "x^2", "x^2"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Unpaired"):
                evaluate(results, references, Path(tmp))
            with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
                evaluate([results[0], results[0]], references, Path(tmp))


if __name__ == "__main__":
    unittest.main()
