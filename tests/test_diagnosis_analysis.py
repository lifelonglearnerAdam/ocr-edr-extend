import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from ocr_edr.diagnosis_analysis import diagnosis_summary, paired_component_cost
except ImportError:
    diagnosis_summary = paired_component_cost = None


class DiagnosisAnalysisTests(unittest.TestCase):
    def test_actual_predictions_recompute_type_location_and_unknown_counts(self):
        self.assertTrue(callable(diagnosis_summary))
        bad = {
            "verdict": "invalid",
            "error": "content",
            "region": {"unit": "cell", "row": 1, "cell": 1},
        }
        good = {"verdict": "valid", "error": None, "region": None}
        rows = [
            {
                "sample_id": "good",
                "variant": "preservation",
                "target": good,
                "prediction": bad,
                "strict_joint": True,
            },
            {
                "sample_id": "bad",
                "variant": "cell_perturbation",
                "target": bad,
                "prediction": {**bad, "region": {"unit": "cell", "row": 1, "cell": 0}},
                "strict_joint": True,
            },
            {
                "sample_id": "unknown",
                "variant": "cell_perturbation",
                "target": bad,
                "prediction": None,
                "strict_joint": True,
            },
        ]
        result = diagnosis_summary(rows, expected_ids={"good", "bad", "unknown"})[0]
        self.assertEqual(result["cases"], 3)
        self.assertEqual(result["strict_joint"], 0)
        self.assertEqual(result["verdict_correct"], 1)
        self.assertEqual(result["valid_false_positive"], 1)
        self.assertEqual(result["invalid_type_correct"], 1)
        self.assertEqual(result["invalid_region_correct"], 0)
        self.assertEqual(result["invalid_unknown"], 1)
        with self.assertRaises(ValueError):
            diagnosis_summary(rows[:-1], expected_ids={"good", "bad", "unknown"})
        with self.assertRaises(ValueError):
            diagnosis_summary(rows + [rows[0]], expected_ids={"good", "bad", "unknown"})

    def test_guided_pipeline_charges_both_calls_including_failed_diagnoses(self):
        self.assertTrue(callable(paired_component_cost))
        refinement = [
            {
                "sample_id": "a",
                "model_calls": 1,
                "input_tokens": 100,
                "output_tokens": 6,
                "generation_seconds": 0.5,
                "render_attempts": 1,
                "render_seconds": 0.2,
            },
            {
                "sample_id": "b",
                "model_calls": 1,
                "input_tokens": 200,
                "output_tokens": 192,
                "generation_seconds": 5.0,
                "render_attempts": 0,
                "render_seconds": 0.0,
            },
        ]
        diagnoses = [
            {"sample_id": "a", "input_tokens": 80, "output_tokens": 22, "generation_seconds": 0.7},
            {"sample_id": "b", "input_tokens": 90, "output_tokens": 192, "generation_seconds": 4.0},
        ]
        result = paired_component_cost(refinement, diagnoses, use_diagnosis=True)
        self.assertEqual(result["model_calls"], 4)
        self.assertEqual(result["input_tokens"], 470)
        self.assertEqual(result["output_tokens"], 412)
        self.assertAlmostEqual(result["generation_seconds"], 10.2)
        self.assertEqual(result["render_attempts"], 1)
        oracle = paired_component_cost(refinement, diagnoses, use_diagnosis=False)
        self.assertEqual(oracle["model_calls"], 2)
        self.assertEqual(oracle["input_tokens"], 300)
        with self.assertRaises(ValueError):
            paired_component_cost(refinement, diagnoses[:1], use_diagnosis=True)
        with self.assertRaises(ValueError):
            paired_component_cost(
                refinement, [{**diagnoses[0], "input_tokens": -1}, diagnoses[1]], use_diagnosis=True
            )


if __name__ == "__main__":
    unittest.main()
