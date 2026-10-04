import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_formula_agreement import replay_agreement

from ocr_edr.loop import Rendered


class FakeRenderer:
    def render(self, observation):
        if observation.prediction == "unrenderable":
            raise ValueError("Unsupported syntax")
        return Rendered(observation.prediction, "hash", "fake")


class AgreementAuditTests(unittest.TestCase):
    @staticmethod
    def predictions(values):
        rows = [
            {
                "sample_id": "a",
                "arm": "unchanged_0",
                "initial_prediction": "correct_initial",
                "final_prediction": "correct_initial",
                "trace": [],
            }
        ]
        for arm, value in zip(["source_only", "source_first", "source_last"], values):
            rows.append(
                {
                    "sample_id": "a",
                    "arm": arm,
                    "initial_prediction": "correct_initial",
                    "final_prediction": value,
                    "trace": [{"generation_seconds": 2}],
                }
            )
        return rows

    def test_shared_wrong_answer_is_accepted_and_all_calls_are_charged(self):
        with patch("audit_formula_agreement.pixel_signature", side_effect=lambda p: str(p)):
            rows, decisions = replay_agreement(
                self.predictions(["same_wrong", "same_wrong", "same_wrong"]), FakeRenderer()
            )
        self.assertTrue(all(d["agreement"] for d in decisions))
        self.assertTrue(all(row["final_prediction"] == "same_wrong" for row in rows))
        self.assertEqual([len(row["trace"]) for row in rows], [2, 2, 3])

    def test_failed_renders_do_not_form_agreement(self):
        with patch("audit_formula_agreement.pixel_signature", side_effect=lambda p: str(p)):
            rows, decisions = replay_agreement(
                self.predictions(["unrenderable"] * 3), FakeRenderer()
            )
        self.assertFalse(any(d["agreement"] for d in decisions))
        self.assertTrue(all(row["final_prediction"] == "correct_initial" for row in rows))
        self.assertEqual([len(row["trace"]) for row in rows], [2, 2, 3])

    def test_disagreement_preserves_baseline_without_losing_rejected_call_costs(self):
        with patch("audit_formula_agreement.pixel_signature", side_effect=lambda p: str(p)):
            rows, decisions = replay_agreement(
                self.predictions(["one", "two", "three"]), FakeRenderer()
            )
        self.assertFalse(any(d["agreement"] for d in decisions))
        self.assertTrue(all(row["final_prediction"] == "correct_initial" for row in rows))
        self.assertEqual(sum(c["generation_seconds"] for c in rows[-1]["trace"]), 6)


if __name__ == "__main__":
    unittest.main()
