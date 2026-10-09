import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_native_repair import prepare


class NativeRepairTests(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {
                "family_id": name,
                "source_image": f"images/{name}.png",
                "source_sha256": name,
                "reference": "x^2",
                "split": "spe",
            }
            for name in ["a", "b"]
        ]
        self.predictions = [
            {"family_id": name, "source_sha256": name, "prediction": "x^3"} for name in ["a", "b"]
        ]

    def test_preparation_keeps_every_source_and_separates_references(self):
        inputs, references = prepare(self.sources, self.predictions)
        self.assertEqual(len(inputs), 2)
        self.assertEqual([r["prediction"] for r in inputs], ["x^3", "x^3"])
        for row in inputs:
            self.assertNotIn("reference", row)
            self.assertNotIn("variant", row)
        self.assertEqual([r["reference"] for r in references], ["x^2", "x^2"])

    def test_partial_or_mismatched_native_predictions_cannot_be_paired(self):
        with self.assertRaises(ValueError):
            prepare(self.sources, self.predictions[:1])
        with self.assertRaises(ValueError):
            prepare(self.sources, self.predictions + self.predictions[:1])
        with self.assertRaises(ValueError):
            prepare(self.sources, [{**r, "source_sha256": "wrong"} for r in self.predictions])


if __name__ == "__main__":
    unittest.main()
