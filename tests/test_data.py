import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_edr.data import assert_trainable, load_manifest, observation


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "a.png").write_bytes(b"synthetic source a")
        (self.root / "b.png").write_bytes(b"synthetic source b")

    def record(self, name="a", **kwargs):
        return {
            "sample_id": name,
            "page_id": name,
            "benchmark": "synthetic",
            "modality": "formula",
            "split": "test",
            "test_only": True,
            "source_image": f"{name}.png",
            "prediction": "x^3",
            "reference": "x^2",
            **kwargs,
        }

    def load(self, rows, name="manifest.jsonl"):
        path = self.root / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return load_manifest(path)

    def test_reference_excluded_from_policy_input(self):
        rows = self.load([self.record()])
        obs = observation(rows[0])
        self.assertEqual(obs.prediction, "x^3")
        self.assertFalse(hasattr(obs, "reference"))
        self.assertEqual(len(rows[0]["source_image_sha256"]), 64)

    def test_test_only_metadata_required(self):
        for kwargs in [{"test_only": False}, {"test_only": None}, {"split": "train"}]:
            with self.assertRaises(ValueError):
                self.load([self.record(**kwargs)])

    def test_duplicate_and_escaping_paths_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load([self.record(), self.record()])
        with self.assertRaises(ValueError):
            self.load([self.record(source_image="../outside.png")])

    def test_distinct_training_and_holdout_pass(self):
        train = self.load([self.record("a", split="train", test_only=False)], "train.jsonl")
        held = self.load([self.record("b")], "held.jsonl")
        assert_trainable(train, held)

    def test_same_parent_page_rejected_even_with_different_crops(self):
        train = self.load([self.record("a", split="train", test_only=False, page_id="parent")])
        held = self.load([self.record("b", page_id="parent")])
        with self.assertRaisesRegex(ValueError, "overlaps"):
            assert_trainable(train, held)

    def test_same_image_rejected_under_different_page_names(self):
        train = self.load([self.record("a", split="train", test_only=False)])
        held = self.load([self.record("b", source_image="a.png")])
        with self.assertRaisesRegex(ValueError, "overlaps"):
            assert_trainable(train, held)


if __name__ == "__main__":
    unittest.main()
