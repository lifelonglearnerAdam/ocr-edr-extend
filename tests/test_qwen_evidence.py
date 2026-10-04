import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ocr_edr.formula_pilot import make_prompt
from ocr_edr.qwen import EVIDENCE_MODES, evidence_message


class EvidenceMessageTests(unittest.TestCase):
    def test_original_message_and_image_order_are_preserved(self):
        paths, messages, prompt = evidence_message(
            Path("source"), "x^3", Path("render"), "source_first"
        )
        self.assertEqual(paths, [Path("source"), Path("render")])
        self.assertEqual(prompt, make_prompt("x^3", True))
        self.assertEqual(
            messages[0]["content"],
            [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        )

    def test_reversed_image_roles_follow_the_reversed_processor_order(self):
        paths, _, prompt = evidence_message(Path("source"), "x^3", Path("render"), "source_last")
        self.assertEqual(paths, [Path("render"), Path("source")])
        self.assertIn("Image 1 is a rendering", prompt)
        self.assertIn("Image 2 is the source", prompt)

    def test_role_labels_and_duplicate_control_match_their_images(self):
        paths, messages, _ = evidence_message(
            Path("source"), "x^3", Path("render"), "labeled_source_last"
        )
        self.assertEqual(paths, [Path("render"), Path("source")])
        labels = [block["text"] for block in messages[0]["content"] if block["type"] == "text"]
        self.assertEqual(labels[:2], ["CURRENT OCR RENDERING:", "SOURCE FORMULA:"])
        paths, _, _ = evidence_message(Path("source"), "x^3", None, "duplicate_source")
        self.assertEqual(paths, [Path("source"), Path("source")])

    def test_evidence_mode_requires_its_candidate_image(self):
        for mode in EVIDENCE_MODES:
            if mode in {"source_only", "duplicate_source"}:
                continue
            with self.assertRaises(ValueError):
                evidence_message(Path("source"), "x^3", None, mode)
        with self.assertRaises(ValueError):
            evidence_message(Path("source"), "x^3", None, "oracle")


if __name__ == "__main__":
    unittest.main()
