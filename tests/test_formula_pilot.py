import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ocr_edr.formula_pilot import (
    MathTextRenderer,
    extract_formula,
    make_prompt,
    pixel_signature,
    sha256_file,
    validate_pilot_inputs,
)
from ocr_edr.loop import Observation


class PilotBoundaryTests(unittest.TestCase):
    def test_reference_and_score_fields_cannot_reach_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source.png").write_bytes(b"fixture")
            case = {
                "sample_id": "c001",
                "family_id": "f01",
                "source_image": "source.png",
                "source_sha256": sha256_file(root / "source.png"),
                "prediction": "x",
            }
            validate_pilot_inputs([case], root)
            for extra in ["reference", "variant", "score", "source_kind"]:
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    validate_pilot_inputs([{**case, extra: "hidden"}], root)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_pilot_inputs([{**case, "source_sha256": "wrong"}], root)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                validate_pilot_inputs([case, case], root)

    def test_prose_is_not_silently_replaced_with_an_embedded_answer(self):
        self.assertEqual(extract_formula("<latex>x^2</latex>"), ("x^2", "latex_tags"))
        raw = "Answer: <latex>x^2</latex> because it is correct."
        self.assertEqual(extract_formula(raw), (raw, "unwrapped"))

    def test_prompt_asks_for_image_fidelity_and_correct_input_preservation(self):
        prompt = make_prompt("E=mc^3", True)
        self.assertIn("E=mc^3", prompt)
        self.assertIn("copy it unchanged", prompt)
        self.assertIn("Do not solve or simplify", prompt)
        self.assertIn("Image 2", prompt)
        self.assertNotIn("Image 2", make_prompt("E=mc^3", False))


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "optional pilot dependencies")
class PilotRasterTests(unittest.TestCase):
    def test_whitespace_preserves_raster_but_exponent_error_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            renderer = MathTextRenderer(Path(tmp))
            paths = [
                Path(renderer.render(Observation("a", "formula", "", s)).path)
                for s in ["E=mc^2", " E=mc^2 ", "E=mc^3"]
            ]
            self.assertEqual(pixel_signature(paths[0]), pixel_signature(paths[1]))
            self.assertNotEqual(pixel_signature(paths[0]), pixel_signature(paths[2]))

    def test_malformed_edit_fails_rasterization(self):
        with tempfile.TemporaryDirectory() as tmp:
            renderer = MathTextRenderer(Path(tmp))
            with self.assertRaises(ValueError):
                renderer.render(Observation("a", "formula", "", r"\frac{a}"))


if __name__ == "__main__":
    unittest.main()
