import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_edr.formula_pilot import make_prompt

try:
    from ocr_edr.sft_screen import adapt_screen_calls
except ImportError:
    adapt_screen_calls = None


class Renderer:
    def render(self, observation):
        if observation.prediction == r"\frac{":
            raise ValueError("malformed formula")
        return None


class ScreenTests(unittest.TestCase):
    def input(self):
        return {
            "sample_id": "one",
            "family_id": "family",
            "source_image": "source.png",
            "source_sha256": "a" * 64,
            "prediction": "x^2",
        }

    def call(self, raw, condition="base", **kwargs):
        return {
            **self.input(),
            "condition": condition,
            "raw_output": raw,
            "prompt": make_prompt("x^2", False),
            "ordered_image_sha256": ["a" * 64],
            "output_tokens": 20,
            "hit_token_cap": False,
            **kwargs,
        }

    def test_correct_syntax_can_still_accept_harmful_edit(self):
        self.assertTrue(callable(adapt_screen_calls))
        result = adapt_screen_calls(
            [self.input()],
            {"base": [self.call("<latex>x^3</latex>")]},
            renderer=Renderer(),
            max_new_tokens=256,
        )
        self.assertEqual(result[0]["arm"], "unchanged_0")
        self.assertEqual(result[0]["final_prediction"], "x^2")
        self.assertEqual(result[1]["final_prediction"], "x^3")
        self.assertEqual(result[1]["trace"][0]["decision"], "accepted_syntax_only")

    def test_rejected_raw_output_and_cost_are_retained(self):
        self.assertTrue(callable(adapt_screen_calls))
        for raw, kwargs in [
            ("Here is x^3", {}),
            (r"<latex>\frac{</latex>", {}),
            ("<latex>x^3</latex>", {"output_tokens": 256, "hit_token_cap": True}),
        ]:
            result = adapt_screen_calls(
                [self.input()],
                {"base": [self.call(raw, **kwargs)]},
                renderer=Renderer(),
                max_new_tokens=256,
            )
            final = result[1]
            self.assertEqual(final["final_prediction"], "x^2")
            self.assertEqual(final["trace"][0]["raw_output"], raw)
            self.assertEqual(final["trace"][0]["output_tokens"], kwargs.get("output_tokens", 20))

    def test_coverage_drift_reference_access_and_cost_drift_rejected(self):
        self.assertTrue(callable(adapt_screen_calls))
        invalid = [
            [],
            [self.call("<latex>x</latex>")] * 2,
            [self.call("<latex>x</latex>", reference="hidden")],
            [self.call("<latex>x</latex>", prediction="y")],
            [self.call("<latex>x</latex>", family_id="other")],
            [self.call("<latex>x</latex>", ordered_image_sha256=["b" * 64])],
            [self.call("<latex>x</latex>", output_tokens=256, hit_token_cap=False)],
        ]
        for calls in invalid:
            with self.subTest(calls=calls):
                with self.assertRaises(ValueError):
                    adapt_screen_calls(
                        [self.input()], {"base": calls}, renderer=Renderer(), max_new_tokens=256
                    )


if __name__ == "__main__":
    unittest.main()
