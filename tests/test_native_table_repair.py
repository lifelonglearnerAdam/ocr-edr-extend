import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from ocr_edr.native_table_repair import (
        native_model_identity,
        native_repair_inputs,
        verify_native_calls,
    )
except ImportError:
    native_repair_inputs = verify_native_calls = None
    native_model_identity = None


@unittest.skipUnless(importlib.util.find_spec("lxml"), "optional table parser")
class NativeRepairIdentityTests(unittest.TestCase):
    def test_all_sources_retained_and_only_five_model_input_fields_emitted(self):
        self.assertTrue(callable(native_repair_inputs))
        sources = [{"family_id": "p1", "source_image": "one.png", "source_sha256": "a" * 64}]
        prediction = {
            **sources[0],
            "prediction": "<table><tr><td>17</td></tr></table>",
            "parser_error": None,
            "details": {"private": "not model input"},
        }
        result = native_repair_inputs([prediction], sources, role="model_dev")
        self.assertEqual(result[0]["sample_id"], "p1-native")
        self.assertEqual(
            set(result[0]),
            {"sample_id", "family_id", "source_image", "source_sha256", "prediction"},
        )
        self.assertNotIn("details", result[0])

    def test_missing_duplicate_source_drift_or_heldout_role_cannot_be_used(self):
        self.assertTrue(callable(native_repair_inputs))
        source = {"family_id": "p1", "source_image": "one.png", "source_sha256": "a" * 64}
        prediction = {
            **source,
            "prediction": "<table><tr><td>17</td></tr></table>",
            "parser_error": None,
        }
        for rows, role in [
            ([], "model_dev"),
            ([prediction, prediction], "model_dev"),
            ([{**prediction, "source_sha256": "b" * 64}], "model_dev"),
            ([prediction], "locked"),
            ([{**prediction, "prediction": None, "parser_error": "failure"}], "model_dev"),
        ]:
            with self.assertRaises(ValueError):
                native_repair_inputs(rows, [source], role=role)

    def test_frozen_calls_preserve_complete_input_order_identity_and_condition(self):
        self.assertTrue(callable(verify_native_calls))
        inputs = [
            {
                "sample_id": str(i),
                "family_id": str(i),
                "source_image": str(i) + ".png",
                "source_sha256": str(i) * 64,
                "prediction": "<table></table>",
            }
            for i in [1, 2]
        ]
        calls = [
            {**r, "condition": "all", "ordered_image_sha256": [r["source_sha256"]]} for r in inputs
        ]
        verify_native_calls(calls, inputs, "all")
        variants = [
            calls[:1],
            calls[::-1],
            [calls[0], calls[0]],
            [{**calls[0], "prediction": "changed"}, calls[1]],
            [{**calls[0], "condition": "base"}, calls[1]],
            [{**calls[0], "ordered_image_sha256": ["different"]}, calls[1]],
        ]
        for variant in variants:
            with self.assertRaises(ValueError):
                verify_native_calls(variant, inputs, "all")

    def test_historical_missing_model_hash_requires_an_explicit_bound_audit(self):
        self.assertTrue(callable(native_model_identity))
        old = {"condition": "all"}
        with self.assertRaises(ValueError):
            native_model_identity(old, "old-receipt")
        audit = {
            "status": "audit_completed",
            "historical_base_receipt_hash_missing": True,
            "current_model_files_and_controller_path_verified": True,
            "current_model_receipt_sha256": "model-hash",
            "evidence_sha256": {"all_receipt": "old-receipt"},
        }
        self.assertEqual(native_model_identity(old, "old-receipt", audit), "model-hash")
        with self.assertRaises(ValueError):
            native_model_identity(old, "changed-receipt", audit)
        self.assertEqual(native_model_identity({"model_receipt_sha256": "new"}, "r"), "new")


if __name__ == "__main__":
    unittest.main()
