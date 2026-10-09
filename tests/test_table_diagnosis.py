import hashlib
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from ocr_edr.table_diagnosis import (
        bind_diagnosis,
        diagnosis_from_action,
        diagnosis_prompt,
        displace_region,
        parse_diagnosis,
        verify_diagnosis_binding,
    )
except ImportError:
    diagnosis_from_action = parse_diagnosis = diagnosis_prompt = displace_region = (
        bind_diagnosis
    ) = verify_diagnosis_binding = None

TABLE = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>27.3</td></tr><tr><td>Total</td><td>100</td></tr></table>"


class TableDiagnosisTests(unittest.TestCase):
    def test_targets_contain_only_class_and_current_candidate_region_not_answer_text(self):
        self.assertTrue(callable(diagnosis_from_action))
        for action, error, region in [
            ({"action": "stop"}, None, None),
            (
                {"action": "replace_cell", "row": 1, "cell": 1, "text": "SECRET_CORRECTION"},
                "content",
                {"unit": "cell", "row": 1, "cell": 1},
            ),
            (
                {"action": "set_span", "row": 0, "cell": 0, "rowspan": 2, "colspan": 1},
                "structure",
                {"unit": "cell", "row": 0, "cell": 0},
            ),
            ({"action": "delete_row", "row": 2}, "extra", {"unit": "row", "row": 2}),
        ]:
            value = diagnosis_from_action(TABLE, action)
            self.assertEqual(
                value,
                {
                    "verdict": "valid" if action["action"] == "stop" else "invalid",
                    "error": error,
                    "region": region,
                },
            )
            self.assertNotIn("SECRET_CORRECTION", json.dumps(value))
            parse_diagnosis(TABLE, json.dumps(value))
        self.assertNotIn("SECRET_CORRECTION", diagnosis_prompt(TABLE))

    def test_invalid_output_duplicate_keys_wrong_indices_or_touching_unknown_regions_reject(self):
        self.assertTrue(callable(parse_diagnosis))
        for raw in [
            '{"verdict":"valid","verdict":"invalid","error":null,"region":null}',
            json.dumps({"verdict": "valid", "error": "content", "region": None}),
            json.dumps(
                {
                    "verdict": "invalid",
                    "error": "content",
                    "region": {"unit": "cell", "row": True, "cell": 1},
                }
            ),
            json.dumps(
                {
                    "verdict": "invalid",
                    "error": "content",
                    "region": {"unit": "cell", "row": 1, "cell": 9},
                }
            ),
            json.dumps(
                {"verdict": "invalid", "error": "unknown", "region": {"unit": "row", "row": 1}}
            ),
            json.dumps({"verdict": "valid", "error": None, "region": None, "text": "answer"}),
        ]:
            with self.assertRaises(ValueError):
                parse_diagnosis(TABLE, raw)

    def test_displacement_is_deterministic_different_legal_region_and_preserves_error_verdict(self):
        self.assertTrue(callable(displace_region))
        value = diagnosis_from_action(
            TABLE, {"action": "replace_cell", "row": 1, "cell": 1, "text": "17.3"}
        )
        shifted = displace_region(TABLE, value, identity="p1", seed=20261007)
        self.assertNotEqual(shifted["region"], value["region"])
        self.assertEqual(shifted["error"], value["error"])
        self.assertEqual(shifted, displace_region(TABLE, value, identity="p1", seed=20261007))
        parse_diagnosis(TABLE, json.dumps(shifted))
        valid = {"verdict": "valid", "error": None, "region": None}
        self.assertEqual(displace_region(TABLE, valid, identity="p1", seed=20261007), valid)

    def test_diagnosis_cannot_be_reused_after_candidate_or_source_changes(self):
        self.assertTrue(callable(bind_diagnosis))
        value = diagnosis_from_action(TABLE, {"action": "stop"})
        case = {"sample_id": "p1", "prediction": TABLE, "source_sha256": "a" * 64}
        bound = bind_diagnosis(case, value, producer="model", output_sha256="b" * 64)
        verify_diagnosis_binding(case, bound)
        for changed in [
            {**case, "prediction": TABLE.replace("27.3", "17.3")},
            {**case, "source_sha256": "c" * 64},
            {**case, "sample_id": "p2"},
        ]:
            with self.assertRaises(ValueError):
                verify_diagnosis_binding(changed, bound)

    def test_optional_hint_is_bound_and_original_messages_are_unchanged(self):
        from ocr_edr.table_sft_screen import adapt_table_call, table_messages

        self.assertTrue(callable(bind_diagnosis))
        case = {
            "sample_id": "p1",
            "family_id": "p1",
            "source_image": "one.png",
            "source_sha256": "a" * 64,
            "prediction": TABLE,
        }
        diagnosis = bind_diagnosis(
            case,
            diagnosis_from_action(TABLE, {"action": "stop"}),
            producer="model",
            output_sha256="b" * 64,
        )
        original, plain = table_messages(TABLE)
        messages, prompt = table_messages(TABLE, diagnosis=diagnosis, case=case)
        self.assertIn("advisory", prompt)
        self.assertIn("candidate_sha256", prompt)
        self.assertEqual(table_messages(TABLE), (original, plain))
        call = {
            **case,
            "prompt": prompt,
            "messages": messages,
            "condition": "all",
            "ordered_image_sha256": ["a" * 64],
            "raw_output": '{"action":"stop"}',
            "input_tokens": 10,
            "output_tokens": 6,
            "generation_seconds": 0.1,
        }
        with self.assertRaises(ValueError):
            adapt_table_call(case, call, renderer=lambda o: {}, max_new_tokens=192)
        final = adapt_table_call(
            case, call, renderer=lambda o: {}, max_new_tokens=192, diagnosis=diagnosis
        )
        self.assertEqual(final["final_prediction"], TABLE)
        self.assertEqual(final["trace"][0]["prompt"], prompt)
        self.assertEqual(hashlib.sha256(TABLE.encode()).hexdigest(), diagnosis["candidate_sha256"])


if __name__ == "__main__":
    unittest.main()
