import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from ocr_edr.pubtabnet import annotation_html, bounded_supervision, document_id, document_role
except ImportError:
    annotation_html = bounded_supervision = document_id = document_role = None


class DocumentRolesTests(unittest.TestCase):
    def test_published_heldout_document_never_enters_optimizer_roles(self):
        self.assertTrue(callable(document_role))
        self.assertIsNone(document_role("PMC123", "train", held_out_documents={"PMC123"}))
        self.assertIsNone(
            document_role("PMC123", "val", held_out_documents={"PMC123"}, test_documents={"PMC123"})
        )
        self.assertIn(
            document_role("PMC123", "train", held_out_documents=set()), {"train", "model_dev"}
        )
        self.assertIn(
            document_role("PMC123", "val", held_out_documents={"PMC123"}),
            {"gate_calibration", "locked_evaluation"},
        )
        self.assertIsNone(document_role("PMC123", "test", held_out_documents={"PMC123"}))
        with self.assertRaises(ValueError):
            document_role("unknown", "train", held_out_documents=set())

    def test_all_tables_in_a_document_have_same_role(self):
        self.assertTrue(callable(document_id))
        ids = [document_id(filename) for filename in ["PMC123_004_00.png", "PMC123_008_02.png"]]
        self.assertEqual(ids, ["PMC123", "PMC123"])
        for invalid in ["other_004_00.png", "../PMC123_004_00.png"]:
            with self.assertRaises(ValueError):
                document_id(invalid)


@unittest.skipUnless(importlib.util.find_spec("lxml"), "optional HTML dependency")
class AnnotationTests(unittest.TestCase):
    def test_text_characters_escaped_inline_markup_and_spans_preserved(self):
        self.assertTrue(callable(annotation_html))
        data = {
            "structure": {
                "tokens": [
                    "<thead>",
                    "<tr>",
                    "<td",
                    ' colspan="2"',
                    ">",
                    "</td>",
                    "</tr>",
                    "</thead>",
                ]
            },
            "cells": [{"tokens": ["<b>", "A", " ", "&", " ", "<", " ", "9", "</b>"]}],
        }
        text = annotation_html(data)
        self.assertIn('<td colspan="2"><b>A &amp; &lt; 9</b></td>', text)
        self.assertEqual(text.count("<td"), 1)

    def test_cell_coverage_and_unsafe_markup_fail(self):
        self.assertTrue(callable(annotation_html))
        for data in [
            {"structure": {"tokens": ["<tr><td></td></tr>"]}, "cells": []},
            {
                "structure": {"tokens": ["<tr><td></td></tr>"]},
                "cells": [{"tokens": ["x"]}, {"tokens": ["y"]}],
            },
            {
                "structure": {"tokens": ["<tr><td></td></tr>"]},
                "cells": [{"tokens": ["<script>", "x", "</script>"]}],
            },
        ]:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    annotation_html(data)

    def test_local_edit_targets_restore_reference_and_keep_is_stop(self):
        self.assertTrue(callable(bounded_supervision))
        import json

        from ocr_edr.table_pilot import apply_table_action, parse_table

        ref = '<table><tr><td colspan="2">Label</td></tr><tr><td>19</td><td>8</td></tr></table>'
        canonical = parse_table(ref)[1]
        rows = bounded_supervision(ref)
        self.assertEqual(rows[0]["variant"], "preservation")
        self.assertEqual(rows[0]["target_action"], {"action": "stop"})
        self.assertIn("span_perturbation", {r["variant"] for r in rows})
        self.assertIn("extra_row", {r["variant"] for r in rows})
        for row in rows:
            final, _ = apply_table_action(row["prediction"], json.dumps(row["target_action"]))
            self.assertEqual(final, canonical)
        with self.assertRaises(ValueError):
            bounded_supervision("<table><tr><td>letters</td></tr></table>")


if __name__ == "__main__":
    unittest.main()
