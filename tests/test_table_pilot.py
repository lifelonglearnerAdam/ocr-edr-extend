import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ocr_edr.loop import Observation
from ocr_edr.table_pilot import HTMLTableRenderer, apply_table_action, parse_table


@unittest.skipUnless(importlib.util.find_spec("lxml"), "optional table dependency")
class BoundedTableTests(unittest.TestCase):
    initial = '<table><tr><td colspan="2">header</td></tr><tr><td>A</td><td>10</td></tr></table>'

    def test_single_cell_edit_preserves_spans_and_other_cells_and_escapes_text(self):
        candidate, _ = apply_table_action(
            self.initial,
            json.dumps({"action": "replace_cell", "row": 1, "cell": 1, "text": "<20>&"}),
        )
        tree, _ = parse_table(candidate)
        self.assertEqual(tree.xpath(".//td")[0].get("colspan"), "2")
        self.assertEqual([n.text for n in tree.xpath(".//td")], ["header", "A", "<20>&"])
        self.assertNotIn("<20>", candidate)

    def test_stop_preserves_original_bytes(self):
        candidate, _ = apply_table_action(self.initial, '{"action":"stop"}')
        self.assertEqual(candidate, self.initial)

    def test_span_edit_and_row_deletion(self):
        candidate, _ = apply_table_action(
            self.initial,
            '{"action":"set_span","row":0,"cell":0,"rowspan":1,"colspan":1}',
        )
        tree, _ = parse_table(candidate)
        self.assertIsNone(tree.xpath(".//td")[0].get("colspan"))
        candidate, _ = apply_table_action(self.initial, '{"action":"delete_row","row":1}')
        tree, _ = parse_table(candidate)
        self.assertEqual(len(tree.xpath(".//tr")), 1)

    def test_invalid_actions_do_not_mutate_initial(self):
        for action in [
            {"action": "replace_cell", "row": 9, "cell": 0, "text": "x"},
            {"action": "set_span", "row": 0, "cell": 0, "rowspan": 0, "colspan": 1},
            {"action": "replace_cell", "row": True, "cell": 0, "text": "x"},
            {"action": "stop", "reference": "hidden"},
            [{"action": "stop"}],
        ]:
            with self.subTest(action=action), self.assertRaises(ValueError):
                apply_table_action(self.initial, json.dumps(action))
        self.assertEqual(parse_table(self.initial)[1], self.initial)

    def test_prose_multiple_tables_external_content_and_truncation_are_rejected(self):
        for value in [
            "Here is " + self.initial,
            self.initial + self.initial,
            '<table><tr><td><img src="https://example.com/x"></td></tr></table>',
            '<table><tr><td rowspan="0">x</td></tr></table>',
            '<table><tr><td onclick="do_it()">x</td></tr></table>',
            "<table><tr><td>x",
        ]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_table(value)


@unittest.skipUnless(os.environ.get("OCR_EDR_TEST_TABLE") == "1", "explicit HTML rendering check")
class TableRenderingTests(unittest.TestCase):
    def test_real_merged_and_chinese_cells_render_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            renderer = HTMLTableRenderer(Path(tmp))
            obs = Observation(
                "table",
                "table",
                "",
                '<table><tr><td rowspan="2">公司</td><td>123</td></tr><tr><td>456</td></tr></table>',
            )
            first = renderer.render(obs)
            self.assertTrue(Path(first.path).is_file())
            self.assertEqual(renderer.render(obs), first)
            self.assertTrue(renderer.fonts)


if __name__ == "__main__":
    unittest.main()
