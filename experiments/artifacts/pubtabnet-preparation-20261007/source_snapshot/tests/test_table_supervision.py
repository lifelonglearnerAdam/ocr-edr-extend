import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from ocr_edr.table_supervision import table_action_prompt, validate_table_target
except ImportError:
    table_action_prompt = validate_table_target = None


@unittest.skipUnless(importlib.util.find_spec("lxml"), "optional table dependency")
class TableSupervisionTests(unittest.TestCase):
    def test_current_html_addresses_are_all_exposed_without_correct_answers(self):
        self.assertTrue(callable(table_action_prompt))
        prompt = table_action_prompt("<table><tr><td>wrong 19</td><td>7</td></tr></table>")
        self.assertIn("wrong 19", prompt)
        self.assertIn('"cell":1', prompt)
        self.assertNotIn("ground_truth", prompt)
        self.assertNotIn("correct plain cell text", prompt)
        self.assertIn("request no external text", prompt)

    def test_keep_and_edit_targets_replayed_independently_of_prompt(self):
        self.assertTrue(callable(validate_table_target))
        initial = "<table><tr><td>19</td><td>7</td></tr></table>"
        self.assertEqual(
            validate_table_target(initial, '{"action":"stop"}', initial), {"action": "stop"}
        )
        reference = "<table><tr><td>18</td><td>7</td></tr></table>"
        good = '{"action":"replace_cell","row":0,"cell":0,"text":"18"}'
        self.assertEqual(validate_table_target(initial, good, reference)["text"], "18")
        for bad in [
            '{"action":"replace_cell","row":0,"cell":1,"text":"18"}',
            '{"action":"stop"}',
            '{"action":"stop","reference":"18"}',
        ]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    validate_table_target(initial, bad, reference)


if __name__ == "__main__":
    unittest.main()
