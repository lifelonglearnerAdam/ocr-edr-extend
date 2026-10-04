import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_table_pilot import message

from ocr_edr import table_pilot


class TablePromptPreservationTests(unittest.TestCase):
    def test_original_patch_prompt_matches_every_archived_call(self):
        root = ROOT / "experiments/artifacts/table-development-20261004"
        inputs = {
            r["sample_id"]: r
            for r in map(json.loads, (root / "data/inputs.jsonl").read_text().splitlines())
        }
        outputs = [
            r
            for r in map(json.loads, (root / "repair/predictions.jsonl").read_text().splitlines())
            if r["arm"] == "source_only_patch"
        ]
        self.assertEqual(len(outputs), 14)
        for row in outputs:
            case = inputs[row["sample_id"]]
            source = Path(case["source_image"])
            paths, messages, prompt = message(source, case["prediction"], None, row["arm"])
            with self.subTest(sample=row["sample_id"]):
                self.assertEqual(paths, [source])
                self.assertEqual(prompt, row["trace"][0]["prompt"])
                self.assertEqual(messages, row["trace"][0]["messages"])

    def test_unknown_condition_cannot_silently_select_full_rewrite(self):
        with self.assertRaisesRegex(ValueError, "Unknown table evidence mode"):
            message(Path("source.png"), "<table><tr><td>x</td></tr></table>", None, "typo")


@unittest.skipUnless(importlib.util.find_spec("lxml"), "optional table dependency")
class TableAddressTests(unittest.TestCase):
    initial = (
        '<table><thead><tr><th colspan="2">标题</th></tr></thead><tbody>'
        '<tr><td rowspan="2"><b>公司</b>A</td><td>10&amp;</td></tr>'
        "<tr><td>20</td></tr></tbody></table>"
    )

    def test_dom_addresses_select_the_right_cell_under_header_and_row_spans(self):
        self.assertTrue(callable(getattr(table_pilot, "table_cell_map", None)), "cell map missing")
        entries = table_pilot.table_cell_map(self.initial)
        self.assertEqual(
            entries,
            [
                {"row": 0, "cell": 0, "tag": "th", "rowspan": 1, "colspan": 2, "text": "标题"},
                {"row": 1, "cell": 0, "tag": "td", "rowspan": 2, "colspan": 1, "text": "公司A"},
                {"row": 1, "cell": 1, "tag": "td", "rowspan": 1, "colspan": 1, "text": "10&"},
                {"row": 2, "cell": 0, "tag": "td", "rowspan": 1, "colspan": 1, "text": "20"},
            ],
        )
        target = entries[3]
        changed, _ = table_pilot.apply_table_action(
            self.initial,
            json.dumps(
                {
                    "action": "replace_cell",
                    "row": target["row"],
                    "cell": target["cell"],
                    "text": "30",
                }
            ),
        )
        tree, _ = table_pilot.parse_table(changed)
        self.assertEqual(
            ["".join(c.itertext()) for c in tree.xpath(".//td|.//th")],
            ["标题", "公司A", "10&", "30"],
        )
        self.assertEqual(tree.xpath(".//td")[0].get("rowspan"), "2")

    def test_indexed_condition_adds_only_the_initial_cell_map_to_the_prompt(self):
        source = Path("source.png")
        original = message(source, self.initial, None, "source_only_patch")
        indexed = message(source, self.initial, None, "source_only_patch_indexed")
        self.assertEqual(indexed[0], original[0])
        self.assertIn("<cell_map>\n", indexed[2])
        opening, body = indexed[2].split("<cell_map>\n", 1)
        payload, closing = body.split("\n</cell_map>\n", 1)
        entries = json.loads(payload)
        self.assertEqual(entries[3]["text"], "20")
        self.assertEqual((entries[3]["row"], entries[3]["cell"]), (2, 0))
        self.assertEqual(opening.split("Cell address map", 1)[0] + closing, original[2])
        self.assertEqual(indexed[1][0]["content"][-1]["text"], indexed[2])


if __name__ == "__main__":
    unittest.main()
