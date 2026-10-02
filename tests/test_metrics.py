import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_edr.metrics import compare_results, indexed_scores


def row(index, score, pred="wrong", reference="right"):
    return {
        "img_id": f"page-{index}.png",
        "gt_idx": [index],
        "gt_position": [index],
        "gt": reference,
        "pred": pred,
        "norm_gt": reference,
        "norm_pred": pred,
        "metric": {"CDM": score, "TEDS": score},
    }


def write_results(root, rows, page_mean):
    root.mkdir()
    for category in ["display_formula", "table"]:
        (root / f"fixture_{category}_result.json").write_text(json.dumps(rows))
    (root / "fixture_metric_result.json").write_text(
        json.dumps(
            {
                "display_formula": {"page": {"CDM": {"ALL": page_mean}}},
                "table": {"page": {"TEDS": {"ALL": page_mean}}},
            }
        )
    )


class MetricTests(unittest.TestCase):
    def test_page_and_sample_averages_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "before"
            write_results(root, [row(0, 1, "right"), row(1, 0)], 0.75)
            result = compare_results(root)
            formula = result["modalities"]["formula"]
            self.assertEqual(formula["sample_average_before"], 0.5)
            self.assertEqual(formula["official_page_average_before"], 0.75)
            self.assertEqual(formula["preserve_proxy"], 1)
            self.assertEqual(formula["bad_fix_proxy"], 0)
            self.assertIsNone(result["vis_fix"])

    def test_repair_and_regression_reported_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            before, after = Path(tmp) / "before", Path(tmp) / "after"
            write_results(before, [row(0, 1, "right"), row(1, 0)], 0.5)
            write_results(after, [row(0, 0, "wrong"), row(1, 1, "right")], 0.5)
            metrics = compare_results(before, after)["modalities"]["table"]
            self.assertEqual(metrics["preserve_proxy"], 0)
            self.assertEqual(metrics["regression_proxy"], 1)
            self.assertEqual(metrics["bad_fix_proxy"], 1)
            self.assertEqual(metrics["normalized_exact_fix"], 1)
            self.assertEqual(metrics["official_page_delta_pp"], 0)

    def test_changed_reference_matching_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            before, after = Path(tmp) / "before", Path(tmp) / "after"
            write_results(before, [row(0, 0)], 0)
            write_results(after, [row(1, 1)], 1)
            with self.assertRaisesRegex(ValueError, "reference matching changed"):
                compare_results(before, after)

    def test_duplicate_and_invalid_scores_fail(self):
        r = row(0, 1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            indexed_scores([r, copy.deepcopy(r)], "CDM")
        for value in [float("nan"), float("inf"), -1, 2, True]:
            with self.assertRaises(ValueError):
                indexed_scores([row(0, value)], "TEDS")

    def test_empty_good_branch_has_no_fabricated_preserve_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "before"
            write_results(root, [row(0, 0)], 0)
            metrics = compare_results(root)["modalities"]["formula"]
            self.assertIsNone(metrics["preserve_proxy"])


if __name__ == "__main__":
    unittest.main()
