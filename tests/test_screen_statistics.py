import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from ocr_edr.screen_statistics import paired_source_bootstrap, variant_summary
except ImportError:
    paired_source_bootstrap = variant_summary = None


class StatisticsTests(unittest.TestCase):
    def row(self, sample, family, arm, before, after, variant="native_parser_prediction"):
        return {
            "sample_id": sample,
            "family_id": family,
            "arm": arm,
            "variant": variant,
            "initial_cdm": {"F1_score": before},
            "final_cdm": {"F1_score": after},
            "delta_cdm": after - before,
            "changed": before != after,
        }

    def test_family_mean_not_repeated_crop_weight(self):
        self.assertTrue(callable(paired_source_bootstrap))
        rows = []
        for sample, family, value in [("a1", "a", 1), ("a2", "a", 1), ("b1", "b", 0)]:
            rows.extend(
                [self.row(sample, family, "base", 0, 0), self.row(sample, family, "all", 0, value)]
            )
        result = paired_source_bootstrap(rows, "all", "base", seed=1, repeats=100)
        self.assertEqual(result["source_groups"], 2)
        self.assertEqual(result["mean_delta_vs_comparator"], 0.5)
        self.assertEqual(result, paired_source_bootstrap(rows, "all", "base", seed=1, repeats=100))

    def test_pair_missing_or_initial_reference_drift_fails(self):
        self.assertTrue(callable(paired_source_bootstrap))
        rows = [self.row("one", "a", "base", 0, 0), self.row("one", "a", "all", 1, 1)]
        with self.assertRaises(ValueError):
            paired_source_bootstrap(rows, "all", "base", seed=1, repeats=100)
        with self.assertRaises(ValueError):
            paired_source_bootstrap(rows[:1], "all", "base", seed=1, repeats=100)

    def test_regression_and_repair_both_count_and_empty_denominator_is_null(self):
        self.assertTrue(callable(variant_summary))
        rows = [self.row("one", "a", "all", 1, 0.5), self.row("two", "b", "all", 0.5, 1)]
        result = variant_summary(rows)
        self.assertEqual(result[0]["mean_delta_cdm"], 0)
        self.assertEqual(result[0]["matching_regressions"], 1)
        self.assertEqual(result[0]["nonmatching_fixed"], 1)
        self.assertEqual(result[0]["preserve_metric_proxy"], 0)
        only_bad = variant_summary([self.row("one", "a", "all", 0, 1)])
        self.assertIsNone(only_bad[0]["preserve_metric_proxy"])


if __name__ == "__main__":
    unittest.main()
