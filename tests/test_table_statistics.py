import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from ocr_edr.table_statistics import summarize_table_statistics
except ImportError:
    summarize_table_statistics = None


class TableStatisticsTests(unittest.TestCase):
    def rows(self):
        records = []
        for sample, document, before, after in [
            ("a1", "A", 0.0, 1.0),
            ("a2", "A", 0.0, 1.0),
            ("b1", "B", 1.0, 0.5),
        ]:
            for arm in ["unchanged_0", "base", "all", "no_explicit_preservation"]:
                final = after if arm == "all" else before
                row = {
                    "sample_id": sample,
                    "document_id": document,
                    "family_id": document,
                    "arm": arm,
                    "variant": "preservation" if before == 1 else "cell_perturbation",
                    "initial_teds": before,
                    "final_teds": final,
                    "delta_teds": final - before,
                    "initial_teds_structure": 1.0,
                    "final_teds_structure": 1.0,
                    "delta_teds_structure": 0.0,
                }
                for field in [
                    "model_calls",
                    "input_tokens",
                    "output_tokens",
                    "generation_seconds",
                    "render_attempts",
                    "render_seconds",
                ]:
                    row[field] = 0 if arm == "unchanged_0" else 1
                records.append(row)
        return records

    def analyze(self, rows):
        self.assertTrue(callable(summarize_table_statistics))
        return summarize_table_statistics(rows, seed=7, repeats=200)

    def test_document_equal_weight_and_paired_interval_ignore_record_order(self):
        result = self.analyze(self.rows())
        self.assertEqual(result, self.analyze(list(reversed(self.rows()))))
        contrast = next(
            row
            for row in result["contrasts"]
            if row["arm"] == "all"
            and row["comparator"] == "base"
            and row["metric"] == "teds"
            and row["variant"] == "all"
        )
        self.assertEqual(contrast["documents"], 2)
        self.assertEqual(contrast["case_pairs"], 3)
        self.assertEqual(contrast["document_mean_delta"], 0.25)
        self.assertEqual(contrast["case_mean_delta"], 0.5)
        self.assertEqual(contrast["descriptive_percentile_95_interval"], [-0.5, 1.0])

    def test_proxy_penalty_keeps_regressions_and_uses_document_means(self):
        result = self.analyze(self.rows())
        utility = next(
            row
            for row in result["metric_proxy_utility"]
            if row["arm"] == "all" and row["regression_weight"] == 2
        )
        self.assertEqual(utility["document_mean_utility"], -0.75)
        self.assertEqual(utility["initial_matching_cases"], 1)
        self.assertEqual(utility["matching_regressions"], 1)
        self.assertEqual(utility["initial_nonmatching_cases"], 2)
        self.assertEqual(utility["nonmatching_fully_fixed"], 2)
        baseline = next(row for row in result["costs"] if row["arm"] == "unchanged_0")
        self.assertEqual(baseline["model_calls_total"], 0)

    def test_missing_duplicate_identity_or_initial_drift_cannot_be_compared(self):
        for change in [
            "missing",
            "duplicate",
            "document",
            "family",
            "initial",
            "variant",
            "unknown",
            "nan",
            "cost",
            "delta",
            "unchanged",
        ]:
            rows = copy.deepcopy(self.rows())
            if change == "missing":
                rows.pop()
            elif change == "duplicate":
                rows.append(rows[0])
            elif change == "document":
                rows[2]["document_id"] = "different"
            elif change == "family":
                rows[2]["family_id"] = "different"
            elif change == "initial":
                rows[2]["initial_teds"] = 0.1
            elif change == "variant":
                rows[2]["variant"] = "extra_row"
            elif change == "unknown":
                rows[2]["arm"] = "selected_subset"
            elif change == "nan":
                rows[2]["final_teds"] = float("nan")
            elif change == "cost":
                rows[2]["generation_seconds"] = -1
            elif change == "delta":
                rows[2]["delta_teds"] = 0
            else:
                rows[0]["final_teds"] = 0.5
                rows[0]["delta_teds"] = 0.5
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.analyze(rows)

    def test_full_cli_emits_all_pairs_and_rejects_modified_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluation = root / "evaluation"
            evaluation.mkdir()
            cases, predictions = [], []
            template = self.rows()[0]
            for document in range(32):
                for variant in range(4 if document < 7 else 3):
                    for arm in ["unchanged_0", "base", "all", "no_explicit_preservation"]:
                        # Synthetic fixture with no quality change, never a model result.
                        row = {
                            **template,
                            "sample_id": f"d{document}-v{variant}",
                            "document_id": f"doc{document}",
                            "family_id": f"family{document}",
                            "variant": f"fixture_variant{variant}",
                            "arm": arm,
                        }
                        cases.append(row)
                        predictions.append({"sample_id": row["sample_id"], "arm": arm})
            (evaluation / "evaluation.json").write_text(json.dumps({"cases": cases}))
            (evaluation / "predictions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in predictions)
            )

            def sha(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()

            (evaluation / "run.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "arms": ["unchanged_0", "base", "all", "no_explicit_preservation"],
                        "evaluation_sha256": sha(evaluation / "evaluation.json"),
                        "prediction_sha256": sha(evaluation / "predictions.jsonl"),
                    }
                )
            )
            script = Path(__file__).resolve().parents[1] / "scripts/summarize_table_sft_screen.py"
            command = [sys.executable, str(script), "--evaluation-dir", str(evaluation), "--output"]
            result = subprocess.run(
                [*command, str(root / "report")], capture_output=True, text=True, timeout=60
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / "report/report.json").read_text())
            self.assertEqual(report["documents"], 32)
            self.assertEqual(report["cases_per_arm"], 103)
            self.assertEqual(len(report["contrasts"]), 40)
            self.assertTrue(all(row["document_mean_delta"] == 0 for row in report["contrasts"]))
            self.assertTrue((root / "report/metric_proxy_utility.csv").is_file())
            (evaluation / "evaluation.json").write_text(json.dumps({"cases": cases[:-1]}))
            rejected = subprocess.run(
                [*command, str(root / "tampered-report")],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse((root / "tampered-report").exists())


if __name__ == "__main__":
    unittest.main()
