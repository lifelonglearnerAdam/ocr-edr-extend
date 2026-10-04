import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ocr_edr.official_tables import (
    load_official_table_normalizer,
    load_official_teds,
    verify_official_table_sources,
)
from ocr_edr.table_evaluation import evaluate_tables


class TableEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.arms = ["unchanged_0", "repair"]
        self.inputs = []
        self.references = []
        self.predictions = []
        # Three variants of page A and one of B must not get equal page weights.
        for i in range(4):
            sample = f"s{i}"
            family = "A" if i < 3 else "B"
            initial = "matching" if i < 3 else "bad"
            self.inputs.append({"sample_id": sample, "family_id": family, "prediction": initial})
            self.references.append(
                {
                    "sample_id": sample,
                    "family_id": family,
                    "parent_page": family + ".png",
                    "annotation_id": 1,
                    "gt_position": [1],
                    "variant": "control" if i < 3 else "native",
                    "reference": "matching",
                }
            )
            for arm in self.arms:
                self.predictions.append(
                    {
                        "sample_id": sample,
                        "family_id": family,
                        "arm": arm,
                        "initial_prediction": initial,
                        "final_prediction": initial if arm == "unchanged_0" else "matching",
                        "trace": (
                            []
                            if arm == "unchanged_0"
                            else [
                                {
                                    "generation_seconds": 2.5,
                                    "input_tokens": 100,
                                    "output_tokens": 8,
                                    "adapter_error": None,
                                    "render_error": None,
                                    "hit_length_cap": False,
                                }
                            ]
                        ),
                    }
                )

    def evaluate(self, predictions=None, references=None):
        def score(prediction, reference):
            return {"teds": 1.0 if prediction == reference else 0.0, "teds_structure": 1.0}

        return evaluate_tables(
            self.predictions if predictions is None else predictions,
            self.inputs,
            self.references if references is None else references,
            self.arms,
            normalize=str.strip,
            score=score,
        )

    def test_page_aggregation_preservation_and_cost_keep_distinct_denominators(self):
        result = self.evaluate()
        baseline, repair = [r for r in result["summary"] if r["variant"] == "all"]
        self.assertEqual(baseline["case_mean_final_teds"], 0.75)
        self.assertEqual(baseline["page_mean_final_teds"], 0.5)
        self.assertEqual(repair["case_mean_delta_teds"], 0.25)
        self.assertEqual(repair["page_mean_delta_teds"], 0.5)
        self.assertEqual(repair["teds_initial_matching_n"], 3)
        self.assertEqual(repair["teds_nonmatching_fixed"], 1)
        self.assertEqual(repair["teds_matching_regressions"], 0)
        self.assertEqual(repair["generation_seconds"], 10)
        self.assertEqual(repair["input_tokens"], 400)
        self.assertEqual(repair["model_calls"], 4)

    def test_harmful_edit_to_matching_input_counts_as_regression(self):
        rows = copy.deepcopy(self.predictions)
        rows[1]["final_prediction"] = "wrong"
        repair = next(
            r
            for r in self.evaluate(rows)["summary"]
            if r["arm"] == "repair" and r["variant"] == "all"
        )
        self.assertEqual(repair["teds_matching_regressions"], 1)
        self.assertEqual(repair["teds_degraded"], 1)
        self.assertEqual(repair["teds_final_matching"], 3)

    def test_missing_duplicate_and_unknown_pairs_are_rejected(self):
        for rows in [self.predictions[:-1], self.predictions + [self.predictions[0]]]:
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                self.evaluate(rows)
        rows = copy.deepcopy(self.predictions)
        rows[0]["arm"] = "other"
        with self.assertRaises(ValueError):
            self.evaluate(rows)

    def test_reference_coverage_family_and_initial_drift_are_rejected(self):
        with self.assertRaises(ValueError):
            self.evaluate(references=self.references[:-1])
        for field, value in [("family_id", "B"), ("initial_prediction", "drift")]:
            rows = copy.deepcopy(self.predictions)
            rows[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.evaluate(rows)
        refs = copy.deepcopy(self.references)
        refs[0]["parent_page"] = "another.png"
        with self.assertRaises(ValueError):
            self.evaluate(references=refs)

    def test_failed_candidate_is_retained_and_charged_after_rollback(self):
        rows = copy.deepcopy(self.predictions)
        rows[-1]["final_prediction"] = "bad"
        rows[-1]["trace"][0].update({"adapter_error": "invalid JSON", "hit_length_cap": True})
        repair = next(
            r
            for r in self.evaluate(rows)["summary"]
            if r["arm"] == "repair" and r["variant"] == "all"
        )
        self.assertEqual(repair["cases"], 4)
        self.assertEqual(repair["adapter_failures"], 1)
        self.assertEqual(repair["model_calls"], 4)
        self.assertEqual(repair["hit_length_cap"], 1)
        self.assertEqual(repair["teds_nonmatching_fixed"], 0)
        rows[-1]["final_prediction"] = "matching"
        with self.assertRaises(ValueError):
            self.evaluate(rows)

    def test_unchanged_baseline_cannot_be_edited_and_silent_missing_calls_are_rejected(self):
        rows = copy.deepcopy(self.predictions)
        rows[0]["final_prediction"] = "wrong"
        with self.assertRaises(ValueError):
            self.evaluate(rows)
        rows = copy.deepcopy(self.predictions)
        rows[-1]["trace"] = []
        with self.assertRaises(ValueError):
            self.evaluate(rows)
        rows[-1].update({"skip_reason": "initial_render_failure", "final_prediction": "bad"})
        result = self.evaluate(rows)
        self.assertTrue(result["cases"][-1]["skipped"])


class OfficialSourceIntegrityTests(unittest.TestCase):
    def test_modified_metric_source_is_rejected_before_import(self):
        paths = [
            "src/metrics/table_metric.py",
            "src/core/preprocess/data_preprocess.py",
            "src/core/preprocess/table_postprocess.py",
            "src/core/preprocess/table_utils.py",
            "src/core/preprocess/text_postprocess.py",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blobs = []
            for relative in paths:
                file = root / relative
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b"unchanged\n")
                blobs.append(
                    {
                        "path": relative,
                        "type": "blob",
                        "sha": hashlib.sha1(b"blob 10\0unchanged\n").hexdigest(),
                    }
                )
            (root / "revision.json").write_text(json.dumps({"revision": "pinned"}))
            (root / "inventory.json").write_text(
                json.dumps({"sha": "pinned", "tree": blobs, "truncated": False})
            )
            self.assertEqual(len(verify_official_table_sources(root, "pinned")["files"]), 5)
            (root / paths[0]).write_bytes(b"modified\n")
            with self.assertRaisesRegex(ValueError, "Modified or missing"):
                verify_official_table_sources(root, "pinned")


@unittest.skipUnless(
    os.environ.get("OCR_EDR_TEST_TABLE") == "1", "explicit pinned official metric check"
)
class OfficialTableIntegrationTests(unittest.TestCase):
    def test_all_ten_published_teds_scores_replay_and_new_normalization_is_symmetric(self):
        root = Path(__file__).resolve().parents[1] / "data/raw/omnidocbench-source"
        verify_official_table_sources(root, "f133a71e9e91c3621c7ce8994200a7b394a06eb3")
        metric = load_official_teds(root)
        content, structure = metric(), metric(structure_only=True)
        records = json.loads((root / "result/end2end_quick_match_table_result.json").read_text())
        self.assertEqual(len(records), 10)
        normalize = load_official_table_normalizer(root)
        for record in records:
            with self.subTest(page=record["img_id"], order=record["gt_position"]):
                self.assertAlmostEqual(
                    content.evaluate(record["norm_pred"], record["norm_gt"]),
                    record["metric"]["TEDS"],
                    places=12,
                )
                self.assertAlmostEqual(
                    structure.evaluate(record["norm_pred"], record["norm_gt"]),
                    record["metric"]["TEDS_structure_only"],
                    places=12,
                )
                reference = normalize(record["gt"])
                self.assertEqual(content.evaluate(reference, reference), 1.0)
                self.assertEqual(structure.evaluate(reference, reference), 1.0)


if __name__ == "__main__":
    unittest.main()
