"""Protect evaluation against silent renderer zeros and incomplete comparisons."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ocr_edr.cdm_evaluation import evaluate_cdm
from ocr_edr.official_cdm import verify_official_cdm_sources


# The external renderer is replaced with hand-counted symbol matches; evaluation is real.
def symbol_metric(reference, prediction):
    if reference == r"\frac{":
        return dict(recall=0.0, precision=0.0, F1_score=0.0, tp=0, gt_tokens=0, pred_tokens=0)
    if prediction == "x^2+1":
        return dict(recall=1.0, precision=1.0, F1_score=1.0, tp=4, gt_tokens=4, pred_tokens=4)
    if prediction == "x^3+1":
        return dict(recall=0.75, precision=0.75, F1_score=0.75, tp=3, gt_tokens=4, pred_tokens=4)
    if prediction == r"\frac{":
        return dict(recall=0.0, precision=0.0, F1_score=0.0, tp=0, gt_tokens=4, pred_tokens=0)
    raise AssertionError("Unexpected external metric input")


class CDMEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.references = [
            dict(sample_id="s0", family_id="A", reference="x^2+1", variant="native"),
            dict(sample_id="s1", family_id="A", reference="x^2+1", variant="native"),
            dict(sample_id="s2", family_id="B", reference="x^2+1", variant="native"),
        ]
        self.arms = ["unchanged", "repair"]
        self.predictions = []
        for i, (initial, final) in enumerate(
            [("x^3+1", "x^2+1"), ("x^2+1", "x^3+1"), ("x^3+1", r"\frac{")]
        ):
            for arm in self.arms:
                self.predictions.append(
                    dict(
                        sample_id=f"s{i}",
                        family_id="A" if i < 2 else "B",
                        arm=arm,
                        initial_prediction=initial,
                        final_prediction=initial if arm == "unchanged" else final,
                    )
                )

    def evaluate(self, predictions=None, references=None, metric=symbol_metric):
        return evaluate_cdm(
            self.predictions if predictions is None else predictions,
            self.references if references is None else references,
            self.arms,
            metric=metric,
        )

    def test_partial_repairs_regressions_and_failed_predictions_remain_in_denominator(self):
        result = self.evaluate()
        summary = next(r for r in result["summary"] if r["arm"] == "repair")
        self.assertEqual(len(result["cases"]), 6)
        self.assertEqual(summary["cases"], 3)
        self.assertEqual(summary["sources"], 2)
        self.assertAlmostEqual(summary["case_mean_initial_cdm"], 5 / 6)
        self.assertAlmostEqual(summary["case_mean_final_cdm"], 7 / 12)
        self.assertAlmostEqual(summary["source_mean_delta_cdm"], -0.375)
        self.assertEqual(summary["matching_regressions"], 1)
        self.assertEqual(summary["nonmatching_fixed"], 1)
        self.assertEqual(summary["improved"], 1)
        self.assertEqual(summary["degraded"], 2)
        self.assertEqual(summary["final_prediction_unrenderable_or_empty"], 1)
        self.assertEqual(len(result["reference_readiness"]), 3)
        self.assertEqual(result["metric_calls"], 3)

    def test_reference_failure_stops_evaluation_instead_of_publishing_zero_scores(self):
        refs = copy.deepcopy(self.references)
        refs[0]["reference"] = r"\frac{"
        with self.assertRaisesRegex(ValueError, "Reference CDM readiness"):
            self.evaluate(references=refs)

    def test_silent_runtime_failure_after_selfcheck_is_rejected(self):
        def broken(reference, prediction):
            if prediction != reference:
                return dict(
                    recall=0.0, precision=0.0, F1_score=0.0, tp=0, gt_tokens=0, pred_tokens=0
                )
            return symbol_metric(reference, prediction)

        with self.assertRaisesRegex(ValueError, "reference token"):
            self.evaluate(metric=broken)

    def test_missing_duplicate_and_unknown_pairs_are_rejected(self):
        for rows in [self.predictions[:-1], self.predictions + [self.predictions[0]]]:
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                self.evaluate(predictions=rows)
        rows = copy.deepcopy(self.predictions)
        rows[0]["arm"] = "unexpected"
        with self.assertRaises(ValueError):
            self.evaluate(predictions=rows)
        with self.assertRaises(ValueError):
            self.evaluate(references=self.references + [self.references[0]])

    def test_family_and_initial_drift_between_arms_are_rejected(self):
        for field, value in [("family_id", "wrong"), ("initial_prediction", "x^2+1")]:
            rows = copy.deepcopy(self.predictions)
            rows[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.evaluate(predictions=rows)

    def test_nonfinite_or_out_of_range_metric_is_rejected(self):
        for value in [float("nan"), float("inf"), -0.1, 1.1]:

            def broken(reference, prediction):
                result = symbol_metric(reference, prediction)
                result["F1_score"] = value
                return result

            with self.subTest(value=value), self.assertRaises(ValueError):
                self.evaluate(metric=broken)


class CDMSourceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.revision = "1" * 40
        self.relative = "src/metrics/cdm/cdm.py"
        path = self.root / self.relative
        path.parent.mkdir(parents=True)
        path.write_bytes(b"# frozen upstream fixture\n")
        content = path.read_bytes()
        blob = hashlib.sha1(b"blob 26\0" + content).hexdigest()
        # The fixture is 26 bytes; using a literal header catches a changed hash algorithm.
        self.assertEqual(len(content), 26)
        (self.root / "revision.json").write_text(json.dumps({"revision": self.revision}))
        self.inventory = {
            "sha": self.revision,
            "truncated": False,
            "tree": [{"path": self.relative, "type": "blob", "sha": blob}],
        }
        self.write_inventory()

    def write_inventory(self):
        (self.root / "inventory.json").write_text(json.dumps(self.inventory))

    def test_source_edit_and_uninventoried_python_module_are_rejected(self):
        receipt = verify_official_cdm_sources(self.root, self.revision)
        self.assertEqual(receipt["revision"], self.revision)
        self.assertEqual(
            receipt["files"][self.relative]["git_blob_sha1"], self.inventory["tree"][0]["sha"]
        )
        path = self.root / self.relative
        path.write_bytes(b"# altered upstream\n")
        with self.assertRaisesRegex(ValueError, "Modified"):
            verify_official_cdm_sources(self.root, self.revision)
        path.write_bytes(b"# frozen upstream fixture\n")
        path.with_name("extra.py").write_text("# could be imported\n")
        with self.assertRaisesRegex(ValueError, "inventory"):
            verify_official_cdm_sources(self.root, self.revision)

    def test_wrong_revision_truncated_and_missing_entrypoint_are_rejected(self):
        with self.assertRaises(ValueError):
            verify_official_cdm_sources(self.root, "2" * 40)
        self.inventory["truncated"] = True
        self.write_inventory()
        with self.assertRaises(ValueError):
            verify_official_cdm_sources(self.root, self.revision)
        self.inventory["truncated"] = False
        self.inventory["tree"] = []
        self.write_inventory()
        with self.assertRaises(ValueError):
            verify_official_cdm_sources(self.root, self.revision)


@unittest.skipUnless(
    os.environ.get("OCR_EDR_TEST_CDM") == "1", "explicit official CDM runtime check"
)
class CDMIntegrationTests(unittest.TestCase):
    def test_cli_retains_malformed_prediction_and_partial_symbol_credit(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            references = [
                dict(sample_id=str(i), family_id=str(i), reference="x^2+1") for i in range(3)
            ]
            predictions = [
                dict(
                    sample_id=str(i),
                    family_id=str(i),
                    arm="control",
                    initial_prediction="x^2+1",
                    final_prediction=prediction,
                )
                for i, prediction in enumerate(["x^2+1", "x^3+1", r"\frac{"])
            ]
            for name, rows in [("references", references), ("predictions", predictions)]:
                (root / (name + ".jsonl")).write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
            result = subprocess.run(
                [
                    sys.executable,
                    str(repo / "scripts/evaluate_cdm.py"),
                    "--predictions",
                    str(root / "predictions.jsonl"),
                    "--references",
                    str(root / "references.jsonl"),
                    "--official-root",
                    str(repo / "data/raw/omnidocbench-source"),
                    "--runtime-env",
                    str(repo / "data/raw/cdm-tex-runtime-20261004/env.json"),
                    "--tmp-dir",
                    str(root / "scratch"),
                    "--output",
                    str(root / "evaluation"),
                ],
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            evaluation = json.loads((root / "evaluation/evaluation.json").read_text())
            self.assertEqual(len(evaluation["cases"]), 3)
            self.assertEqual(
                [row["final_cdm"]["F1_score"] for row in evaluation["cases"]], [1.0, 0.75, 0.0]
            )
            self.assertEqual(evaluation["summary"][0]["matching_regressions"], 2)
            self.assertEqual(evaluation["summary"][0]["final_prediction_unrenderable_or_empty"], 1)
            self.assertEqual(evaluation["new_model_calls"], 0)
            self.assertEqual(
                json.loads((root / "evaluation/run.json").read_text())["status"], "completed"
            )


if __name__ == "__main__":
    unittest.main()
