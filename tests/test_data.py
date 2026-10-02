import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_edr.data import assert_trainable, prepare_note, read_jsonl


def fixture(root, test_only=True, crop_path=None):
    gt_dir = root / "OmniDocBench_note"
    (gt_dir / "images").mkdir(parents=True)
    (gt_dir / "images/page.png").write_bytes(b"synthetic page fixture")
    (gt_dir / "OmniDocBench_note.json").write_text(
        json.dumps(
            [
                {
                    "page_info": {
                        "image_path": "page.png",
                        "page_attribute": {"data_source": "note"},
                    },
                    "layout_dets": [{"anno_id": 1, "order": 1}, {"anno_id": 2, "order": 2}],
                }
            ]
        )
    )
    results = root / "baselines/monkeyocrv2_b_note/omnidocbench_results_cdm"
    results.mkdir(parents=True)
    for modality, category, metric, anno, reference in [
        ("formula", "display_formula", "CDM", 1, "$$x^2$$"),
        ("table", "table", "TEDS", 2, "<table><tr><td>1</td></tr></table>"),
    ]:
        directory = root / f"diagnostics/OmniDocBench_note_test_only/{modality}"
        (directory / "images").mkdir(parents=True)
        (directory / "images/crop.png").write_bytes(f"synthetic {modality} fixture".encode())
        (directory / "manifest_test_only.jsonl").write_text(
            json.dumps(
                {
                    "split": "test",
                    "test_only": test_only,
                    "anno_id": anno,
                    "image_path": "page.png",
                    "bbox": [0, 0, 1, 1],
                    "ground_truth": reference,
                    "crop": crop_path
                    or f"/data/hzhang/diagnostics/OmniDocBench_note_test_only/{modality}/images/crop.png",
                }
            )
            + "\n"
        )
        row = {
            "gt_idx": [0],
            "gt_position": [anno],
            "img_id": "page.png",
            "gt": reference,
            "pred": reference,
            "norm_gt": reference,
            "norm_pred": reference,
            "metric": {metric: 1},
        }
        (results / f"fixture_{category}_result.json").write_text(json.dumps([row]))
    (results / "fixture_metric_result.json").write_text(
        json.dumps(
            {
                "display_formula": {"page": {"CDM": {"ALL": 1}}},
                "table": {"page": {"TEDS": {"ALL": 1}}},
            }
        )
    )


class DataTests(unittest.TestCase):
    def test_import_preserves_test_provenance_and_rebases_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp) / "mirror", Path(tmp) / "output"
            fixture(root)
            report = prepare_note(root, output)
            self.assertEqual(report["diagnostic_elements"], {"formula": 1, "table": 1})
            self.assertEqual(report["blocked_image_hashes"], 3)
            records = read_jsonl(output / "elements_test_only.jsonl")
            for row in records:
                self.assertTrue(row["test_only"])
                self.assertEqual(row["alignment"], "one_to_one")
                self.assertTrue(Path(row["source_image"]).is_file())
            with self.assertRaises(ValueError):
                assert_trainable(records)

    def test_split_matches_are_retained_without_guessing_element_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp) / "mirror", Path(tmp) / "output"
            fixture(root)
            p = (
                root
                / "baselines/monkeyocrv2_b_note/omnidocbench_results_cdm/fixture_display_formula_result.json"
            )
            rows = json.loads(p.read_text())
            second = dict(rows[0], gt_idx=[1], gt="y", norm_gt="y", pred="y", norm_pred="y")
            p.write_text(json.dumps(rows + [second]))
            report = prepare_note(root, output)
            self.assertEqual(report["official_matches"]["formula"], 2)
            record = read_jsonl(output / "elements_test_only.jsonl")[0]
            self.assertEqual(record["alignment"], "split_or_merged")
            self.assertIsNone(record["initial_prediction"])

    def test_import_rejects_missing_test_only_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mirror"
            fixture(root, test_only=False)
            with self.assertRaisesRegex(ValueError, "test-only"):
                prepare_note(root, Path(tmp) / "output")

    def test_import_rejects_crop_path_outside_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mirror"
            fixture(root, crop_path="/data/hzhang/../../outside.png")
            with self.assertRaises(ValueError):
                prepare_note(root, Path(tmp) / "output")

    def test_training_metadata_requires_explicit_independent_split(self):
        assert_trainable([{"split": "train", "test_only": False}])
        for record in [{}, {"split": "train"}, {"split": "test", "test_only": False}]:
            with self.assertRaises(ValueError):
                assert_trainable([record])


if __name__ == "__main__":
    unittest.main()
