import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from ocr_edr.native_tables import (
        evaluate_native_tables,
        load_native_table_sources,
        native_editor_profile,
        original_frame_box,
        recognize_table_source,
    )
except ImportError:
    evaluate_native_tables = load_native_table_sources = None
    native_editor_profile = recognize_table_source = None
    original_frame_box = None


class NativeTableBoundaryTests(unittest.TestCase):
    def test_train_native_pool_uses_admission_metadata_without_loading_targets_or_excluded_image(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "images/good.png").write_bytes(b"good")
            inputs = [
                {
                    "family_id": name,
                    "source_image": "images/" + name + ".png",
                    "source_sha256": hashlib.sha256(name.encode()).hexdigest(),
                }
                for name in ["good", "excluded"]
            ]
            inventory = [
                {**r, "role": "train", "document_id": "PMC" + r["family_id"]} for r in inputs
            ]
            for name, rows in [
                ("train-source-inputs.jsonl", inputs),
                ("selected_sources.jsonl", inventory),
            ]:
                (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
            (root / "dataset.json").write_text(
                json.dumps(
                    {
                        "roles": {"train": {"sources": 2}},
                        "file_sha256": {
                            n: hashlib.sha256((root / n).read_bytes()).hexdigest()
                            for n in ["train-source-inputs.jsonl", "selected_sources.jsonl"]
                        },
                    }
                )
            )
            (root / "admitted-supervision").mkdir()
            receipt = {
                "status": "admitted_as_published_weak_supervision",
                "documents": 1,
                "excluded_families": {"excluded": "reviewed label conflict"},
            }
            (root / "admitted-supervision/admission.json").write_text(json.dumps(receipt))
            try:
                rows = load_native_table_sources(root, role="train")
            except TypeError as error:
                self.fail(f"Train-role source admission unavailable: {error}")
            self.assertEqual(rows, inputs[:1])
            for forbidden in ["gate_calibration", "locked_evaluation"]:
                with self.assertRaises(ValueError):
                    load_native_table_sources(root, role=forbidden)

    def test_original_frame_coordinates_invert_non_square_training_normalization(self):
        self.assertTrue(callable(original_frame_box))
        normalized = [0.1, 0.2, 0.5, 0.7]
        self.assertEqual(
            original_frame_box(normalized, [50, 200, 2.44, 2.44, 488, 488]), [20, 10, 100, 35]
        )
        self.assertEqual(
            original_frame_box(normalized, [200, 50, 2.44, 2.44, 488, 488]), [5, 40, 25, 140]
        )
        self.assertEqual(normalized, [0.1, 0.2, 0.5, 0.7])
        with self.assertRaises(ValueError):
            original_frame_box(normalized, [0, 200, 1, 1, 488, 488])
        with self.assertRaises(ValueError):
            original_frame_box([0, float("nan"), 1, 1], [50, 200, 1, 1, 488, 488])

    def test_source_loader_never_needs_reference_and_rejects_role_or_payload_drift(self):
        self.assertTrue(callable(load_native_table_sources))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "images/x.png").write_bytes(b"source")
            row = {
                "family_id": "p1",
                "source_image": "images/x.png",
                "source_sha256": hashlib.sha256(b"source").hexdigest(),
            }

            def save(value, role="model_dev"):
                inventory = {**row, "role": role, "document_id": "PMC1"}
                files = {
                    "model_dev-source-inputs.jsonl": value,
                    "selected_sources.jsonl": inventory,
                }
                for name, item in files.items():
                    (root / name).write_text(json.dumps(item) + "\n")
                (root / "dataset.json").write_text(
                    json.dumps(
                        {
                            "roles": {"model_dev": {"sources": 1}},
                            "file_sha256": {
                                n: hashlib.sha256((root / n).read_bytes()).hexdigest()
                                for n in files
                            },
                        }
                    )
                )

            save(row)
            self.assertEqual(load_native_table_sources(root), [row])
            for value, role in [
                ({**row, "reference": "hidden"}, "model_dev"),
                (row, "locked_evaluation"),
                ({**row, "source_sha256": "0" * 64}, "model_dev"),
            ]:
                save(value, role)
                with self.assertRaises(ValueError):
                    load_native_table_sources(root)

    def test_parser_error_preserves_source_identity_and_attempt_cost(self):
        self.assertTrue(callable(recognize_table_source))
        row = {"family_id": "p1", "source_image": "images/x.png", "source_sha256": "a" * 64}

        def fail(path):
            raise RuntimeError("fixture recognition failure")

        result = recognize_table_source(row, Path("/fixture"), recognize=fail)
        self.assertEqual(result["family_id"], "p1")
        self.assertIsNone(result["prediction"])
        self.assertIn("fixture recognition failure", result["parser_error"])
        self.assertGreaterEqual(result["generation_seconds"], 0)
        self.assertEqual(result["pipeline_calls"], 1)

    def test_full_coverage_scores_failed_parser_as_zero_and_rejects_missing_or_duplicate(self):
        self.assertTrue(callable(evaluate_native_tables))
        inputs = [
            {"family_id": f, "source_image": f + ".png", "source_sha256": f * 64}
            for f in ["a", "b"]
        ]
        refs = [
            {"family_id": f, "document_id": "PMC" + f, "reference": "good", "role": "model_dev"}
            for f in ["a", "b"]
        ]
        rows = [
            {
                **inputs[0],
                "prediction": "good",
                "parser_error": None,
                "generation_seconds": 2.0,
                "pipeline_calls": 1,
            },
            {
                **inputs[1],
                "prediction": None,
                "parser_error": "failed",
                "generation_seconds": 3.0,
                "pipeline_calls": 1,
            },
        ]

        def evaluate(predictions, references=refs):
            return evaluate_native_tables(
                predictions,
                inputs,
                references,
                normalize=str.strip,
                score=lambda p, r: {"teds": float(p == r), "teds_structure": float(p == r)},
            )

        result = evaluate(rows)
        self.assertEqual(result["summary"]["sources"], 2)
        self.assertEqual(result["summary"]["mean_teds"], 0.5)
        self.assertEqual(result["summary"]["parser_failures"], 1)
        self.assertEqual(result["summary"]["pipeline_calls"], 2)
        self.assertEqual(result["summary"]["generation_seconds"], 5)
        for altered in [
            rows[:1],
            rows + rows[:1],
            [{**rows[0], "source_sha256": "c" * 64}, rows[1]],
        ]:
            with self.assertRaises(ValueError):
                evaluate(altered)
        with self.assertRaises(ValueError):
            evaluate(rows, [{**r, "role": "gate_calibration"} for r in refs])


@unittest.skipUnless(importlib.util.find_spec("lxml"), "optional table dependency")
class NativeEditorProfileTests(unittest.TestCase):
    def test_missing_rows_and_cells_are_outside_existing_noninserting_action_space(self):
        self.assertTrue(callable(native_editor_profile))
        reference = "<table><tr><td>A</td><td>17</td></tr><tr><td>B</td><td>19</td></tr></table>"
        missing = "<table><tr><td>A</td><td>17</td></tr></table>"
        result = native_editor_profile(missing, reference)
        self.assertEqual(result["prediction_rows"], 1)
        self.assertEqual(result["reference_rows"], 2)
        self.assertTrue(result["requires_row_insertion_for_reference_dom"])
        self.assertTrue(result["requires_cell_insertion_for_reference_dom"])
        self.assertIsNone(result["aligned_plain_text_mismatches"])
        edited = "<table><tr><td>A</td><td>18</td></tr><tr><td>B</td><td>19</td></tr></table>"
        result = native_editor_profile(edited, reference)
        self.assertFalse(result["requires_row_insertion_for_reference_dom"])
        self.assertEqual(result["aligned_plain_text_mismatches"], 1)
        self.assertFalse(native_editor_profile("not html", reference)["editor_input_valid"])


if __name__ == "__main__":
    unittest.main()
