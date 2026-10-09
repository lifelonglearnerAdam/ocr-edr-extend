import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from ocr_edr.table_training import (
        admit_training_records,
        document_balanced_schedule,
        load_table_supervision,
    )
except ImportError:
    admit_training_records = document_balanced_schedule = load_table_supervision = None


class TableAdmissionTests(unittest.TestCase):
    def rows(self):
        return [
            {
                "sample_id": family + "-" + variant,
                "family_id": family,
                "document_id": doc,
                "role": "train",
                "variant": variant,
            }
            for family, doc, variants in [
                ("good", "PMC1", ["preservation", "cell_perturbation", "extra_row"]),
                (
                    "flagged",
                    "PMC2",
                    ["preservation", "cell_perturbation", "extra_row", "span_perturbation"],
                ),
            ]
            for variant in variants
        ]

    def test_confirmed_disagreement_excludes_whole_family_without_relabeling(self):
        self.assertTrue(callable(admit_training_records))
        original = self.rows()
        before = copy.deepcopy(original)
        admitted, exclusions = admit_training_records(
            original, {"flagged": "released_reference_missing_visible_row"}
        )
        self.assertEqual(original, before)
        self.assertEqual({r["family_id"] for r in admitted}, {"good"})
        self.assertEqual(len(exclusions), 4)
        self.assertTrue(
            all(r["reason"] == "released_reference_missing_visible_row" for r in exclusions)
        )

    def test_unknown_review_duplicate_identity_and_nontrain_role_fail(self):
        self.assertTrue(callable(admit_training_records))
        with self.assertRaises(ValueError):
            admit_training_records(self.rows(), {"unknown": "reviewed"})
        with self.assertRaises(ValueError):
            admit_training_records(self.rows() + self.rows()[:1], {})
        for role in ["model_dev", "gate_calibration", "locked_evaluation"]:
            with self.subTest(role=role):
                rows = self.rows()
                rows[0]["role"] = role
                with self.assertRaises(ValueError):
                    admit_training_records(rows, {})

    def test_documents_have_equal_exposure_in_both_preservation_arms(self):
        self.assertTrue(callable(document_balanced_schedule))
        rows = self.rows()
        for arm in ["all", "no_explicit_preservation"]:
            schedule = document_balanced_schedule(
                rows, arm=arm, passes_per_document=12, seed=20261007
            )
            self.assertEqual(len(schedule), 24)
            self.assertEqual(
                Counter(rows[i]["document_id"] for i in schedule), {"PMC1": 12, "PMC2": 12}
            )
            per_record = Counter(schedule)
            for i, row in enumerate(rows):
                if arm == "no_explicit_preservation" and row["variant"] == "preservation":
                    self.assertEqual(per_record[i], 0)
            self.assertEqual(
                schedule,
                document_balanced_schedule(rows, arm=arm, passes_per_document=12, seed=20261007),
            )
        with self.assertRaises(ValueError):
            document_balanced_schedule(rows, arm="all", passes_per_document=5, seed=1)

    @unittest.skipUnless(importlib.util.find_spec("lxml"), "optional table dependency")
    def test_loader_rebases_verified_images_but_rejects_role_target_or_prompt_drift(self):
        self.assertTrue(callable(load_table_supervision))
        from ocr_edr.table_supervision import table_action_prompt

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            image = root / "images/PMC1_001_00.png"
            image.write_bytes(b"synthetic source image")
            image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
            source = {
                "family_id": "p1",
                "document_id": "PMC1",
                "role": "train",
                "source_image": "images/PMC1_001_00.png",
                "source_sha256": image_hash,
            }
            sources = root / "selected_sources.jsonl"
            sources.write_text(json.dumps(source) + "\n")
            (root / "dataset.json").write_text(
                json.dumps(
                    {
                        "file_sha256": {
                            "selected_sources.jsonl": hashlib.sha256(
                                sources.read_bytes()
                            ).hexdigest()
                        }
                    }
                )
            )
            admitted = root / "admitted"
            admitted.mkdir()
            candidate = "<table><tr><td>19</td><td>7</td></tr></table>"
            row = {
                "sample_id": "p1-keep",
                "family_id": "p1",
                "document_id": "PMC1",
                "role": "train",
                "modality": "table",
                "source_image": "/old-host/data/images/PMC1_001_00.png",
                "source_sha256": image_hash,
                "candidate": candidate,
                "prompt": table_action_prompt(candidate),
                "target": '{"action":"stop"}',
                "variant": "preservation",
                "target_provenance": "published_annotation_controlled_corruption_not_teacher_or_native",
            }

            def save(value):
                path = admitted / "train-sft.jsonl"
                path.write_text(json.dumps(value) + "\n")
                (admitted / "admission.json").write_text(
                    json.dumps(
                        {
                            "status": "admitted_as_published_weak_supervision",
                            "excluded_families": {},
                            "file_sha256": {
                                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                            },
                        }
                    )
                )

            save(row)
            loaded = load_table_supervision(admitted, root, role="train")
            self.assertEqual(loaded[0]["resolved_source_image"], str(image))
            for change in [
                {"role": "gate_calibration"},
                {"prompt": "correct answer from reference"},
                {"target": '{"action":"delete_row","row":0}'},
                {"source_sha256": "0" * 64},
                {"source_image": "/old-host/other.png"},
            ]:
                save({**row, **change})
                with self.assertRaises(ValueError):
                    load_table_supervision(admitted, root, role="train")
            with self.assertRaises(ValueError):
                load_table_supervision(admitted, root, role="locked_evaluation")


if __name__ == "__main__":
    unittest.main()
