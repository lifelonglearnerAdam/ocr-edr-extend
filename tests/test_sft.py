import hashlib
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from ocr_edr.sft import (
        assistant_labels,
        load_supervision,
        proposal_contract,
        training_schedule,
        validate_disjoint,
    )
except ImportError:
    assistant_labels = load_supervision = proposal_contract = training_schedule = (
        validate_disjoint
    ) = None

from ocr_edr.formula_pilot import make_prompt

try:
    from ocr_edr.sft import verify_model_files
except ImportError:
    verify_model_files = None


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "train").mkdir()
        (self.root / "dev").mkdir()
        (self.root / "supervision").mkdir()
        (self.root / "train/a.png").write_bytes(b"synthetic train image")
        (self.root / "dev/b.png").write_bytes(b"synthetic dev image")

    def row(self, family="a", split="train", variant="preservation", candidate="x^2"):
        source = f"../{split}/{family}.png"
        image = self.root / split / f"{family}.png"
        return {
            "sample_id": family + "-" + variant,
            "family_id": family,
            "split": split,
            "modality": "formula",
            "source_image": source,
            "source_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "candidate": candidate,
            "prompt": make_prompt(candidate, False),
            "target": "<latex>x^2</latex>",
            "variant": variant,
            "target_provenance": "released_annotation_or_preservation_copy",
        }

    def save(self, rows, name="train"):
        path = self.root / "supervision" / (name + ".jsonl")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_train_and_dev_loaded_with_hash_and_contained_source_paths(self):
        self.assertTrue(callable(load_supervision), "supervision validator not implemented")
        p, digest = self.save([self.row()])
        rows = load_supervision(p, self.root, split="train", expected_sha256=digest)
        self.assertEqual(rows[0]["sample_id"], "a-preservation")
        self.assertTrue(Path(rows[0]["resolved_source_image"]).is_file())
        self.assertEqual(rows[0]["candidate"], "x^2")

    def test_hash_drift_wrong_split_duplicate_and_prompt_leak_rejected(self):
        self.assertTrue(callable(load_supervision))
        p, _ = self.save([self.row()])
        with self.assertRaisesRegex(ValueError, "hash"):
            load_supervision(p, self.root, split="train", expected_sha256="0" * 64)
        invalid = [
            [dict(self.row(), split="dev")],
            [self.row(), self.row()],
            [dict(self.row(), prompt="Hidden reference x^2")],
            [dict(self.row(), source_image="../../../outside.png")],
            [dict(self.row(), source_sha256="0" * 64)],
            [dict(self.row(), reference="x^2")],
            [dict(self.row(), target="<latex>y^2</latex>")],
            [dict(self.row(), target="x^2")],
        ]
        for rows in invalid:
            with self.subTest(rows=rows):
                p, digest = self.save(rows)
                with self.assertRaises(ValueError):
                    load_supervision(p, self.root, split="train", expected_sha256=digest)

    def test_disjoint_families_and_images_required(self):
        self.assertTrue(callable(validate_disjoint))
        train, dev = [self.row()], [self.row("b", "dev")]
        validate_disjoint(train, dev)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_disjoint(train, [dict(dev[0], family_id="a")])
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_disjoint(train, [dict(dev[0], source_sha256=train[0]["source_sha256"])])

    def test_equal_source_exposure_and_step_budget_for_both_arms(self):
        self.assertTrue(callable(training_schedule))
        rows = [
            {"sample_id": f"{family}-{variant}", "family_id": family, "variant": variant}
            for family in ["a", "b"]
            for variant in ["preservation", "controlled_error", "native_parser_prediction"]
        ]
        full = training_schedule(rows, arm="all", exposures=12, seed=7)
        drop = training_schedule(rows, arm="no_explicit_preservation", exposures=12, seed=7)
        self.assertEqual(len(full), len(drop))
        self.assertEqual(Counter(rows[i]["family_id"] for i in full), {"a": 6, "b": 6})
        self.assertEqual(Counter(rows[i]["family_id"] for i in drop), {"a": 6, "b": 6})
        self.assertFalse(any(rows[i]["variant"] == "preservation" for i in drop))
        self.assertEqual(full, training_schedule(rows, arm="all", exposures=12, seed=7))
        with self.assertRaises(ValueError):
            training_schedule(rows, arm="unknown", exposures=12, seed=7)
        with self.assertRaises(ValueError):
            training_schedule(rows, arm="all", exposures=5, seed=7)

    def test_assistant_only_mask_and_attention_mask(self):
        self.assertTrue(callable(assistant_labels))
        self.assertEqual(
            assistant_labels([11, 12], [11, 12, 13, 14, 0], [1, 1, 1, 1, 0]),
            [-100, -100, 13, 14, -100],
        )
        for prefix, complete, attention in [
            ([11, 9], [11, 12, 13], [1, 1, 1]),
            ([11, 12], [11, 12], [1, 1]),
            ([11], [11, 12], [1, 0]),
            ([11], [11, 12], [1]),
            ([], [11, 12], [1, 1]),
        ]:
            with self.subTest(prefix=prefix):
                with self.assertRaises(ValueError):
                    assistant_labels(prefix, complete, attention)

    def test_inference_contract_and_cap_do_not_guess_from_prose(self):
        self.assertTrue(callable(proposal_contract))
        self.assertEqual(
            proposal_contract("<latex>x^2</latex>", hit_cap=False), ("x^2", "accepted_contract")
        )
        self.assertEqual(
            proposal_contract(r"<latex>\begin{equation}x^2\end{equation}</latex>", hit_cap=False),
            ("x^2", "accepted_contract"),
        )
        self.assertEqual(proposal_contract("<latex>x^2</latex>", hit_cap=True), (None, "token_cap"))
        for raw in [
            "x^2",
            "The answer is <latex>x^2</latex>",
            "<latex></latex>",
            "<latex>x</latex><latex>y</latex>",
        ]:
            with self.subTest(raw=raw):
                candidate, _ = proposal_contract(raw, hit_cap=False)
                self.assertIsNone(candidate)

    def test_model_receipts_support_hf_blobs_and_reject_external_links_and_names(self):
        self.assertTrue(callable(verify_model_files))
        revision = "f" * 40
        cache = self.root / "models--Fixture--OCR"
        snapshot = cache / "snapshots" / revision
        blobs = cache / "blobs"
        snapshot.mkdir(parents=True)
        blobs.mkdir()
        content = b"fixed model config"
        digest = hashlib.sha256(content).hexdigest()
        blob = blobs / ("a" * 40)
        blob.write_bytes(content)
        link = snapshot / "config.json"
        link.symlink_to(blob)
        receipt = {
            "model": "Fixture/OCR",
            "revision": revision,
            "files": {"config.json": {"bytes": len(content), "sha256": digest}},
        }
        verify_model_files(snapshot, receipt)
        link.unlink()
        link.symlink_to(self.root / "train/a.png")
        with self.assertRaises(ValueError):
            verify_model_files(snapshot, receipt)
        link.unlink()
        link.write_bytes(content)
        verify_model_files(snapshot, receipt)
        with self.assertRaises(ValueError):
            verify_model_files(
                snapshot, {**receipt, "files": {"../outside": {"bytes": 1, "sha256": "0" * 64}}}
            )
        with self.assertRaisesRegex(ValueError, "integrity"):
            verify_model_files(
                snapshot,
                {**receipt, "files": {"config.json": {"bytes": len(content), "sha256": "0" * 64}}},
            )


if __name__ == "__main__":
    unittest.main()
