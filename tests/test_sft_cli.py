import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_edr.formula_pilot import make_prompt


class TrainingFailureReceiptTests(unittest.TestCase):
    def test_no_cuda_or_missing_training_dependency_leaves_failed_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            (data / "supervision").mkdir(parents=True)
            for split, family in [("train", "a"), ("dev", "b")]:
                folder = data / split
                folder.mkdir()
                image = folder / (family + ".png")
                image.write_bytes(f"fixture-{family}".encode())
                rows = []
                for variant in ["preservation", "controlled_error", "native_parser_prediction"]:
                    candidate = "x^2" if variant == "preservation" else "x^3"
                    rows.append(
                        {
                            "sample_id": family + variant,
                            "family_id": family,
                            "split": split,
                            "modality": "formula",
                            "source_image": f"../{split}/{family}.png",
                            "source_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                            "candidate": candidate,
                            "prompt": make_prompt(candidate, False),
                            "target": "<latex>x^2</latex>",
                            "variant": variant,
                            "target_provenance": "released_annotation_or_preservation_copy",
                        }
                    )
                p = data / "supervision" / (split + "-sft.jsonl")
                p.write_text("".join(json.dumps(row) + "\n" for row in rows))
            summary = {
                "splits": {
                    split: {
                        "sft_sha256": hashlib.sha256(
                            (data / "supervision" / (split + "-sft.jsonl")).read_bytes()
                        ).hexdigest()
                    }
                    for split in ["train", "dev"]
                }
            }
            (data / "supervision/summary.json").write_text(json.dumps(summary))
            revision = "f" * 40
            model = root / revision
            model.mkdir()
            (model / "fixture.txt").write_text("fixture")
            receipt = root / "model.json"
            receipt.write_text(
                json.dumps(
                    {
                        "revision": revision,
                        "files": {
                            "fixture.txt": {
                                "bytes": 7,
                                "sha256": hashlib.sha256(b"fixture").hexdigest(),
                            }
                        },
                    }
                )
            )
            config = root / "config.yaml"
            config.write_text(
                json.dumps(
                    {
                        "study": "failure_fixture",
                        "model_revision": revision,
                        "steps": 3,
                        "gradient_accumulation": 1,
                        "seed": 1,
                        "data": {"train_records_all": 3, "dev_records": 3},
                    }
                )
            )
            script = Path(__file__).resolve().parents[1] / "scripts/train_formula_sft.py"
            output = root / "failed-run"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--config",
                    str(config),
                    "--dataset",
                    str(data),
                    "--model-path",
                    str(model),
                    "--model-receipt",
                    str(receipt),
                    "--arm",
                    "all",
                    "--output",
                    str(output),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=45,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((output / "run.json").is_file(), result.stderr)
            saved = json.loads((output / "run.json").read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertIn("error_type", saved)
            self.assertFalse((output / "checkpoint").exists())


if __name__ == "__main__":
    unittest.main()
