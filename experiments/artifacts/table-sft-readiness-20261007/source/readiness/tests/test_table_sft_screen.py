import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from ocr_edr.table_sft_screen import (
        adapt_table_call,
        load_table_screen_inputs,
        validate_table_adapter,
    )
except ImportError:
    adapt_table_call = load_table_screen_inputs = validate_table_adapter = None


@unittest.skipUnless(importlib.util.find_spec("lxml"), "optional table dependency")
class TableScreenBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.case = {
            "sample_id": "p1-cell",
            "family_id": "p1",
            "source_image": "images/PMC1.png",
            "source_sha256": hashlib.sha256(b"source").hexdigest(),
            "prediction": "<table><tr><td>A</td><td>19</td></tr></table>",
        }

    def call(self, raw, tokens=20):
        from ocr_edr.table_supervision import table_action_prompt

        prompt = table_action_prompt(self.case["prediction"])
        return {
            **self.case,
            "condition": "base",
            "raw_output": raw,
            "input_tokens": 300,
            "output_tokens": tokens,
            "generation_seconds": 2.5,
            "prompt": prompt,
            "messages": [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
            ],
            "ordered_image_sha256": [self.case["source_sha256"]],
        }

    def adapt(self, call, renderer=None):
        self.assertTrue(callable(adapt_table_call))
        return adapt_table_call(
            self.case, call, renderer=renderer or (lambda observation: None), max_new_tokens=192
        )

    def test_atomic_edit_preserves_other_cells_and_stop_preserves_bytes(self):
        call = self.call('{"action":"replace_cell","row":0,"cell":1,"text":"17"}')
        before = copy.deepcopy(call)
        result = self.adapt(call)
        self.assertEqual(
            result["final_prediction"], "<table><tr><td>A</td><td>17</td></tr></table>"
        )
        self.assertEqual(result["trace"][0]["generation_seconds"], 2.5)
        self.assertEqual(result["trace"][0]["contract"], "accepted_contract")
        self.assertEqual(call, before)
        result = self.adapt(self.call('{"action":"stop"}'))
        self.assertEqual(result["final_prediction"], self.case["prediction"])

    def test_truncation_malformed_duplicate_keys_and_bad_address_roll_back_and_keep_cost(self):
        for raw, tokens in [
            ('{"action":"replace_cell","row":0,"cell":1,"text":"17"}', 192),
            ('{"action":"stop"} {"action":"stop"}', 20),
            ('```json\n{"action":"stop"}\n```', 20),
            ('{"action":"delete_row","action":"stop"}', 20),
            ('{"action":"replace_cell","row":0,"cell":9,"text":"17"}', 20),
            ('{"action":"stop","reference":"hidden"}', 20),
        ]:
            with self.subTest(raw=raw, tokens=tokens):
                result = self.adapt(self.call(raw, tokens))
                self.assertEqual(result["final_prediction"], self.case["prediction"])
                self.assertIsNotNone(result["trace"][0]["adapter_error"])
                self.assertEqual(result["trace"][0]["output_tokens"], tokens)
                self.assertEqual(result["trace"][0]["generation_seconds"], 2.5)
                self.assertEqual(result["trace"][0]["hit_length_cap"], tokens >= 192)

    def test_render_failure_rolls_back_with_candidate_and_cost_retained(self):
        def fail(observation):
            raise ValueError("fixture render failure")

        result = self.adapt(
            self.call('{"action":"replace_cell","row":0,"cell":1,"text":"17"}'), fail
        )
        self.assertEqual(result["final_prediction"], self.case["prediction"])
        self.assertIn("17", result["trace"][0]["candidate"])
        self.assertIn("fixture render failure", result["trace"][0]["render_error"])

    def test_prompt_image_identity_and_initial_drift_are_fatal(self):
        for change in [
            {"prompt": "use hidden target"},
            {"messages": []},
            {"ordered_image_sha256": ["0" * 64]},
            {"prediction": "different initial"},
            {"sample_id": "other"},
            {"output_tokens": -1},
            {"generation_seconds": float("nan")},
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.adapt({**self.call('{"action":"stop"}'), **change})

    def test_loader_needs_no_reference_file_and_rejects_role_image_or_schema_drift(self):
        self.assertTrue(callable(load_table_screen_inputs))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / self.case["source_image"]).write_bytes(b"source")
            source = {
                "family_id": "p1",
                "role": "model_dev",
                "document_id": "PMC1",
                "source_image": self.case["source_image"],
                "source_sha256": self.case["source_sha256"],
            }

            def save(case, identity):
                for name, row in [
                    ("model_dev-inputs.jsonl", case),
                    ("selected_sources.jsonl", identity),
                ]:
                    (root / name).write_text(json.dumps(row) + "\n")
                (root / "dataset.json").write_text(
                    json.dumps(
                        {
                            "roles": {"model_dev": {"cases": 1, "sources": 1}},
                            "file_sha256": {
                                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in ["model_dev-inputs.jsonl", "selected_sources.jsonl"]
                            },
                        }
                    )
                )

            save(self.case, source)
            self.assertEqual(load_table_screen_inputs(root), [self.case])
            for case, identity in [
                ({**self.case, "reference": "hidden"}, source),
                (self.case, {**source, "role": "locked_evaluation"}),
                (self.case, {**source, "source_sha256": "0" * 64}),
            ]:
                save(case, identity)
                with self.assertRaises(ValueError):
                    load_table_screen_inputs(root)


class TableAdapterProvenanceTests(unittest.TestCase):
    def test_only_completed_same_protocol_table_adapter_is_accepted(self):
        self.assertTrue(callable(validate_table_adapter))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            for name in ["adapter_config.json", "adapter_model.safetensors"]:
                (checkpoint / name).write_bytes(b"fixture")
            cfg = {"study": "table-fixture", "steps": 381, "model_revision": "a" * 40}
            receipt = {
                "status": "completed",
                "arm": "all",
                "study": "table-fixture",
                "config": cfg,
                "config_sha256": "b" * 64,
                "completed_steps": 381,
                "dev_optimizer_examples": 0,
                "calibration_locked_optimizer_examples": 0,
                "admission_sha256": "c" * 64,
                "checkpoint_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir()
                },
            }

            def validate(value):
                (root / "run.json").write_text(json.dumps(value))
                return validate_table_adapter(
                    root,
                    condition="all",
                    config=cfg,
                    config_sha256="b" * 64,
                    admission_sha256="c" * 64,
                )

            self.assertEqual(validate(receipt), receipt["checkpoint_sha256"])
            for change in [
                {"status": "failed"},
                {"completed_steps": 380},
                {"study": "formula"},
                {"dev_optimizer_examples": 1},
                {"calibration_locked_optimizer_examples": 1},
                {"admission_sha256": "d" * 64},
                {"config_sha256": "d" * 64},
                {"checkpoint_sha256": {}},
            ]:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    validate({**receipt, **change})
            (checkpoint / "adapter_model.safetensors").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                validate(receipt)


@unittest.skipUnless(importlib.util.find_spec("lxml"), "optional table dependency")
class TableCliFailureTests(unittest.TestCase):
    def test_training_resource_failure_records_zero_steps_without_checkpoint(self):
        from ocr_edr.table_supervision import table_action_prompt

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, admission = root / "data", root / "admission"
            (data / "images").mkdir(parents=True)
            admission.mkdir()
            sources = []
            hashes = {}
            for role, family in [("train", "a"), ("model_dev", "b")]:
                image = data / "images" / (family + ".png")
                image.write_bytes(family.encode())
                source_hash = hashlib.sha256(image.read_bytes()).hexdigest()
                sources.append(
                    {
                        "family_id": family,
                        "document_id": "PMC" + family,
                        "role": role,
                        "source_image": "images/" + image.name,
                        "source_sha256": source_hash,
                    }
                )
                rows = []
                for variant, candidate, target in [
                    ("preservation", "<table><tr><td>17</td></tr></table>", {"action": "stop"}),
                    (
                        "cell_perturbation",
                        "<table><tr><td>19</td></tr></table>",
                        {"action": "replace_cell", "row": 0, "cell": 0, "text": "17"},
                    ),
                    (
                        "extra_row",
                        "<table><tr><td>17</td></tr><tr><td>17</td></tr></table>",
                        {"action": "delete_row", "row": 1},
                    ),
                ]:
                    rows.append(
                        {
                            "sample_id": family + variant,
                            "family_id": family,
                            "document_id": "PMC" + family,
                            "role": role,
                            "modality": "table",
                            "source_image": image.name,
                            "source_sha256": source_hash,
                            "candidate": candidate,
                            "prompt": table_action_prompt(candidate),
                            "target": json.dumps(target),
                            "variant": variant,
                            "target_provenance": "published_annotation_controlled_corruption_not_teacher_or_native",
                        }
                    )
                path = admission / (role + "-sft.jsonl")
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            inventory = data / "selected_sources.jsonl"
            inventory.write_text("".join(json.dumps(s) + "\n" for s in sources))
            (data / "dataset.json").write_text(
                json.dumps(
                    {
                        "file_sha256": {
                            inventory.name: hashlib.sha256(inventory.read_bytes()).hexdigest(),
                        }
                    }
                )
            )
            policy = root / "policy.yaml"
            policy.write_text("fixture: true\n")
            (admission / "admission.json").write_text(
                json.dumps(
                    {
                        "status": "admitted_as_published_weak_supervision",
                        "excluded_families": {},
                        "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
                        "file_sha256": hashes,
                    }
                )
            )
            revision = "f" * 40
            model = root / revision
            model.mkdir()
            (model / "fixture.txt").write_bytes(b"fixture")
            integrity = root / "model.json"
            integrity.write_text(
                json.dumps(
                    {
                        "revision": revision,
                        "model": "Qwen/Qwen2-VL-2B-Instruct",
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
                        "study": "training-resource-fixture",
                        "model_revision": revision,
                        "seed": 1,
                        "passes_per_document": 12,
                        "steps": 12,
                        "gradient_accumulation": 1,
                        "data": {
                            "admitted_train_records": 3,
                            "admitted_train_documents": 1,
                            "model_dev_records": 3,
                            "model_dev_documents": 1,
                            "excluded_training_families": [],
                            "admission_policy": str(policy),
                        },
                    }
                )
            )
            script = Path(__file__).resolve().parents[1] / "scripts/train_table_sft.py"
            output = root / "failed"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(data),
                    "--admission",
                    str(admission),
                    "--config",
                    str(config),
                    "--model-path",
                    str(model),
                    "--model-receipt",
                    str(integrity),
                    "--arm",
                    "all",
                    "--output",
                    str(output),
                ],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"},
                capture_output=True,
                text=True,
                timeout=45,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((output / "run.json").exists(), result.stderr)
            receipt = json.loads((output / "run.json").read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt.get("completed_steps"), 0)
            self.assertFalse((output / "checkpoint").exists())

    def test_missing_model_runtime_leaves_initialization_failure_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            (dataset / "images").mkdir(parents=True)
            image = dataset / "images/PMC1.png"
            image.write_bytes(b"source")
            case = {
                "sample_id": "p1-keep",
                "family_id": "p1",
                "source_image": "images/PMC1.png",
                "source_sha256": hashlib.sha256(b"source").hexdigest(),
                "prediction": "<table><tr><td>1</td></tr></table>",
            }
            source = {k: case[k] for k in ["family_id", "source_image", "source_sha256"]}
            source.update(role="model_dev", document_id="PMC1")
            files = {"model_dev-inputs.jsonl": case, "selected_sources.jsonl": source}
            for name, value in files.items():
                (dataset / name).write_text(json.dumps(value) + "\n")
            (dataset / "dataset.json").write_text(
                json.dumps(
                    {
                        "roles": {"model_dev": {"cases": 1, "sources": 1}},
                        "file_sha256": {
                            n: hashlib.sha256((dataset / n).read_bytes()).hexdigest() for n in files
                        },
                    }
                )
            )
            revision = "f" * 40
            model = root / revision
            model.mkdir()
            (model / "fixture.txt").write_bytes(b"fixture")
            integrity = root / "model.json"
            integrity.write_text(
                json.dumps(
                    {
                        "revision": revision,
                        "model": "Qwen/Qwen2-VL-2B-Instruct",
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
                        "study": "failure-fixture",
                        "model_revision": revision,
                        "seed": 1,
                        "min_pixels": 100352,
                        "max_pixels": 200704,
                        "inference": {"max_new_tokens": 192, "do_sample": False},
                        "data": {"model_dev_documents": 1, "model_dev_records": 1},
                    }
                )
            )
            script = Path(__file__).resolve().parents[1] / "scripts/run_table_sft_screen.py"
            output = root / "failed"
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"}
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(dataset),
                    "--config",
                    str(config),
                    "--model-path",
                    str(model),
                    "--model-receipt",
                    str(integrity),
                    "--condition",
                    "base",
                    "--device",
                    "cpu",
                    "--output",
                    str(output),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=45,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((output / "run.json").exists(), result.stderr)
            receipt = json.loads((output / "run.json").read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["completed_calls"], 0)
            self.assertIn("error_type", receipt)


if __name__ == "__main__":
    unittest.main()
