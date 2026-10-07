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
    from ocr_edr.table_screen_evaluation import evaluate_table_screen
except ImportError:
    evaluate_table_screen = None


@unittest.skipUnless(importlib.util.find_spec("lxml"), "optional table dependency")
class TableScreenEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.good = "<table><tr><td>A</td><td>17</td></tr></table>"
        self.bad = "<table><tr><td>A</td><td>19</td></tr></table>"
        self.inputs, self.references, self.predictions = [], [], []
        for sample, initial, variant, action, family, doc in [
            ("a-keep", self.good, "preservation", {"action": "stop"}, "a", "PMC1"),
            (
                "a-edit",
                self.bad,
                "cell_perturbation",
                {"action": "replace_cell", "row": 0, "cell": 1, "text": "17"},
                "a",
                "PMC1",
            ),
            ("b-keep", self.good, "preservation", {"action": "stop"}, "b", "PMC2"),
        ]:
            self.inputs.append({"sample_id": sample, "family_id": family, "prediction": initial})
            self.references.append(
                {
                    "sample_id": sample,
                    "family_id": family,
                    "document_id": doc,
                    "role": "model_dev",
                    "variant": variant,
                    "reference": self.good,
                    "target_action": action,
                }
            )
            for arm in ["unchanged_0", "base"]:
                trace = (
                    []
                    if arm == "unchanged_0"
                    else [
                        {
                            "generation_seconds": 2.0,
                            "input_tokens": 100,
                            "output_tokens": 10,
                            "action": action,
                            "adapter_error": None,
                            "render_error": None,
                            "render_attempts": 1,
                            "render_seconds": 0.5,
                            "hit_length_cap": False,
                        }
                    ]
                )
                self.predictions.append(
                    {
                        "sample_id": sample,
                        "family_id": family,
                        "arm": arm,
                        "initial_prediction": initial,
                        "final_prediction": initial if arm == "unchanged_0" else self.good,
                        "trace": trace,
                    }
                )

    def evaluate(self, predictions=None, refs=None):
        self.assertTrue(callable(evaluate_table_screen))
        return evaluate_table_screen(
            self.predictions if predictions is None else predictions,
            self.inputs,
            self.references if refs is None else refs,
            ["unchanged_0", "base"],
            normalize=str.strip,
            score=lambda p, r: {"teds": float(p == r), "teds_structure": 1.0},
        )

    def test_document_means_costs_and_action_address_are_reported_separately(self):
        result = self.evaluate()
        base = next(r for r in result["summary"] if r["arm"] == "base" and r["variant"] == "all")
        self.assertEqual(base["cases"], 3)
        self.assertEqual(base["documents"], 2)
        self.assertAlmostEqual(base["case_mean_delta_teds"], 1 / 3)
        self.assertEqual(base["document_mean_delta_teds"], 0.25)
        self.assertEqual(base["generation_seconds"], 6.0)
        self.assertEqual(base["render_seconds"], 1.5)
        self.assertEqual(base["exact_target_actions"], 3)
        self.assertEqual(base["target_address_matches"], 3)
        self.assertEqual(base["accepted_actions"], 3)
        self.assertEqual({r["document_id"] for r in result["cases"]}, {"PMC1", "PMC2"})

    def test_right_address_wrong_text_and_render_rejection_are_not_successes(self):
        rows = copy.deepcopy(self.predictions)
        rows[3]["trace"][0]["action"] = {
            "action": "replace_cell",
            "row": 0,
            "cell": 1,
            "text": "18",
        }
        rows[3]["trace"][0]["render_error"] = "failed"
        rows[3]["final_prediction"] = self.bad
        result = self.evaluate(rows)
        row = next(r for r in result["cases"] if r["sample_id"] == "a-edit" and r["arm"] == "base")
        self.assertTrue(row["target_address_matches"])
        self.assertFalse(row["exact_target_actions"])
        self.assertFalse(row["accepted_actions"])
        self.assertEqual(row["delta_teds"], 0)
        self.assertEqual(row["model_calls"], 1)

    def test_wrong_role_or_nonreplayable_target_cannot_enter_development_scoring(self):
        for change in [{"role": "gate_calibration"}, {"target_action": {"action": "stop"}}]:
            refs = copy.deepcopy(self.references)
            refs[1].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.evaluate(refs=refs)


@unittest.skipUnless(os.environ.get("OCR_EDR_TEST_TABLE") == "1", "explicit table CLI integration")
class TableScreenCliPairingTests(unittest.TestCase):
    def test_complete_cli_replay_rejects_gpu_cuda_admission_or_peft_mismatches(self):
        from PIL import Image

        from ocr_edr.table_sft_screen import table_messages

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            (data / "images").mkdir(parents=True)
            Image.new("RGB", (40, 30), "white").save(data / "images/PMC1.png")
            case = {
                "sample_id": "p1-preservation",
                "family_id": "p1",
                "source_image": "images/PMC1.png",
                "source_sha256": hashlib.sha256(
                    (data / "images/PMC1.png").read_bytes()
                ).hexdigest(),
                "prediction": "<table><tr><td>17</td></tr></table>",
            }
            source = {k: case[k] for k in ["family_id", "source_image", "source_sha256"]}
            source.update(role="model_dev", document_id="PMC1")
            ref = {
                "sample_id": case["sample_id"],
                "family_id": "p1",
                "document_id": "PMC1",
                "role": "model_dev",
                "variant": "preservation",
                "reference": case["prediction"],
                "target_action": {"action": "stop"},
            }
            files = {
                "model_dev-inputs.jsonl": case,
                "selected_sources.jsonl": source,
                "model_dev-references.jsonl": ref,
            }
            for name, row in files.items():
                (data / name).write_text(json.dumps(row) + "\n")
            (data / "dataset.json").write_text(
                json.dumps(
                    {
                        "roles": {"model_dev": {"cases": 1, "sources": 1}},
                        "file_sha256": {
                            n: hashlib.sha256((data / n).read_bytes()).hexdigest() for n in files
                        },
                    }
                )
            )
            messages, prompt = table_messages(case["prediction"])
            common = {
                "status": "completed",
                "cases": 1,
                "completed_calls": 1,
                "reference_access": "none",
                "calibration_locked_images_loaded": False,
                "device": "cuda:0",
                "dtype": "bfloat16",
                "attention": "sdpa",
                "cpu_threads": 4,
                "source_sha256": {"fixture": "0" * 64},
                "config_sha256": "1" * 64,
                "model_receipt_sha256": "2" * 64,
                "config": {"inference": {"max_new_tokens": 192}},
                "dataset_sha256": hashlib.sha256((data / "dataset.json").read_bytes()).hexdigest(),
                "input_sha256": hashlib.sha256(
                    (data / "model_dev-inputs.jsonl").read_bytes()
                ).hexdigest(),
                "gpu": "NVIDIA GeForce RTX 3090",
                "cuda_runtime": "12.4",
                "versions": {"torch": "fixture", "transformers": "fixture", "Pillow": "fixture"},
                "admission_sha256": None,
            }
            run_dirs, original = [], []
            for arm in ["base", "all", "no_explicit_preservation"]:
                folder = root / arm
                folder.mkdir()
                call = {
                    **case,
                    "condition": arm,
                    "messages": messages,
                    "prompt": prompt,
                    "ordered_image_sha256": [case["source_sha256"]],
                    "raw_output": '{"action":"stop"}',
                    "input_tokens": 100,
                    "output_tokens": 7,
                    "generation_seconds": 1.5,
                }
                (folder / "calls.jsonl").write_text(json.dumps(call) + "\n")
                run = copy.deepcopy(common)
                run.update(
                    condition=arm,
                    calls_sha256=hashlib.sha256((folder / "calls.jsonl").read_bytes()).hexdigest(),
                )
                if arm != "base":
                    run["admission_sha256"] = "3" * 64
                    run["versions"]["peft"] = "0.17.1"
                (folder / "run.json").write_text(json.dumps(run))
                original.append(run)
                run_dirs.append(folder)

            def run_cli(label):
                return subprocess.run(
                    [
                        sys.executable,
                        str(repo / "scripts/evaluate_table_sft_screen.py"),
                        "--dataset",
                        str(data),
                        "--runs",
                        *map(str, run_dirs),
                        "--official-root",
                        str(repo / "data/raw/omnidocbench-source"),
                        "--output",
                        str(root / label),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=45,
                )

            good = run_cli("compatible")
            self.assertEqual(good.returncode, 0, good.stderr)
            evaluation = json.loads((root / "compatible/evaluation.json").read_text())
            self.assertEqual(len(evaluation["cases"]), 4)
            self.assertEqual(sum(r["model_calls"] for r in evaluation["cases"]), 3)
            last_calls = run_dirs[-1] / "calls.jsonl"
            original_calls = last_calls.read_bytes()
            for index, change in enumerate(
                [
                    {"gpu": "NVIDIA GeForce RTX 4090"},
                    {"cuda_runtime": "12.1"},
                    {"admission_sha256": "4" * 64},
                    {"prompt_format": "literal_examples"},
                    {"versions": {**original[-1]["versions"], "peft": "0.18.1"}},
                ]
            ):
                last_calls.write_bytes(original_calls)
                changed_run = {**original[-1], **change}
                if "prompt_format" in change:
                    # Each run remains internally valid. The only intended
                    # rejection is pairing two different prompt protocols.
                    literal_call = json.loads(original_calls)
                    messages, prompt = table_messages(
                        case["prediction"], prompt_format="literal_examples"
                    )
                    literal_call.update(messages=messages, prompt=prompt)
                    last_calls.write_text(json.dumps(literal_call) + "\n")
                    changed_run["calls_sha256"] = hashlib.sha256(
                        last_calls.read_bytes()
                    ).hexdigest()
                (run_dirs[-1] / "run.json").write_text(json.dumps(changed_run))
                with self.subTest(change=change):
                    bad = run_cli("mismatch" + str(index))
                    self.assertNotEqual(bad.returncode, 0, "Incompatible runs were silently paired")
                    expected_error = (
                        "SFT conditions differ"
                        if {"admission_sha256", "versions"} & change.keys()
                        else "Paired conditions require"
                    )
                    self.assertIn(expected_error, bad.stderr)


if __name__ == "__main__":
    unittest.main()
