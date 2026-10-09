import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ocr_edr.loop import Rendered, digest
from ocr_edr.sft import sha256

try:
    from ocr_edr.teacher_tables import execute_teacher_decision
except ImportError:
    execute_teacher_decision = None


@unittest.skipUnless(importlib.util.find_spec("lxml"), "optional table dependency")
class TeacherTransitionTests(unittest.TestCase):
    initial = "<table><tr><td>A</td><td>19</td></tr></table>"

    def decision(self, action):
        return {
            "case_id": "opaque-1",
            "observed_html_sha256": digest(self.initial),
            "source_evidence": "The visible second cell reads17.",
            "uncertainty": "low",
            "action": action,
        }

    def test_local_patch_preserves_scope_and_binds_a_new_render(self):
        self.assertTrue(callable(execute_teacher_decision))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "render.png"

            def renderer(observation):
                path.write_bytes(observation.prediction.encode())
                return Rendered(str(path), digest(observation.prediction), "test renderer")

            action = {"action": "replace_cell", "row": 0, "cell": 1, "text": "17"}
            result = execute_teacher_decision(
                self.initial, self.decision(action), renderer=renderer
            )
            self.assertTrue(result["applied"])
            self.assertEqual(result["final_html"], "<table><tr><td>A</td><td>17</td></tr></table>")
            self.assertEqual(result["render_prediction_sha256"], digest(result["final_html"]))
            self.assertEqual(result["input_html_sha256"], digest(self.initial))
            self.assertTrue(result["teacher_selfcheck_only"])

    def test_stale_observation_or_render_is_rejected(self):
        self.assertTrue(callable(execute_teacher_decision))
        decision = self.decision({"action": "replace_cell", "row": 0, "cell": 1, "text": "17"})
        with self.assertRaises(ValueError):
            execute_teacher_decision(
                self.initial, {**decision, "observed_html_sha256": "0" * 64}, renderer=None
            )
        result = execute_teacher_decision(
            self.initial,
            decision,
            renderer=lambda observation: Rendered("unused.png", digest(self.initial), "stale"),
        )
        self.assertFalse(result["applied"])
        self.assertEqual(result["final_html"], self.initial)
        self.assertIn("stale", result["render_error"].lower())

    def test_global_patch_is_explicit_and_unsafe_html_cannot_enter(self):
        self.assertTrue(callable(execute_teacher_decision))
        unsafe = self.decision(
            {
                "action": "global_patch",
                "html": '<table><tr><td><img src="https://example.com/x"></td></tr></table>',
            }
        )
        result = execute_teacher_decision(self.initial, unsafe, renderer=None)
        self.assertFalse(result["applied"])
        self.assertEqual(result["final_html"], self.initial)
        self.assertTrue(result["action_error"])
        result = execute_teacher_decision(
            self.initial, self.decision({"action": "stop"}), renderer=None
        )
        self.assertEqual(result["final_html"], self.initial)
        self.assertEqual(result["render_calls"], 0)


@unittest.skipUnless(
    os.environ.get("OCR_EDR_TEST_TABLE") == "1", "explicit teacher CLI integration"
)
class TeacherCliTests(unittest.TestCase):
    def test_preparation_freezes_the_prompt_required_by_the_auditor(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, native, packet = root / "dataset", root / "native", root / "packet"
            dataset.mkdir()
            native.mkdir()
            (dataset / "sources").mkdir()
            (dataset / "admitted-supervision").mkdir()
            inputs, inventory, controlled = [], [], []
            html = "<table><tr><td>fixture</td></tr></table>"

            def write_lines(path, rows):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))

            for number in range(1, 21):
                family = f"p{number:04d}"
                image = dataset / "sources" / (family + ".png")
                Image.new("RGB", (10, 10), (number, 0, 0)).save(image)
                row = {
                    "family_id": family,
                    "source_image": image.relative_to(dataset).as_posix(),
                    "source_sha256": sha256(image),
                }
                inputs.append(row)
                inventory.append({**row, "role": "train", "document_id": f"doc{number}"})
                controlled.append({**row, "sample_id": family + "-keep", "prediction": html})
            write_lines(dataset / "train-source-inputs.jsonl", inputs)
            write_lines(dataset / "selected_sources.jsonl", inventory)
            write_lines(dataset / "train-inputs.jsonl", controlled)
            (dataset / "dataset.json").write_text(
                json.dumps(
                    {
                        "roles": {"train": {"sources": 20}},
                        "file_sha256": {p.name: sha256(p) for p in dataset.glob("*.jsonl")},
                    }
                )
            )
            admission = dataset / "admitted-supervision/admission.json"
            admission.write_text(
                json.dumps(
                    {
                        "status": "admitted_as_published_weak_supervision",
                        "excluded_families": ["p0002"],
                        "documents": 19,
                    }
                )
            )
            write_lines(
                native / "predictions.jsonl",
                [{**row, "prediction": html} for row in inputs if row["family_id"] != "p0002"],
            )
            (native / "run.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "source_role": "train",
                        "completed_sources": 19,
                        "reference_access": "none",
                        "admission_sha256": sha256(admission),
                        "input_sha256": sha256(dataset / "train-source-inputs.jsonl"),
                        "predictions_sha256": sha256(native / "predictions.jsonl"),
                    }
                )
            )
            prompt = root / "instructions.txt"
            prompt.write_text("Inspect SOURCE; do not access references.\n")
            project = Path(__file__).resolve().parents[1]
            result = subprocess.run(
                [
                    sys.executable,
                    str(project / "scripts/prepare_teacher_table_pilot.py"),
                    "--dataset",
                    str(dataset),
                    "--native-run",
                    str(native),
                    "--output",
                    str(packet),
                    "--teacher-instructions",
                    str(prompt),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads((packet / "packet.json").read_text())
            self.assertEqual(metadata["teacher_prompt_sha256"], sha256(prompt))
            self.assertEqual(
                (packet / "teacher-instructions.txt").read_bytes(), prompt.read_bytes()
            )
            self.assertEqual(len(metadata["tasks"]), 12)
            prompt.write_text("Later instructions must not mutate the packet.\n")
            self.assertEqual(
                sha256(packet / "teacher-instructions.txt"), metadata["teacher_prompt_sha256"]
            )
            seal_path = root / "frozen" / "cohort-seal.json"
            seal_command = [
                sys.executable,
                str(project / "scripts/seal_teacher_table_pilot.py"),
                "--packet",
                str(packet),
                "--output",
                str(seal_path),
            ]
            incomplete = subprocess.run(seal_command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(incomplete.returncode, 0)
            self.assertIn("not terminal", incomplete.stderr)
            self.assertFalse(seal_path.exists())
            for task in metadata["tasks"]:
                case = task["case_id"]
                state = json.loads((packet / "cases" / case / "state.json").read_text())
                decision_path = packet / "decisions" / (case + "-00.json")
                decision_path.write_text(
                    json.dumps(
                        {
                            "case_id": case,
                            "observed_html_sha256": state["html_sha256"],
                            "source_evidence": "Synthetic fixture stop, not a research label.",
                            "uncertainty": "low",
                            "action": {"action": "stop"},
                        }
                    )
                )
                executed = subprocess.run(
                    [
                        sys.executable,
                        str(project / "scripts/execute_teacher_table_step.py"),
                        "--packet",
                        str(packet),
                        "--decision",
                        str(decision_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(executed.returncode, 0, executed.stderr)
            sealed = subprocess.run(seal_command, capture_output=True, text=True, timeout=30)
            self.assertEqual(sealed.returncode, 0, sealed.stderr)
            receipt = json.loads(seal_path.read_text())
            self.assertEqual(receipt["terminal_episodes"], 12)
            self.assertEqual(receipt["edit_attempts"], 0)
            from ocr_edr.teacher_audit import verify_frozen_files

            verify_frozen_files(packet, receipt["packet_files_sha256"])
            repeated = subprocess.run(seal_command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(json.loads(seal_path.read_text()), receipt)

    def test_abstaining_on_unrenderable_initial_candidate_still_seals_a_terminal_event(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = "opaque-case"
            work = root / "cases" / case
            work.mkdir(parents=True)
            Image.new("RGB", (10, 10), "white").save(work / "source.png")
            (work / "task.json").write_text(
                json.dumps({"case_id": case, "current_html": "not html"})
            )
            (root / "packet.json").write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "case_id": case,
                                "task_sha256": sha256(work / "task.json"),
                                "source_sha256": sha256(work / "source.png"),
                            }
                        ],
                        "max_edit_attempts_per_case": 3,
                        "teacher_model_label": "fixture",
                        "teacher_effort_label": "fixture",
                    }
                )
            )
            (work / "state.json").write_text(
                json.dumps(
                    {
                        "html": "not html",
                        "html_sha256": digest("not html"),
                        "steps": [],
                        "edit_attempts": 0,
                        "terminal": False,
                    }
                )
            )
            decision = root / "decision.json"
            decision.write_text(
                json.dumps(
                    {
                        "case_id": case,
                        "observed_html_sha256": digest("not html"),
                        "source_evidence": "Source content is unreadable in this fixture.",
                        "uncertainty": "unreadable",
                        "action": {"action": "stop"},
                    }
                )
            )
            script = Path(__file__).resolve().parents[1] / "scripts/execute_teacher_table_step.py"
            result = subprocess.run(
                [sys.executable, str(script), "--packet", str(root), "--decision", str(decision)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state = json.loads((work / "state.json").read_text())
            self.assertTrue(state["terminal"])
            self.assertEqual(state["terminal_assessment"], "teacher_abstention")
            self.assertEqual(len(state["steps"]), 1)
            self.assertIsNone(
                json.loads((work / "latest-view.json").read_text())["current_cell_addresses"]
            )


if __name__ == "__main__":
    unittest.main()
