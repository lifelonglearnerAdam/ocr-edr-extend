import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ocr_edr.loop import Rendered, digest
from ocr_edr.sft import sha256

try:
    from ocr_edr.teacher_audit import (
        paired_teacher_views,
        replay_teacher_episode,
        validate_teacher_mapping,
        verify_frozen_files,
    )
except ImportError:
    paired_teacher_views = replay_teacher_episode = validate_teacher_mapping = None
    verify_frozen_files = None


@unittest.skipUnless(importlib.util.find_spec("lxml"), "optional table dependency")
class TeacherAuditTests(unittest.TestCase):
    def fixture(self, root, *, render_fails=False):
        from ocr_edr.teacher_tables import execute_teacher_decision

        initial = "<table><tr><td>A</td><td>19</td></tr></table>"

        def renderer(observation):
            path = root / (digest(observation.prediction) + ".png")
            path.write_bytes(observation.prediction.encode())
            return Rendered(str(path), digest(observation.prediction), "fixture")

        def failure(observation):
            raise RuntimeError("fixture renderer unavailable")

        edit = {
            "case_id": "case1",
            "observed_html_sha256": digest(initial),
            "source_evidence": "Second cell reads 17.",
            "uncertainty": "low",
            "action": {"action": "replace_cell", "row": 0, "cell": 1, "text": "17"},
        }
        event = execute_teacher_decision(
            initial, edit, renderer=failure if render_fails else renderer
        )
        stop = {
            **edit,
            "observed_html_sha256": digest(event["final_html"]),
            "action": {"action": "stop"},
            "uncertainty": "ambiguous" if render_fails else "low",
        }
        terminal = execute_teacher_decision(event["final_html"], stop, renderer=renderer)
        state = {
            "case_id": "case1",
            "html": terminal["final_html"],
            "html_sha256": digest(terminal["final_html"]),
            "edit_attempts": 1,
            "terminal": True,
            "terminal_assessment": "teacher_abstention" if render_fails else "teacher_self_check",
        }
        task = {
            "case_id": "case1",
            "current_html": initial,
            "current_html_sha256": digest(initial),
        }
        return task, [edit, stop], [event, terminal], state, renderer

    def replay(self, fixture, budget=3):
        self.assertTrue(callable(replay_teacher_episode))
        task, decisions, events, state, renderer = fixture
        return replay_teacher_episode(
            task, decisions, events, state, max_edit_attempts=budget, renderer=renderer
        )

    def test_edit_stop_chain_replays_and_retains_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.replay(self.fixture(Path(tmp)))
            self.assertTrue(report["mechanically_valid"])
            self.assertEqual(report["edit_attempts"], 1)
            self.assertEqual(report["fresh_render_matches"], 1)
            self.assertEqual(report["local_edits"], 1)

    def test_state_render_or_observation_tampering_and_budget_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            for change in ["state", "render", "observation", "after_stop", "scope"]:
                modified = copy.deepcopy(fixture)
                if change == "state":
                    modified[3]["html"] = modified[3]["html"].replace("17", "18")
                    modified[3]["html_sha256"] = digest(modified[3]["html"])
                elif change == "render":
                    modified[2][0]["render_image_sha256"] = "0" * 64
                elif change == "observation":
                    modified[1][1]["observed_html_sha256"] = "0" * 64
                elif change == "scope":
                    modified[2][0]["edit_scope"] = "global"
                else:
                    modified[1].append(modified[1][-1])
                    modified[2].append(modified[2][-1])
                with self.subTest(change=change), self.assertRaises(ValueError):
                    self.replay(modified)
            with self.assertRaises(ValueError):
                self.replay(fixture, budget=0)

    def test_original_render_failure_is_retained_when_diagnostic_retry_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp), render_fails=True)
            report = self.replay(fixture)
            self.assertEqual(report["render_failures"], 1)
            self.assertEqual(report["final_html_sha256"], fixture[0]["current_html_sha256"])
            self.assertEqual(report["terminal_assessment"], "teacher_abstention")
            self.assertEqual(report["applied_edits"], 0)

    def test_frozen_manifest_rejects_modified_extra_or_escaping_files(self):
        self.assertTrue(callable(verify_frozen_files))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = root / "one.json"
            item.write_text("first")
            manifest = {"one.json": sha256(item)}
            verify_frozen_files(root, manifest)
            item.write_text("changed")
            with self.assertRaises(ValueError):
                verify_frozen_files(root, manifest)
            manifest = {"one.json": sha256(item)}
            (root / "extra").write_text("unlogged")
            with self.assertRaises(ValueError):
                verify_frozen_files(root, manifest)
            with self.assertRaises(ValueError):
                verify_frozen_files(root, {"../one.json": "0" * 64})

    def test_train_mapping_rejects_duplicates_exclusions_and_heldout_overlap(self):
        self.assertTrue(callable(validate_teacher_mapping))
        sources = [
            {"family_id": "p1", "document_id": "d1", "role": "train", "source_sha256": "a"},
            {"family_id": "p2", "document_id": "d2", "role": "locked", "source_sha256": "b"},
        ]
        mapping = [{"case_id": "c1", "family_id": "p1", "role": "train", "source_sha256": "a"}]
        validate_teacher_mapping(mapping, sources, set())
        for change in ["heldout", "duplicate", "excluded", "document_overlap", "image_overlap"]:
            rows, inventory = copy.deepcopy(mapping), copy.deepcopy(sources)
            excluded = set()
            if change == "heldout":
                rows[0].update(family_id="p2", role="locked", source_sha256="b")
            elif change == "duplicate":
                rows.append({**rows[0], "case_id": "c2"})
            elif change == "excluded":
                excluded.add("p1")
            elif change == "document_overlap":
                inventory[1]["document_id"] = "d1"
            else:
                inventory[1]["source_sha256"] = "a"
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_teacher_mapping(rows, inventory, excluded)

    def test_paired_views_keep_same_targets_and_do_not_import_reference_answers(self):
        self.assertTrue(callable(paired_teacher_views))
        episodes = [
            {
                "case_id": case,
                "source_image": "cases/" + case + "/source.png",
                "source_sha256": case,
                "initial_html": "initial",
                "final_html": "teacher answer",
                "steps": [{"decision": {"action": {"action": "stop"}}}],
            }
            for case in ["ready", "review"]
        ]
        audits = [
            {"case_id": "ready", "provisionally_admitted": True},
            {"case_id": "review", "provisionally_admitted": False},
        ]
        final, trajectory = paired_teacher_views(episodes, audits)
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]["case_id"], trajectory[0]["case_id"])
        self.assertEqual(final[0]["input"], trajectory[0]["input"])
        self.assertEqual(final[0]["target_html"], trajectory[0]["target_html"])
        self.assertNotIn("steps", final[0])
        self.assertEqual(trajectory[0]["steps"], episodes[0]["steps"])
        with self.assertRaises(ValueError):
            paired_teacher_views(episodes, audits + [audits[0]])


if __name__ == "__main__":
    unittest.main()
