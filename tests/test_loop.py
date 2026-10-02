import sys
import unittest
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_edr import Action, Budget, Observation, Rendered, RepairLoop, Verdict
from ocr_edr.loop import digest


class Renderer:
    def render(self, observation):
        return Rendered("fixture.png", digest(observation.prediction), "fixture")


class Judge:
    def assess(self, observation, rendered):
        return Verdict("good" if observation.prediction == "x^2" else "bad", 1, "fixture")


class Policy:
    def __init__(self, actions):
        self.actions = iter(actions)

    def act(self, observation, rendered, verdict, history):
        return next(self.actions, Action("stop"))


class LoopTests(unittest.TestCase):
    def run_loop(self, actions, initial="x^3", renderer=None, judge=None, budget=None):
        return RepairLoop(renderer or Renderer(), judge or Judge(), Policy(actions), budget).run(
            Observation("sample", "formula", "source.png", initial)
        )

    def test_no_reference_in_policy_observation(self):
        self.assertNotIn("reference", {f.name for f in fields(Observation)})

    def test_paper_diagnostic_actions_keep_prediction_unchanged(self):
        out = self.run_loop([Action("diagnose_scope"), Action("localize"), Action("stop")])
        self.assertEqual(out.final_prediction, "x^3")
        self.assertEqual(out.steps, 3)

    def test_span_patch_can_target_repeated_symbols(self):
        class RepeatedSymbolJudge:
            def assess(self, observation, rendered):
                return Verdict("good" if observation.prediction == "x+x^2" else "bad", 1, "fixture")

        out = self.run_loop(
            [Action("patch", "x", "x^2", start=2, end=3), Action("request_render"), Action("stop")],
            initial="x+x",
            judge=RepeatedSymbolJudge(),
        )
        self.assertEqual(out.final_prediction, "x+x^2")
        self.assertEqual(out.status, "accepted")
        invalid = self.run_loop([Action("patch", "x", "y", start=-1, end=1)])
        self.assertEqual(invalid.status, "invalid_patch_span_rollback")

    def test_good_input_preserved_without_policy_or_extra_render(self):
        out = self.run_loop([Action("global_patch", replacement="bad")], initial="x^2")
        self.assertEqual((out.final_prediction, out.steps, out.renders), ("x^2", 0, 1))

    def test_edit_requires_new_render_before_acceptance(self):
        stale = self.run_loop([Action("patch", "3", "2"), Action("stop")])
        fresh = self.run_loop([Action("patch", "3", "2"), Action("request_render")])
        self.assertEqual(stale.final_prediction, "x^3")
        self.assertEqual(stale.status, "unverified_rollback")
        self.assertEqual(fresh.final_prediction, "x^2")
        self.assertEqual(fresh.renders, 2)

    def test_wrong_patch_and_ambiguous_local_patch_rollback(self):
        wrong = self.run_loop([Action("patch", "3", "4"), Action("request_render"), Action("stop")])
        ambiguous = self.run_loop([Action("patch", "x", "y")], initial="x+x")
        self.assertEqual(wrong.final_prediction, "x^3")
        self.assertEqual(ambiguous.status, "ambiguous_patch_rollback")

    def test_step_and_render_budget_do_not_return_partial_edits(self):
        action = [Action("patch", "3", "2"), Action("request_render")]
        a = self.run_loop(action, budget=Budget(max_steps=1))
        b = self.run_loop(action, budget=Budget(max_renders=1))
        self.assertEqual(a.status, "step_budget_rollback")
        self.assertEqual(b.status, "render_budget_rollback")
        self.assertEqual(a.final_prediction, b.final_prediction)

    def test_renderer_failure_and_mismatched_evidence_rollback(self):
        class Broken:
            def render(self, observation):
                raise RuntimeError("renderer unavailable")

        class Mismatched:
            def render(self, observation):
                return Rendered("old.png", digest("old markup"), "fixture")

        for renderer in (Broken(), Mismatched()):
            out = self.run_loop([], renderer=renderer)
            self.assertEqual(out.status, "backend_error_rollback")
            self.assertEqual(out.final_prediction, "x^3")
            self.assertEqual(out.renders, 1)

    def test_low_confidence_good_verdict_is_not_accepted(self):
        class Uncertain:
            def assess(self, observation, rendered):
                return Verdict("good", 0.5, "fixture")

        out = self.run_loop(
            [Action("global_patch", replacement="x^2"), Action("request_render")], judge=Uncertain()
        )
        self.assertEqual(out.final_prediction, "x^3")

    def test_invalid_budgets_and_verdicts_rejected(self):
        for kwargs in [{"max_steps": 0}, {"max_renders": -1}, {"accept_confidence": float("nan")}]:
            with self.assertRaises(ValueError):
                Budget(**kwargs)
        with self.assertRaises(ValueError):
            Verdict("good", float("inf"), "fixture")


if __name__ == "__main__":
    unittest.main()
