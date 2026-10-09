import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from ocr_edr.research_dashboard import render_dashboard, validate_snapshot
except ImportError:
    render_dashboard = validate_snapshot = None

try:
    from ocr_edr.dashboard_evidence import (
        verify_completion_binding,
        verify_four_arm_coverage,
        verify_prompt_stage,
    )
except ImportError:
    verify_completion_binding = verify_four_arm_coverage = verify_prompt_stage = None


class DashboardIntegrityTests(unittest.TestCase):
    def snapshot(self):
        return {
            "schema_version": 1,
            "updated_at": "2026-10-09T00:00:00+00:00",
            "table_screen": {
                "cases": 103,
                "documents": 32,
                "arms": [
                    {"arm": arm, "teds": 0.9, "repairs": 0, "regressions": 0, "changed": 0}
                    for arm in ["unchanged_0", "base", "all", "no_explicit_preservation"]
                ],
            },
            "senior": {
                "evaluation": {
                    "n": 1800,
                    "metrics": [
                        {
                            "model": "Qwen3.5",
                            "condition": "Normal",
                            "verdict_percent": 99.72,
                            "strict_joint_percent": 97.44,
                            "admissible_joint_percent": 98.17,
                        }
                    ],
                }
            },
        }

    def test_html_embeds_data_without_script_breakout_and_works_offline(self):
        self.assertTrue(callable(render_dashboard))
        snapshot = self.snapshot()
        snapshot["note"] = '</script><script>alert("x")</script>'
        template = (
            '<html><script type="application/json" id="research-data">{{DATA}}</script></html>'
        )
        result = render_dashboard(snapshot, template, {})
        self.assertNotIn("</script><script>alert", result)
        encoded = result.split('id="research-data">', 1)[1].split("</script>", 1)[0]
        self.assertEqual(json.loads(encoded), snapshot)
        self.assertNotIn("{{DATA}}", result)

    def test_partial_invalid_or_secret_snapshot_is_rejected(self):
        self.assertTrue(callable(validate_snapshot))
        for change in ["partial", "nan", "rates", "secrets"]:
            snapshot = self.snapshot()
            if change == "partial":
                snapshot["table_screen"]["arms"].pop()
            elif change == "nan":
                snapshot["table_screen"]["arms"][0]["teds"] = float("nan")
            elif change == "rates":
                snapshot["senior"]["evaluation"]["metrics"][0]["verdict_percent"] = 101
            else:
                snapshot["private_path"] = "/home/zimozeng/.ssh/id_rsa"
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_snapshot(snapshot)

    def test_only_vetted_local_png_assets_are_embedded(self):
        self.assertTrue(callable(render_dashboard))
        result = render_dashboard(
            self.snapshot(),
            '<img src="{{ASSET:plot.png}}">{{DATA}}',
            {"plot.png": b"\x89PNG\r\n\x1a\nfixture"},
        )
        self.assertIn("data:image/png;base64,", result)
        with self.assertRaises(ValueError):
            render_dashboard(self.snapshot(), "{{ASSET:missing.png}}{{DATA}}", {})


class EvidenceBindingTests(unittest.TestCase):
    def test_completed_seal_rejects_changed_stats_training_or_evaluation(self):
        self.assertTrue(callable(verify_completion_binding))
        hashes = {
            "training_all": "a",
            "training_no_explicit_preservation": "b",
            "evaluation": "c",
            "statistics_report": "d",
        }
        completion = {
            "status": "completed",
            "scientific_outputs_verified": True,
            "evidence_sha256": hashes,
        }
        verify_completion_binding(completion, hashes)
        for name in hashes:
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify_completion_binding(completion, {**hashes, name: "changed"})

    def test_partial_prompt_cohort_or_stale_pipeline_seal_is_rejected(self):
        self.assertTrue(callable(verify_four_arm_coverage))
        arms = ["unchanged_0", "base", "all", "no_explicit_preservation"]
        cases = [
            {"arm": arm, "sample_id": str(i), "document_id": str(i % 32)}
            for arm in arms
            for i in range(103)
        ]
        summary = [{"arm": arm, "variant": "all", "cases": 103, "documents": 32} for arm in arms]
        evaluation = {"cases": cases, "summary": summary}
        verify_four_arm_coverage(evaluation)
        with self.assertRaises(ValueError):
            verify_four_arm_coverage({"cases": cases, "summary": summary[:3]})
        with self.assertRaises(ValueError):
            verify_four_arm_coverage({"cases": cases[:-1], "summary": summary})
        pipeline = {
            "completed_stages": [
                {"stage": "descriptive_schema_evaluation", "receipt_sha256": "sealed"}
            ]
        }
        verify_prompt_stage(pipeline, "descriptive_schema", "sealed")
        with self.assertRaises(ValueError):
            verify_prompt_stage(pipeline, "descriptive_schema", "modified")


if __name__ == "__main__":
    unittest.main()
