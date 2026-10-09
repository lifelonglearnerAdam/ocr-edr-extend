import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from ocr_edr.prompt_protocol import validate_prompt_protocol
except ImportError:
    validate_prompt_protocol = None


class PromptProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "precision_profile": "nf4_lora_8gb",
            "inference": {
                "prompt_format": "literal_examples",
                "logits_projection": "last_position_only",
                "max_new_tokens": 192,
                "do_sample": False,
            },
            "data": {"model_dev_records": 103, "model_dev_documents": 32},
        }
        self.protocol = {
            "study": "fixed_prompt_trial",
            "training_config_sha256": "a" * 64,
            "prompt_formats": ["literal_examples", "descriptive_schema"],
            "conditions": ["base", "all", "no_explicit_preservation"],
            "cases": 103,
            "documents": 32,
            "max_new_tokens": 192,
            "do_sample": False,
            "post_hoc_exploratory": True,
            "adapter_model_sha256": {"all": "b" * 64, "no_explicit_preservation": "c" * 64},
        }

    def check(
        self,
        *,
        prompt="descriptive_schema",
        protocol=True,
        config=None,
        condition="all",
        checkpoint=None,
    ):
        self.assertTrue(callable(validate_prompt_protocol))
        return validate_prompt_protocol(
            self.cfg if config is None else config,
            config_sha256="a" * 64,
            prompt_format=prompt,
            condition=condition,
            protocol=self.protocol if protocol is True else protocol,
            checkpoint_hashes=(
                {"adapter_model.safetensors": "b" * 64} if checkpoint is None else checkpoint
            ),
        )

    def test_explicit_trial_allows_both_prompts_with_same_frozen_checkpoint(self):
        self.assertEqual(self.check()["study"], "fixed_prompt_trial")
        self.assertEqual(self.check(prompt="literal_examples"), self.protocol)
        self.assertEqual(self.check(condition="base", checkpoint={}), self.protocol)

    def test_default_nf4_still_rejects_undeclared_prompt_or_projection_change(self):
        with self.assertRaises(ValueError):
            self.check(protocol=None)
        self.assertIsNone(self.check(prompt="literal_examples", protocol=None))
        bad = copy.deepcopy(self.cfg)
        bad["inference"]["logits_projection"] = "full"
        with self.assertRaises(ValueError):
            self.check(config=bad)

    def test_protocol_budget_hash_coverage_and_adapter_drift_are_rejected(self):
        for key, value in [
            ("training_config_sha256", "d" * 64),
            ("max_new_tokens", 193),
            ("do_sample", True),
            ("cases", 102),
            ("conditions", ["all"]),
            ("prompt_formats", ["descriptive_schema"]),
            ("reference_html", "hidden target"),
        ]:
            changed = {**self.protocol, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(protocol=changed)
        with self.assertRaises(ValueError):
            self.check(checkpoint={"adapter_model.safetensors": "d" * 64})


if __name__ == "__main__":
    unittest.main()
