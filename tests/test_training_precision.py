import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from ocr_edr.training_precision import (
        configure_training_device,
        select_readiness_examples,
        training_precision,
    )
except ImportError:
    configure_training_device = select_readiness_examples = training_precision = None


class TrainingPrecisionTests(unittest.TestCase):
    def torch(self, free, *, available=True, bf16=True):
        self.fractions = []
        return SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: available,
                mem_get_info=lambda device: (int(free * 1024**3), 8 * 1024**3),
                is_bf16_supported=lambda: bf16,
                set_per_process_memory_fraction=lambda fraction, device: self.fractions.append(
                    (fraction, device)
                ),
            )
        )

    def test_default_guard_remains_sixteen_gib_and_unknown_profile_fails(self):
        self.assertTrue(callable(training_precision))
        self.assertEqual(training_precision({})["minimum_free_gib"], 16)
        with self.assertRaises(RuntimeError):
            configure_training_device({}, torch_module=self.torch(7.5))
        with self.assertRaises(ValueError):
            training_precision({"precision_profile": "automatic_smaller_model"})

    def test_explicit_nf4_requires_free_memory_and_caps_allocator(self):
        self.assertTrue(callable(configure_training_device))
        config = {"precision_profile": "nf4_lora_8gb"}
        result = configure_training_device(config, torch_module=self.torch(7.5))
        self.assertEqual(result["weight_storage"], "nf4")
        self.assertEqual(self.fractions, [(0.8, 0)])
        for torch in [
            self.torch(6.9),
            self.torch(7.5, available=False),
            self.torch(7.5, bf16=False),
        ]:
            with self.assertRaises(RuntimeError):
                configure_training_device(config, torch_module=torch)

    def test_boundary_selection_is_deterministic_and_does_not_filter_on_labels(self):
        self.assertTrue(callable(select_readiness_examples))
        rows = [{"sample_id": s, "role": "train"} for s in ["long", "wide", "tied"]]
        metadata = [
            {"sample_id": "long", "processed_tokens": 2708, "image_grid_thw": [[1, 16, 36]]},
            {"sample_id": "wide", "processed_tokens": 968, "image_grid_thw": [[1, 10, 64]]},
            {"sample_id": "tied", "processed_tokens": 2708, "image_grid_thw": [[1, 10, 32]]},
        ]
        self.assertEqual(select_readiness_examples(rows, metadata), [0, 1])
        for bad_rows, bad_metadata in [
            ([{**rows[0], "role": "locked"}, *rows[1:]], metadata),
            (rows, metadata[:-1]),
            (rows, [*metadata, metadata[0]]),
        ]:
            with self.assertRaises(ValueError):
                select_readiness_examples(bad_rows, bad_metadata)


if __name__ == "__main__":
    unittest.main()
