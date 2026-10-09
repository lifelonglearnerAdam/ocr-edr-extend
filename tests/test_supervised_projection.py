import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from ocr_edr.supervised_projection import projected_causal_loss, qwen_lora_supervised_loss
except ImportError:
    projected_causal_loss = qwen_lora_supervised_loss = None

try:
    from ocr_edr.inference_precision import enable_last_position_projection
except ImportError:
    enable_last_position_projection = None


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch runtime")
class ProjectionLossTests(unittest.TestCase):
    def test_loss_and_gradients_match_full_shifted_cross_entropy(self):
        import torch
        import torch.nn.functional as functional

        self.assertTrue(callable(projected_causal_loss))
        torch.manual_seed(7)
        labels = torch.tensor([[-100, -100, 1, 2, 3], [-100, 2, -100, 4, -100]])
        hidden = torch.randn(2, 5, 4, requires_grad=True)
        weight = torch.randn(7, 4, requires_grad=True)
        bias = torch.randn(7, requires_grad=True)
        reference = functional.cross_entropy(
            functional.linear(hidden, weight, bias)[:, :-1].reshape(-1, 7),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        expected = torch.autograd.grad(reference, [hidden, weight, bias], retain_graph=True)
        projected = projected_causal_loss(
            hidden, labels, lambda x: functional.linear(x, weight, bias)
        )
        actual = torch.autograd.grad(projected, [hidden, weight, bias])
        torch.testing.assert_close(reference, projected, rtol=1e-6, atol=1e-6)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-6)
        for invalid in [torch.full((2, 5), -100), labels[:, :-1], torch.tensor([[1]])]:
            with self.assertRaises(ValueError):
                projected_causal_loss(hidden, invalid, lambda x: functional.linear(x, weight))


@unittest.skipUnless(
    importlib.util.find_spec("peft") and importlib.util.find_spec("transformers"),
    "optional PEFT/Transformers runtime",
)
class QwenProjectionParityTests(unittest.TestCase):
    def test_real_tiny_qwen_lora_loss_and_all_adapter_gradients_agree(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration

        self.assertTrue(callable(qwen_lora_supervised_loss))
        torch.manual_seed(13)
        config = Qwen2VLConfig(
            text_config={
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "rope_scaling": {"type": "mrope", "mrope_section": [1, 1, 2]},
            },
            vision_config={
                "depth": 1,
                "embed_dim": 16,
                "hidden_size": 16,
                "num_heads": 2,
                "mlp_ratio": 2,
                "patch_size": 2,
                "temporal_patch_size": 1,
                "spatial_merge_size": 1,
                "in_channels": 3,
            },
            image_token_id=28,
            video_token_id=29,
            vision_start_token_id=30,
            vision_end_token_id=31,
            tie_word_embeddings=False,
        )
        base = Qwen2VLForConditionalGeneration(config)
        model = get_peft_model(
            base,
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
                bias="none",
            ),
        )
        model.train()
        inputs = {
            "input_ids": torch.tensor([[2, 3, 4, 5, 6, 7]]),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
        }
        labels = torch.tensor([[-100, -100, -100, 5, 6, 7]])
        expected_loss = model(**inputs, labels=labels, use_cache=False).loss
        expected_loss.backward()
        expected_gradients = {
            name: param.grad.detach().clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        model.zero_grad(set_to_none=True)
        actual_loss = qwen_lora_supervised_loss(model, inputs, labels)
        actual_loss.backward()
        torch.testing.assert_close(expected_loss, actual_loss, rtol=1e-6, atol=1e-6)
        actual_gradients = {
            name: param.grad for name, param in model.named_parameters() if param.grad is not None
        }
        self.assertEqual(set(expected_gradients), set(actual_gradients))
        self.assertTrue(expected_gradients)
        for name in expected_gradients:
            torch.testing.assert_close(
                expected_gradients[name], actual_gradients[name], rtol=1e-5, atol=1e-7
            )
        # A real visual patch exercises Qwen multimodal position IDs and insertion.
        model.zero_grad(set_to_none=True)
        vision_inputs = {
            "input_ids": torch.tensor([[2, 30, 28, 31, 6, 7]]),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
            "pixel_values": torch.randn(1, 12),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
        }
        vision_labels = torch.tensor([[-100, -100, -100, -100, 6, 7]])
        base.model.rope_deltas = None
        reference = model(**vision_inputs, labels=vision_labels, use_cache=False).loss
        reference.backward()
        reference_grads = {
            name: p.grad.detach().clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        model.zero_grad(set_to_none=True)
        base.model.rope_deltas = None
        projected = qwen_lora_supervised_loss(model, vision_inputs, vision_labels)
        projected.backward()
        torch.testing.assert_close(reference, projected, rtol=1e-6, atol=1e-6)
        for name, p in model.named_parameters():
            if name in reference_grads:
                torch.testing.assert_close(reference_grads[name], p.grad, rtol=1e-5, atol=1e-7)
        with self.assertRaises(ValueError):
            qwen_lora_supervised_loss(model, {**inputs, "adapter_names": ["default"]}, labels)
        self.assertTrue(callable(enable_last_position_projection))
        model.eval()
        base.model.rope_deltas = None
        with torch.inference_mode():
            expected_tokens = model.generate(**vision_inputs, max_new_tokens=4, do_sample=False)
        handle = enable_last_position_projection(base)
        base.model.rope_deltas = None
        with torch.inference_mode():
            actual_tokens = model.generate(**vision_inputs, max_new_tokens=4, do_sample=False)
        torch.testing.assert_close(expected_tokens, actual_tokens, rtol=0, atol=0)
        handle.remove()


if __name__ == "__main__":
    unittest.main()
