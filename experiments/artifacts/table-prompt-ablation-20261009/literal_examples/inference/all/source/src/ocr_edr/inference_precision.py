"""Explicit same-precision NF4 inference, with last-position vocabulary projection."""

from __future__ import annotations

from .training_precision import configure_training_device, load_training_base


def enable_last_position_projection(base):
    if base.config.model_type != "qwen2_vl":
        raise ValueError("Last-position inference projection supports only Qwen2-VL")

    def last_position(module, args):
        if len(args) != 1 or args[0].ndim != 3 or args[0].shape[1] < 1:
            raise ValueError("Unexpected lm_head input for last-position generation")
        return (args[0][:, -1:, :],)

    # Qwen2-VL generation consumes only logits[:, -1, :]. Transformer inputs and
    # KV-cache are unchanged. This hook must never be attached to a training model.
    return base.lm_head.register_forward_pre_hook(last_position)


def load_nf4_inference_base(model_path):
    import torch

    config = {"precision_profile": "nf4_lora_8gb"}
    metadata = {"precision": configure_training_device(config, torch_module=torch)}
    base = load_training_base(model_path, config, metadata)
    enable_last_position_projection(base)
    metadata["logits_projection"] = "last_position_only"
    return base, metadata
