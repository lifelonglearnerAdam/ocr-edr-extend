"""Compute the same causal CE while omitting ignored-position vocabulary logits."""

from __future__ import annotations


def projected_causal_loss(hidden_states, labels, lm_head):
    import torch
    import torch.nn.functional as functional

    if (
        hidden_states.ndim != 3
        or labels.ndim != 2
        or tuple(labels.shape) != tuple(hidden_states.shape[:2])
        or labels.shape[1] < 2
        or labels.dtype != torch.long
    ):
        raise ValueError(
            "Causal projection requires matching batch/sequence hidden states and labels"
        )
    targets = labels[:, 1:]
    selected = targets != -100
    if not selected.any():
        raise ValueError("No supervised next-token targets")
    # All context hidden states are computed upstream; only zero-loss vocabulary
    # projections are omitted. Flattening preserves the original mean denominator.
    logits = lm_head(hidden_states[:, :-1][selected]).float()
    return functional.cross_entropy(logits, targets[selected], reduction="mean")


def qwen_lora_supervised_loss(model, inputs, labels):
    if (
        not hasattr(model, "get_base_model")
        or len(model.peft_config) != 1
        or any(str(cfg.peft_type) != "PeftType.LORA" for cfg in model.peft_config.values())
        or any(cfg.is_prompt_learning for cfg in model.peft_config.values())
    ):
        raise ValueError("Projected loss supports one ordinary language LoRA adapter")
    base = model.get_base_model()
    if base.config.model_type != "qwen2_vl" or not hasattr(base, "lm_head"):
        raise ValueError("Projected forward is specific to Qwen2-VL")
    if set(inputs) - {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}:
        raise ValueError("Unexpected projected training inputs")
    # Ordinary LoRA matrices are injected in the transformer itself. The wrapper
    # context preserves PEFT forward hooks; prompt/mixed-adapter modes are rejected.
    with model._enable_peft_forward_hooks(**inputs):
        output = base.model(**inputs, use_cache=False, return_dict=True)
        return projected_causal_loss(output.last_hidden_state, labels, base.lm_head)
