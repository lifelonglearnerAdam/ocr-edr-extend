"""Explicit fixed-checkpoint prompt ablation; training config is never rewritten."""

from __future__ import annotations

import re


def validate_prompt_protocol(
    config, *, config_sha256, prompt_format, condition, protocol, checkpoint_hashes
):
    nf4 = config.get("precision_profile", "bf16_lora") == "nf4_lora_8gb"
    if nf4 and config["inference"]["logits_projection"] != "last_position_only":
        raise ValueError("Frozen NF4 vocabulary projection differs")
    if protocol is None:
        if nf4 and prompt_format != config["inference"]["prompt_format"]:
            raise ValueError("Frozen NF4 prompt differs without an explicit ablation protocol")
        return None
    fields = {
        "study",
        "training_config_sha256",
        "prompt_formats",
        "conditions",
        "cases",
        "documents",
        "max_new_tokens",
        "do_sample",
        "post_hoc_exploratory",
        "adapter_model_sha256",
    }
    if (
        not nf4
        or set(protocol) != fields
        or not isinstance(protocol["study"], str)
        or not protocol["study"].strip()
        or protocol["training_config_sha256"] != config_sha256
        or protocol["prompt_formats"] != ["literal_examples", "descriptive_schema"]
        or protocol["conditions"] != ["base", "all", "no_explicit_preservation"]
        or condition not in protocol["conditions"]
        or prompt_format not in protocol["prompt_formats"]
        or protocol["cases"] != config["data"]["model_dev_records"]
        or protocol["documents"] != config["data"]["model_dev_documents"]
        or protocol["max_new_tokens"] != config["inference"]["max_new_tokens"]
        or protocol["do_sample"] is not False
        or config["inference"]["do_sample"] is not False
        or protocol["post_hoc_exploratory"] is not True
    ):
        raise ValueError("Prompt-ablation schema, training hash, coverage or budget differs")
    pinned = protocol["adapter_model_sha256"]
    if (
        not isinstance(pinned, dict)
        or set(pinned) != {"all", "no_explicit_preservation"}
        or any(
            not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value)
            for value in pinned.values()
        )
    ):
        raise ValueError("Both frozen adapter weights must be explicitly pinned")
    if condition == "base":
        if checkpoint_hashes:
            raise ValueError("Base prompt control cannot carry an adapter")
    elif checkpoint_hashes.get("adapter_model.safetensors") != pinned[condition]:
        raise ValueError("Prompt experiment cannot replace the frozen trained checkpoint")
    return dict(protocol)
