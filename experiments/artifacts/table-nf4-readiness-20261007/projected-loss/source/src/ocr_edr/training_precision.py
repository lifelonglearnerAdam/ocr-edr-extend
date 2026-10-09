"""Explicit training precision profiles; never silently lower the BF16 GPU guard."""

from __future__ import annotations

import math
from collections import Counter


def training_precision(config):
    name = config.get("precision_profile", "bf16_lora")
    if name == "bf16_lora":
        return {
            "name": name,
            "weight_storage": "bfloat16",
            "compute_dtype": "bfloat16",
            "minimum_free_gib": 16,
            "allocator_fraction": 1.0,
        }
    if name == "nf4_lora_8gb":
        return {
            "name": name,
            "weight_storage": "nf4",
            "compute_dtype": "bfloat16",
            "double_quant": True,
            "nonquantized_parameters": "float32_after_peft_preparation",
            "minimum_free_gib": 7,
            "allocator_fraction": 0.8,
        }
    raise ValueError("Unknown explicit training precision profile")


def configure_training_device(config, *, torch_module):
    profile = training_precision(config)
    cuda = torch_module.cuda
    if not cuda.is_available():
        raise RuntimeError("Training requires an available CUDA device")
    free, total = cuda.mem_get_info(0)
    if free < profile["minimum_free_gib"] * 1024**3:
        raise RuntimeError(f"Selected profile requires {profile['minimum_free_gib']} GiB free")
    if profile["weight_storage"] == "nf4" and not cuda.is_bf16_supported():
        raise RuntimeError("This NF4 profile requires native BF16 compute support")
    # The explicit 8 GiB profile reserves allocator headroom for the local desktop.
    # CUDA/library allocations outside PyTorch are not constrained by this API.
    cuda.set_per_process_memory_fraction(profile["allocator_fraction"], 0)
    return {**profile, "initial_free_gib": free / 1024**3, "total_gib": total / 1024**3}


def load_training_base(model_path, config, receipt):
    import torch
    from transformers import Qwen2VLForConditionalGeneration

    profile = training_precision(config)
    kwargs = dict(local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa")
    if profile["weight_storage"] == "bfloat16":
        return Qwen2VLForConditionalGeneration.from_pretrained(str(model_path), **kwargs).to(
            "cuda:0"
        )

    import bitsandbytes as bnb
    from peft import prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    base = Qwen2VLForConditionalGeneration.from_pretrained(
        str(model_path),
        **kwargs,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    modules = [
        name for name, module in base.named_modules() if isinstance(module, bnb.nn.Linear4bit)
    ]
    if not getattr(base, "is_loaded_in_4bit", False) or not modules:
        raise ValueError("NF4 loading did not produce any four-bit modules")
    if any(parameter.device.type != "cuda" for parameter in base.parameters()):
        raise ValueError("CPU/meta/disk offload is outside the explicit NF4 profile")
    # Let the shared optimizer enable its existing nonreentrant checkpointing once.
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    receipt["quantization"] = {
        **profile,
        "linear4bit_modules": modules,
        "visual_linear4bit_modules": sum("visual" in name.split(".") for name in modules),
        "nonquantized_parameter_dtypes": dict(
            Counter(
                str(parameter.dtype)
                for parameter in base.parameters()
                if not isinstance(parameter, bnb.nn.Params4bit)
            )
        ),
        "offload": False,
    }
    return base


def select_readiness_examples(rows, preflight):
    by_id = {row["sample_id"]: index for index, row in enumerate(rows)}
    meta = {row["sample_id"]: row for row in preflight}
    if (
        not rows
        or len(by_id) != len(rows)
        or len(meta) != len(preflight)
        or set(by_id) != set(meta)
        or any(row["role"] != "train" for row in rows)
    ):
        raise ValueError("Readiness selection requires complete unique training-only metadata")
    for row in preflight:
        if (
            type(row["processed_tokens"]) is not int
            or row["processed_tokens"] <= 0
            or not row["image_grid_thw"]
            or any(
                len(grid) != 3 or any(type(v) is not int or v < 1 for v in grid)
                for grid in row["image_grid_thw"]
            )
        ):
            raise ValueError("Invalid token/grid metadata")
    selected = []
    for score in [
        lambda row: row["processed_tokens"],
        lambda row: sum(math.prod(grid) for grid in row["image_grid_thw"]),
    ]:
        best = min(preflight, key=lambda row: (-score(row), row["sample_id"]))
        index = by_id[best["sample_id"]]
        if index not in selected:
            selected.append(index)
    return selected
