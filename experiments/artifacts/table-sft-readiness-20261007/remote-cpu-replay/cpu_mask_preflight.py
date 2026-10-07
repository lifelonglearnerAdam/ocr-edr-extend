"""Replay the recorded CPU mask/length check in an independently staged runtime."""
import argparse
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser()
for name in ["project", "dataset", "model", "output"]:
    parser.add_argument("--" + name, type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.project / "src"))
from ocr_edr.sft import assistant_labels, sha256
from ocr_edr.table_training import document_balanced_schedule, load_table_supervision
import yaml

cfg_path = args.project / "configs/train/sft_table_screen.yaml"
cfg = yaml.safe_load(cfg_path.read_text())
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
receipt = {
    "status": "initializing", "started_at": datetime.now(timezone.utc).isoformat(),
    "config_sha256": sha256(cfg_path), "driver_sha256": sha256(Path(__file__)),
    "optimizer_steps_completed": 0, "calibration_locked_targets_loaded": False,
}
(output / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
try:
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    torch.set_num_threads(4)
    receipt["versions"] = {n: importlib.metadata.version(n) for n in ["torch", "transformers", "Pillow", "lxml"]}
    train = load_table_supervision(args.dataset / "admitted-supervision", args.dataset, role="train")
    dev = load_table_supervision(args.dataset / "admitted-supervision", args.dataset, role="model_dev")
    assert len(train) == 406 and len(dev) == 103
    for name in ["document_id", "source_sha256", "family_id"]:
        assert not {r[name] for r in train} & {r[name] for r in dev}
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True, use_fast=False,
        min_pixels=cfg["min_pixels"], max_pixels=cfg["max_pixels"])
    expected_path = args.project / "experiments/artifacts/table-sft-readiness-20261007/token-preflight.jsonl"
    expected = {r["sample_id"]: r for r in map(json.loads, expected_path.read_text().splitlines())}
    rows = []
    for row in train:
        user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row["prompt"]}]}
        prefix_text = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
        complete_text = processor.apply_chat_template(
            [user, {"role": "assistant", "content": [{"type": "text", "text": row["target"]}]}],
            tokenize=False, add_generation_prompt=False)
        with Image.open(row["resolved_source_image"]) as source:
            image = source.convert("RGB")
        prefix = processor(text=[prefix_text], images=[image], return_tensors="pt")
        full = processor(text=[complete_text], images=[image], return_tensors="pt")
        labels = assistant_labels(prefix.input_ids[0].tolist(), full.input_ids[0].tolist(), full.attention_mask[0].tolist())
        if len(labels) > cfg["max_sequence_tokens"]:
            raise ValueError("Training sequence exceeds frozen cap")
        observed = {"sample_id": row["sample_id"], "family_id": row["family_id"], "document_id": row["document_id"],
                    "variant": row["variant"], "processed_tokens": len(labels),
                    "supervised_tokens": sum(x != -100 for x in labels),
                    "masked_prefix_tokens": len(prefix.input_ids[0]), "image_grid_thw": full.image_grid_thw.tolist()}
        if observed != expected[row["sample_id"]]:
            raise ValueError("Cross-runtime preflight mismatch: " + row["sample_id"])
        rows.append(observed)
        if len(rows) % 64 == 0:
            print("verified", len(rows), "/", len(train), flush=True)
    (output / "token-preflight.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    metadata = {r["sample_id"]: r for r in rows}
    arms = {}
    for arm in cfg["arms"]:
        schedule = document_balanced_schedule(train, arm=arm, passes_per_document=cfg["passes_per_document"], seed=cfg["seed"])
        assert len(schedule) == cfg["steps"] * cfg["gradient_accumulation"]
        arms[arm] = {"exposures": len(schedule), "planned_optimizer_steps": cfg["steps"],
                     "processed_tokens": sum(metadata[train[i]["sample_id"]]["processed_tokens"] for i in schedule),
                     "supervised_tokens": sum(metadata[train[i]["sample_id"]]["supervised_tokens"] for i in schedule)}
    receipt.update(status="passed", examples=len(rows), model_dev_identity_checks=len(dev),
                   every_row_matches_local_preflight=True, arms=arms,
                   token_preflight_sha256=sha256(output / "token-preflight.jsonl"),
                   expected_preflight_sha256=sha256(expected_path))
except Exception as error:
    receipt.update(status="failed", error_type=type(error).__name__, error=str(error)[:500])
    raise
finally:
    receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
    (output / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2), flush=True)
