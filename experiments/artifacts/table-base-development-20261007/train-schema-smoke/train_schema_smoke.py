"""Small train-only paired prompt-format diagnostic; not a quality evaluation."""
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo / "src"))
from ocr_edr.formula_pilot import validate_pilot_inputs
from ocr_edr.qwen import QwenFormulaProposer
from ocr_edr.sft import sha256, verify_model_files
from ocr_edr.table_pilot import apply_table_action
from ocr_edr.table_sft_screen import _unique_object, table_messages

examples = (
    '\nExact response-shape examples (indices and replacement text are placeholders; '
    'choose your own from the current HTML and source image, and emit exactly one object):\n'
    '{"action":"stop"}\n'
    '{"action":"replace_cell","row":0,"cell":0,"text":"correct plain cell text"}\n'
    '{"action":"set_span","row":0,"cell":0,"rowspan":1,"colspan":1}\n'
    '{"action":"delete_row","row":0}'
)
data = repo / "data/processed/pubtabnet-four-roles-20261007"
out = Path(__file__).parent / "train-schema-smoke"
out.mkdir(exist_ok=False)
dataset = json.loads((data / "dataset.json").read_text())
input_path = data / "train-inputs.jsonl"
assert sha256(input_path) == dataset["file_sha256"][input_path.name]
cases = [json.loads(line) for line in input_path.read_text().splitlines()]
cases = [row for row in cases if row["family_id"] == "p0001"]
assert len(cases) == 3
validate_pilot_inputs(cases, data)
revision = "895c3a49bc3fa70a340399125c650a463535e71c"
model = repo / "data/raw/model-cache/models--Qwen--Qwen2-VL-2B-Instruct/snapshots" / revision
integrity = repo / "experiments/runs/server-selection-20261006/model-receipt.json"
verify_model_files(model, json.loads(integrity.read_text()))
receipt = {
    "status": "initializing", "started_at": datetime.now(timezone.utc).isoformat(),
    "purpose": "train-only prompt-format feasibility diagnostic; no quality/repair-gain estimate",
    "selection": "first admitted train family p0001; all three of its existing variants",
    "samples": [r["sample_id"] for r in cases], "independent_source_documents": 1,
    "reference_access": "none", "calibration_locked_access": False,
    "conditions": ["descriptive_schema", "literal_examples"],
    "literal_suffix": examples, "max_new_tokens": 192,
    "min_pixels": 100352, "max_pixels": 200704, "seed": 20261007,
    "device": "cpu", "dtype": "bfloat16", "cpu_threads": 4,
    "model_receipt_sha256": sha256(integrity), "model_revision": revision,
    "train_input_sha256": sha256(input_path), "driver_sha256": sha256(Path(__file__)),
    "completed_calls": 0, "optimizer_steps": 0,
}
(out / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
rows = []
try:
    import torch

    torch.manual_seed(20261007)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    receipt["versions"] = {n: importlib.metadata.version(n) for n in ["torch", "transformers", "Pillow"]}
    proposer = QwenFormulaProposer(model, device="cpu", max_new_tokens=192, min_pixels=100352, max_pixels=200704)
    with (out / "calls.jsonl").open("w") as stream:
        for case in cases:
            for condition in receipt["conditions"]:
                messages, prompt = table_messages(case["prediction"])
                if condition == "literal_examples":
                    prompt += examples
                    messages[0]["content"][-1]["text"] = prompt
                call = proposer.generate([data / case["source_image"]], messages, prompt)
                action, failure = None, None
                try:
                    if call["output_tokens"] >= 192:
                        raise ValueError("token_cap")
                    value = json.loads(call["raw_output"], object_pairs_hook=_unique_object)
                    _, action = apply_table_action(case["prediction"], json.dumps(value))
                except (ValueError, TypeError) as error:
                    failure = f"{type(error).__name__}: {str(error)[:200]}"
                row = {**case, **call, "condition": condition, "executable_action": action, "contract_failure": failure}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                rows.append(row)
                print(case["sample_id"], condition, "valid", action is not None, flush=True)
    receipt.update(status="completed", calls_sha256=sha256(out / "calls.jsonl"), summary={
        condition: {"calls": sum(r["condition"] == condition for r in rows),
                    "executable_actions": sum(r["condition"] == condition and r["executable_action"] is not None for r in rows),
                    "input_tokens": sum(r["input_tokens"] for r in rows if r["condition"] == condition),
                    "output_tokens": sum(r["output_tokens"] for r in rows if r["condition"] == condition),
                    "generation_seconds": sum(r["generation_seconds"] for r in rows if r["condition"] == condition)}
        for condition in receipt["conditions"]})
except Exception as error:
    receipt.update(status="failed", error_type=type(error).__name__, error=str(error)[:500])
    raise
finally:
    receipt.update(completed_calls=len(rows), finished_at=datetime.now(timezone.utc).isoformat())
    (out / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt["summary"], indent=2), flush=True)
