# Explicit NF4 readiness implementation plan

> **For agentic workers:** Execute inline with the existing test/review workflow; preserve the BF16 default and historical source snapshots.

**Goal:** Run the two frozen largest-sequence/grid training checks on available local hardware without weakening the original BF16 protocol.

**Architecture:** An explicit precision helper controls device requirements and base-model loading; the existing optimizer/data mask loop remains shared. A separate readiness CLI selects only the two predeclared train records, records runtime evidence and verifies a saved adapter roundtrip.

**Tech Stack:** Existing PyTorch/Transformers plus isolated PEFT0.17.1 and bitsandbytes0.48.1.

**Spec:** `docs/research/TABLE_NF4_READINESS_PROTOCOL_20261007.md`.

## Global constraints

7 GiB minimum free for explicit NF4, 80% allocator fraction, 16 GiB unchanged default BF16, no CPU/offload, no dev/calibration/locked gradients, no altered image bounds or truncation, exact predeclared two-example trial, all failures retained.

## Task 1: Explicit precision dispatch

Files: create `src/ocr_edr/training_precision.py`, test `tests/test_training_precision.py`; minimally modify `src/ocr_edr/sft_training.py` and dependency receipt in `scripts/train_table_sft.py`.

- [x] Write failing tests that default BF16 rejects 8 GiB, explicit NF4 enforces its free-memory/allocator constraints, unavailable CUDA and unknown profile fail, and boundary-sample selection is deterministic and excludes non-train/unknown examples.
- [x] Implement `training_precision(config)`, `configure_training_device(config, torch_module=...)`, `load_training_base(model_path, config, receipt)`, and `select_readiness_examples(rows, preflight)` with strict named profiles and no automatic fallback.
- [x] Keep original masks, schedule, optimizer and LoRA constraints in `train_fixed_schedule`; use the helper only for device setup/base loading. Record quantized modules and nonquantized dtype preparation.
- [x] Verify focused tests and optional dependency behavior before hardware execution.

## Task 2: Actual bounded GPU run

Files: `scripts/check_table_nf4_readiness.py`, `configs/train/table_nf4_readiness_20261007.yaml` and private run directory.

- [x] Verify pinned model/admission/selection hashes; require no other compute PID before startup, check current versions and GPU; reject reused output directories.
- [x] Execute two sequential optimizer steps on p0055-extra_row then p0079-cell_perturbation. Keep 4096 token cap and 100352–200704 pixel bounds. Compare regenerated preflight records with their frozen source metadata.
- [x] Release GPU objects and reload the saved adapter onto a freshly loaded quantized base; compare every saved adapter tensor with its loaded state.
- [x] Report failure or measured peak/time/gradient/adapter invariants. Only a successful run permits preparing the independent full NF4 arm comparison; it is not a claim of accuracy improvement.

## Task 3: Review and integration

- [x] Run checks and read-only review; preserve exact executed sources, original failed receipts and dependency hashes. Full-environment 140 tests pass; independent checkpoint reload verifies all 112 tensors.
Publication target: existing draft PR #4, without merging; git/PR readback records completion.
- [x] Keep the full five-direction research goal active; proceed to actual full training if this hardware check succeeds.

The first NF4 trial OOMed before step 1. The second completed two optimizer steps but retained failed status after same-process reload preflight. A separate process verified the unchanged checkpoint. Full NF4 all-arm training and the failure-stopping continuation pipeline are live; full two-arm quality evaluation is not yet complete.
