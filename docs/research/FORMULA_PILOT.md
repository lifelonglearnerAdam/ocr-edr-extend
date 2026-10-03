# Controlled formula diagnostic protocol

Date: 2026-10-02. This protocol was fixed before the batch run. The smoke case only checks model loading and execution; it is excluded from batch evidence.

## Purpose and scope

Determine whether the untrained Qwen2-VL-2B student can propose image-faithful formula repairs, whether a candidate rendering helps, and whether second-turn edits damage correct predictions. This harness generates candidates without a learned acceptance verifier. It does not claim to implement or evaluate the full RepairLoop model policy, DocEDR, GRPO, tables, CDM, or VisFix.

The renderer is Matplotlib MathText on a restricted LaTeX subset. Source and candidate images share the same font and rasterizer, which makes the diagnostic easier and biases its exact-raster proxy. Full TeX and independently sourced document images are required before real benchmark evaluation.

## Fixed cases

Four hand-authored formula templates cover exponents, fraction operators, sum limits and integral limits. Each supplies two source directions: the base expression and a deliberately altered, possibly mathematically false expression. The OCR prediction must follow the source even when it conflicts with a familiar formula. Each source has a correct and an incorrect prediction; the four base sources also have whitespace-only rendering-equivalent controls.

Total: 20 cases, 8 source images, 4 original templates; 8 incorrect and 12 correct/equivalent inputs. These related cases are not 20 independent observations. The counterfactuals detect prior-driven rewriting, not generalization.

`inputs.jsonl` contains only sample IDs, generic source family IDs, source-image paths/hashes and initial predictions. `references.jsonl` contains offline references and branch labels and is never loaded by inference. Inputs are rejected if they contain any additional fields. No reference normalization or correctness score is passed to the model.

## Comparisons

| Condition | Evidence and calls |
| --- | --- |
| Unchanged input | Zero calls; required net-gain baseline |
| `source_only_1` | Source image + initial prediction; one call |
| `with_render_1` | Source image + initial prediction + its render; one call |
| `fresh_render_2` | Same rendered first turn, followed by current prediction and its updated render; two calls |
| `stale_render_2` | Same rendered first turn, followed by current prediction but the initial render; two calls |

The two iteration conditions share the exact first proposal, so the second call isolates fresh versus stale evidence when the proposal changes. When the first proposal is unchanged, those conditions have identical inputs; count these separately. The prompt asks for source fidelity, copying correct inputs, and no solving or simplification. Greedy decoding, a pinned model snapshot, a fixed 128-token cap, fixed image limits and no conversation history keep this probe bounded.

Rendering failure rolls back that individual proposal to its input. Rendering success is only a syntax check; it is not correctness acceptance. Report raw proposals, extraction behavior, renderer errors, final outputs and per-call tokens/latency. Syntax rollback can itself mask harmful proposals, so count it explicitly.

## Offline measurements

Report reference-normalized string exactness and same-renderer exact raster equality as separate diagnostic proxies. Neither is CDM, VisFix, a semantic equivalence guarantee, or a general LaTeX metric. Report repairs among initially incorrect inputs, regressions among initially correct/equivalent inputs, counterfactual-source errors, full-set net gains, syntax failures and cost.

A positive result only warrants a real-data pilot. Zero gain or regressions warrant failure analysis and controlled editing/verifier work before heavier training. No significance or confidence interval is justified by four hand-selected templates.

## Reproduction

Install `requirements-pilot.txt` into an isolated Python 3.12 environment; the pinned Torch/Torchvision wheels use the CUDA 12.8 wheel index. CPU fallback is supported. Download the `Qwen/Qwen2-VL-2B-Instruct` snapshot `895c3a49bc3fa70a340399125c650a463535e71c` into a private ignored cache.

**Platform note.** The archived 2026-10-02 batch and its `.venv-pilot` snapshot were produced on the Ubuntu/POSIX research host. On Windows, create a native venv (`py -3.12 -m venv .venv-pilot`) and never mount or reuse a Linux `bin/` venv; see `docs/SERVER_RUNBOOK.md`. Offline score replay (`evaluate_formula_pilot.py`) is OS-independent when artifact hashes match.

```bash
.venv-pilot/bin/python scripts/make_formula_pilot.py \
  --output experiments/runs/formula-pilot-20261002/data

.venv-pilot/bin/python scripts/run_formula_pilot.py \
  --inputs experiments/runs/formula-pilot-20261002/data/inputs.jsonl \
  --model-path data/raw/model-cache/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c \
  --output experiments/runs/formula-pilot-20261002/batch-cpu \
  --device cpu --max-new-tokens 128

.venv-pilot/bin/python scripts/evaluate_formula_pilot.py \
  --run experiments/runs/formula-pilot-20261002/batch-cpu \
  --references experiments/runs/formula-pilot-20261002/data/references.jsonl
```

Use a new output directory for each run. A run records exact dependency versions, source-file hashes, the Git base/dirty flag, input hash, prompts, raw responses, token counts and timing. Correctness references only enter the separate evaluator. The batch outputs and raw model cache are ignored by Git; a compact, reviewable result report will be published after evaluation.
