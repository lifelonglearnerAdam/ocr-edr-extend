# Image-role diagnostic artifacts, October 3, 2026

All 20 synthetic cases from the October 2 diagnostic were retained. Six image-role conditions produced 120 new model calls. This is a diagnostic on four formula templates and eight source-image directions, not a benchmark or an independent test set.

`data/inputs.jsonl` is reference-free. `data/references.jsonl` is offline evaluation only. `predictions.jsonl` keeps raw outputs, messages, ordered image hashes, processor grids, token counts, and measured generation time. `run.json` pins the model/code/dependencies. Its source path keys have been made portable, with the original metadata file hash preserved. `inference_source/` preserves hash-matched clean-commit inference code; `evaluation_reproduction.json` records exact case/summary score replay. `reproduction.json` records comparison against October 2: source-only and source-first each reproduced all 20 final strings exactly.

Replay without model inference:

```bash
.venv-pilot/bin/python scripts/evaluate_formula_pilot.py \
  --run experiments/artifacts/image-role-20261003 \
  --references experiments/artifacts/image-role-20261003/data/references.jsonl
```

The evaluator creates an ignored `evaluation_renders` cache. Compare evaluation and summary hashes against `checksums.sha256`. Source-first versus source-last changes both image order and the correct textual role indices. Interleaved labels and duplicate source images are additional controls; none establishes internal attention behavior. See [the protocol](../../../docs/research/IMAGE_ROLE_PROTOCOL.md) and [the results](../../../docs/research/RESULTS_20261003.md).
