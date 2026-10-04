# Table development evidence, October 4, 2026

Four fixed source pages from the official OmniDocBench demo produce 14 development inputs and 56 Qwen2-VL-2B calls. The same cases appear in all four proposal methods plus an unchanged zero-call baseline. The native-looking predictions come from the released demo records; their original parser is not identified. These are inspected development examples, not a held-out benchmark or a named-parser transfer result.

Source-only rewrite fixes two digit perturbations and regresses one reference-matching control; the two-image rewrites each fix one digit perturbation, with many invalid-HTML rollbacks. No full rewrite improves a published demo prediction; no method repairs either span perturbation. All TEDS-S values remain unchanged. Valid JSON edits change three cells, each at a wrong address; one damages a merged label without changing TEDS. Source/reference disagreements on the financial and rating tables are recorded separately, without changing frozen labels or scores. See the [full findings](../../../docs/research/RESULTS_20261004.md).

- `data/`: exact reference-free inputs, offline references, all ten eligibility decisions, crop/page hashes, polygon bounds and the reversed-dimension exceptions.
- `repair/`: completed original inference receipt and all raw responses, prompts, ordered image hashes, processor grids, token counts, failures, actions and generation time.
- `evaluation/`: all paired case/variant/page scores and CSV summaries. These apply the pinned official normalization symmetrically.
- `official/`: official source-integrity receipt and the ten-record TEDS/TEDS-S replay receipt. The fetch script retrieves upstream code/license/demo metadata at an exact Git revision; no benchmark images or weights are needed for evaluation replay.
- `inference_source/`: exact bytes for all three files hashed by the inference receipt, preserved before later formatting. This run used a dirty checkout; the code hashes identify the actual proposer/driver/renderer.
- `preflight.json`, `annotation_audit.jsonl`, `post_inference_audit.jsonl`: original private render paths, crop/reference observations and inspection of every final case. Private contact sheets are omitted; their hashes are recorded for the original workspace.
- `reproduction.json`: comparison of all cases, summary records and per-page aggregates against a fresh offline replay.

No source/page/crop images, generated panels, model weights or raw data archives are included. All accessed pages must be excluded from future frozen test claims. TEDS agreement is agreement with the released reference, which can be flawed; it is not a visual correctness certificate. Rendering success and valid actions likewise do not certify an edit. Formula/CDM dependencies are not imported for this isolated table evaluator.

Replay with a clean output directory:

```bash
# Install these into an environment with the core dependencies.
python -m pip install -r requirements-table-pilot.txt
python scripts/fetch_official_table_sources.py \
  --output data/raw/omnidocbench-table-replay \
  --revision f133a71e9e91c3621c7ce8994200a7b394a06eb3
python scripts/evaluate_table_pilot.py \
  --run experiments/artifacts/table-development-20261004/repair \
  --inputs experiments/artifacts/table-development-20261004/data/inputs.jsonl \
  --references experiments/artifacts/table-development-20261004/data/references.jsonl \
  --official-root data/raw/omnidocbench-table-replay \
  --output experiments/runs/table-replay
```

No model inference, GPU, renderer or benchmark image is needed for the score replay. Compare the `cases`, `summary` and `per_page` records with the saved evaluation. Source paths and output hashes are provenance rather than the metric comparison target. The original inference versions remain in `repair/run.json`; timing is copied from those calls and excludes rendering/loading. To reproduce model calls separately, reconstruct the exact crops from the pinned official pages and require their recorded hashes, then use the saved prompts/adapter and model snapshot. Consult the [frozen protocol](../../../docs/research/TABLE_DEVELOPMENT_PROTOCOL.md).
