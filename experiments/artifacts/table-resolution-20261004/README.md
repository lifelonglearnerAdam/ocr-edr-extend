# Table visual-resolution development control — October 4, 2026

Fourteen previously inspected inputs from four source pages receive the unchanged original JSON repair prompt at two verified processor grids. Both conditions make three accepted changes and zero fixes; higher visual exposure introduces a matching-control regression and four output-cap failures. These are development diagnostics, not benchmark or novelty claims.

The [frozen protocol](../../../docs/research/TABLE_RESOLUTION_PROTOCOL.md) and driver were committed at `ca64cae3ea185215101ce906930d9fcee622b8cc` before model loading. The [report](../../../docs/research/RESULTS_20261004.md) retains the source/reference caveats and separate visual audit.

| Condition | Case TEDS | Page TEDS | Changes | Stops | Schema failures | Matching regressions | Fixes | Cap hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unchanged | 0.940112 | 0.937403 | 0 | — | 0 | 0 | 0 | 0 |
| Low: 100352–200704 pixels | 0.936268 | 0.932918 | 3 | 5 | 6 | 0 | 0 | 0 |
| High: 802816–802816 pixels | 0.936144 | 0.932774 | 3 | 4 | 7 | 1 | 0 | 4 |

The high setting forces interpolation and larger actual visual grids for every source family; it does not acquire new source detail. Low uses 8,893 input / 353 output tokens and 72.114 generation seconds. High uses 20,980 / 2,309 and 554.387 seconds. Generation excludes loading, rendering and scoring; fixed condition order and CPU state may affect timing. No TEDS-S score changes. Four pages are the independent units; repeated variants and duplicated unchanged rows are correlated.

## Contents

- `pre_inference.json`, `call_manifest.jsonl`: clean committed source revision and all 28 frozen prompts, messages, source/initial hashes, bounds, grids and token counts.
- `data/`: frozen reference-free inputs, offline references, source/selection mapping and hashes; benchmark images omitted.
- `low/`, `high/`: both receipts, all 56 output rows including unchanged comparators, raw outputs/actions/costs and all fixed-pair official TEDS/TEDS-S scores.
- `integrity.json`: all manifest/runtime/source/action checks; all 14 low calls reproduce preceding raw outputs/actions/final strings exactly, excluding newly measured time.
- `inference_source/`: exact bytes of all three inference files named in the receipts.
- `original_annotation_audit.jsonl`, `original_preflight.json`, `post_inference_audit.jsonl`: frozen reference disagreements and all 28 proposal audits; all three accepted high changes were visually inspected. Private panel hashes are retained; panels are omitted.
- `official/`: pinned Git-blob integrity/license/demo replay receipts.
- `reproduction.json`: fresh replay through separately downloaded official sources; both conditions exactly reproduce every case, summary and page record.
- `checksums.sha256`: all other portable files.

The high `d001` and `d003` proposals copy a website over the industry label. High `d009` damages the merged rating label but does not reduce its TEDS. High `d011`–`d014` repeat schema-example actions/placeholders until the 512-token cap; the frozen adapter rejects and rolls back. Lower schema failure alone did not establish source grounding in the preceding cell-map run; larger grids do not resolve it here. Investigate proposal capability/supervision before scaling this untrained prompt to RL. This does not falsify learned repair, SFT, or RL generally.

## Offline reproduction

No model call, source image or weights are needed for score replay:

```bash
python -m pip install -r requirements-table-pilot.txt
python scripts/fetch_official_table_sources.py \
  --output data/raw/omnidocbench-resolution-replay \
  --revision f133a71e9e91c3621c7ce8994200a7b394a06eb3
for condition in low high; do
  python scripts/evaluate_table_pilot.py \
    --run experiments/artifacts/table-resolution-20261004/$condition/repair \
    --inputs experiments/artifacts/table-resolution-20261004/data/inputs.jsonl \
    --references experiments/artifacts/table-resolution-20261004/data/references.jsonl \
    --official-root data/raw/omnidocbench-resolution-replay \
    --output experiments/runs/table-resolution-replay/$condition
done
```

Compare `cases`, `summary` and `per_page` with each saved evaluation. Inference reproduction additionally requires the exact hashed crops, model snapshot, processor, fonts and runtime from the receipt. References, correctness scores and annotation audits remain outside inference. The published-demo parser identity is unknown; the one-action schema cannot insert missing rows. No benchmark image, weight, private cache, credential, or system runtime binary is in this package.
