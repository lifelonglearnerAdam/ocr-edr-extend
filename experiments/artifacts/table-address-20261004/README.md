# Table cell-address development evidence, October 4, 2026

The frozen all-cell-map intervention preserves all 14 initial strings but produces zero repairs: 13 stops and one rejected multiple-action response. The original prompt produces three wrong-cell edits, five stops and six schema failures. Its fresh outputs/actions/finals reproduce all 14 earlier calls exactly. This is a negative development diagnostic on four already inspected pages, not a held-out accuracy gain or a novelty claim. See the [findings](../../../docs/research/RESULTS_20261004.md) and [pre-inference protocol](../../../docs/research/TABLE_ADDRESS_PROTOCOL.md).

- `pre_inference.json` and `call_manifest.jsonl`: clean committed source revision, fixed conditions/budget and all 28 prompt/message/image/initial hashes before inference.
- `data/`: exact original reference-free inputs, offline references, source mapping, selection and image hashes. Images are omitted.
- `repair/`: completed receipt and all 42 outputs, 28 raw calls, processor grids, prompts, costs and strict adapter outcomes.
- `evaluation/`: all case/variant/page TEDS and TEDS-S scores from hash-verified upstream code.
- `integrity.json`: every call matches the frozen manifest and every accepted action replays; all 14 original-prompt outputs reproduce prior calls exactly, excluding freshly measured timing.
- `inference_source/`: exact bytes for the three inference files recorded in the run receipt.
- `original_annotation_audit.jsonl`, `original_preflight.json` and `post_inference_audit.jsonl`: preserved source/reference caveats, byte-preservation checks, and visual reinspection of the three accepted original-prompt edits. Private contact sheets are excluded; their hashes are retained.
- `official/`: pinned upstream integrity/license/demo receipts. `reproduction.json` verifies a fresh offline replay of every case, summary and page record using a separately acquired official source root.

The indexed condition costs 16,972 input tokens and 101.24 generation seconds versus 8,893 and 74.60 for the fresh original condition. Loading/rendering/scoring are excluded. TEDS-S is unchanged in every case. Repeated inputs from four pages remain development evidence. The source/reference disagreements and unknown published-demo parser are retained, and the action set still cannot insert missing rows.

No benchmark page/crop image, private panel, weight, raw archive, cache or credentials are included. Score replay requires neither model calls nor images:

```bash
python -m pip install -r requirements-table-pilot.txt
python scripts/fetch_official_table_sources.py \
  --output data/raw/omnidocbench-address-replay \
  --revision f133a71e9e91c3621c7ce8994200a7b394a06eb3
python scripts/evaluate_table_pilot.py \
  --run experiments/artifacts/table-address-20261004/repair \
  --inputs experiments/artifacts/table-address-20261004/data/inputs.jsonl \
  --references experiments/artifacts/table-address-20261004/data/references.jsonl \
  --official-root data/raw/omnidocbench-address-replay \
  --output experiments/runs/table-address-replay
```

Compare `cases`, `summary` and `per_page` with the saved evaluation. Inference reproduction additionally needs the exact source crops and pinned model, fonts and processor versions; require their recorded hashes. Use the saved original/model-map prompts and the archived action adapter rather than adapting it after inspection.
