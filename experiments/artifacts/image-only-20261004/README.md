# Image-only formula recognition control, October 4, 2026

All 16 sources from the earlier UniMER development diagnostic were retained. The frozen 2B model saw only a source image and a fixed transcription prompt: no native prediction, reference, error label, or score. There were 32 calls, two per source, with no retry or training. This is development evidence, not CDM or a held-out benchmark result.

Both pixel-bound settings produce 5/16 same-renderer exact-raster matches, compared with 10/16 for the unchanged native parser output. Offline joining shows zero exact-proxy fixes among six initially nonmatching cases and five matching-to-nonmatching transitions among ten initially matching cases. One recognition fails rendering; it is kept as a failed recognition without parser rollback. Whole-display-environment normalization leaves those counts unchanged. These counts do not mean there were no useful content corrections: `u009` recognizes `mg` instead of the native parser's unrelated mathematical symbols, but omits the released reference's bold styling.

`processor_comparison.json` verifies that all 16 low/high pairs have identical actual processor grids and identical output strings. Raising only the upper pixel bound did not increase these images' processed resolution. The two conditions therefore do not test the benefit of higher-resolution inputs. Timing differs descriptively, while the actual input/output token counts are equal. None of the 32 calls reaches the 128-token cap.

- `recognition/` preserves original inference metadata and raw calls.
- `data/` contains reference-free source inputs, frozen native predictions, offline references, and source/archive mappings. Its `dataset.json` is the original 16-source selection receipt from the 32-case controlled study; the source-only 16-case input hash and two conditions are pinned separately in `recognition/run.json`.
- `evaluation/` contains the offline joined predictions and proxy scores, including the separately labeled post-hoc normalization control.
- `inference_source/` preserves the exact driver and proposer bytes matching the original run's source hashes. The original inference receipt does not contain a dependency-version map; do not manufacture a historical one from the current environment.
- `annotation_audit.jsonl` carries the previously recorded source/reference concerns. Labels were not changed.

No benchmark source images or weights are included. Every inspected image remains development data and must be excluded from later test claims. The direct-recognition and repair prompts differ, so candidate absence and task framing change together; this control does not isolate anchoring.

Replay without source images or model inference, using Tectonic and `pdftoppm`:

```bash
.venv-pilot/bin/python scripts/evaluate_source_recognition_control.py \
  --recognition experiments/artifacts/image-only-20261004/recognition \
  --native-inputs experiments/artifacts/image-only-20261004/data/native_inputs.jsonl \
  --references experiments/artifacts/image-only-20261004/data/references.jsonl \
  --output experiments/runs/image-only-replay
```

Use a new output directory for each replay. `reproduction.json` records an exact replay of all 80 case records and five summary records. Compare its case/summary content with the saved evaluation; timing is copied from the fixed calls. See [protocol](../../../docs/research/SOURCE_RECOGNITION_PROTOCOL.md) and [October 4 findings](../../../docs/research/RESULTS_20261004.md).
