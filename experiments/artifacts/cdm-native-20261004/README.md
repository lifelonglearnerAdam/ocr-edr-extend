# Official core CDM replay — October 4, 2026

A full offline replay of existing formula outputs reveals one useful mg-unit correction hidden by the exact-raster proxy, but no aggregate improvement for the untrained repair/recognition methods. Every saved output remains included. This package contains 320 records across 20 arms on 16 inspected development sources, not 320 independent examples.

The [protocol](../../../docs/research/CDM_PROTOCOL.md) and evaluator were committed at `a4218fc08b186da46dfc2a907bb1e26f692eefa4` before scoring. See the [results and decisions](../../../docs/research/RESULTS_20261004.md). The unmodified official OmniDocBench `cdm_metrics` core callable is pinned to `f133a71e9e91c3621c7ce8994200a7b394a06eb3`; every one of its 19 Python source files is Git-blob verified before isolated import. This is fixed-pair core metric evaluation, not the full benchmark preprocessing/matching pipeline. No new model call occurred.

| Condition | Mean CDM F1 / 16 | Full fixes / 4 initial nonmatches | Regressions / 12 initial matches | Improved / degraded |
| --- | ---: | ---: | ---: | ---: |
| Unchanged Nougat-LaTeX native | 0.9695625 | 0 | 0 | 0 / 0 |
| Original source-only repair | 0.8628125 | 0 | 2 | 0 / 2 |
| Original source-first repair | 0.8569375 | 0 | 3 | 0 / 3 |
| Original source-last repair | 0.9160000 | 0 | 1 | 0 / 1 |
| Image-only low / high upper bound | 0.8364375 | 1 | 3 | 1 / 4 |
| Post-hoc source-only environment normalization | 0.8843750 | 1 | 2 | 2 / 2 |

All 20 summaries, not just those listed above, remain in the corresponding evaluations. Case and source means coincide because each source has one native input. Both recognition bounds had identical actual grids and output strings; this is not a resolution experiment. Adapter and agreement policies are saved post-hoc development controls, not selected held-out improvements.

The native mg case (`n009`) scores 0.778; image-only recognition and saved normalized source-only repair score 1.0 with the visible correct unit. Raw repair arms had retained the wrong native unit under the original renderer/adapter. One other normalized proposal increases reference agreement but drops an overbar visible in the source and a closing parenthesis (`n013`). Core CDM also scores the native wrong subscript placement in `n012` as 1.0 despite the visible structure mismatch. The case audit preserves these cautions separately from frozen references. CDM=1 and successful rendering are not literal image-fidelity certificates.

## Contents and integrity

- `pre_evaluation.json`: clean committed evaluator revision, exact source/data hashes, planned 320 records and fresh verification receipts.
- `data/`: shared frozen native references, source mapping and historical selection receipt. `original_dataset.json` preserves the earlier dataset/proxy metadata unchanged; the new metric is described in each evaluation receipt.
- `native/`: all 64 original native repair output records, 4 summaries, 16 selfchecks and 43 unique metric calls.
- `adapter/`: all 128 saved adapter output records, 8 summaries, 16 selfchecks and 52 unique metric calls.
- `agreement/`: all 48 saved agreement output records, 3 summaries, 16 selfchecks and 37 unique metric calls.
- `recognition/`: all 80 saved image-only output records, 5 summaries, 16 selfchecks and 45 unique metric calls.
- Each condition includes `predictions.jsonl`, `run.json`, `evaluation.json`, `summary.csv` and `metric_calls.jsonl`. Identical exact pairs are cached only within that evaluation; NumPy seed 20261004 resets for every actual call.
- `original_annotation_audit.jsonl`, `recognition_processor_comparison.json` and `post_evaluation_audit.json`: original annotation caveats, unchanged recognition grids and the new visual/structural observations with private panel hashes.
- `evaluation_source/`: exact runner/loader/evaluator bytes, matching every original receipt.
- `official/`: pinned upstream integrity/revision/license receipts. The source subtree is fetched separately rather than modified or bundled.
- `runtime/`: exact package archive versions/hashes, supplied font-map hashes and both failed and successful smoke receipts. Runtime binaries/archives are omitted.
- `reproduction.json`: independent fresh-source replay exactly matches all 320 case records, 20 summaries, 64 reference selfchecks and all listed scoring/source/version fields. Newly measured metric times and temporary paths are excluded from that exact comparison.
- `checksums.sha256`: every other package file.

All four pre-evaluation controls pass: identical and equivalent bracing score 1, wrong exponent 0.75, malformed prediction 0 with a rendered reference. Every reference self-scores 1 with nonzero tokens; subsequent reference counts remain stable. No actual initial/final prediction has zero CDM tokens, including strings that the earlier Tectonic adapter rejected or that merely contain coordinates. Failed or empty predictions would remain zero-valued and in the denominator; reference/runtime failure instead aborts evaluation with a failed receipt.

## Reproduction without model calls or images

Use Python 3.11+ (recorded environment: 3.12) and `requirements-cdm.txt`, with pdfLaTeX, ImageMagick 7 including PDF support and Ghostscript available. Recorded versions are TeX Live 2025/Debian pdfTeX 1.40.28, ImageMagick 7.1.2-18 Q16, Ghostscript 10.06.0, NumPy 2.3.3, SciPy 1.16.2, Pillow 11.3.0 and pylatexenc 2.10. Exact archive and font-map hashes are in the receipts. No CJK-font completeness is established by these non-CJK development formulas.

```bash
python -m pip install -r requirements-cdm.txt
python scripts/fetch_official_cdm_sources.py \
  --output data/raw/omnidocbench-cdm-replay \
  --revision f133a71e9e91c3621c7ce8994200a7b394a06eb3
for condition in native adapter agreement recognition; do
  python scripts/evaluate_cdm.py \
    --predictions experiments/artifacts/cdm-native-20261004/$condition/predictions.jsonl \
    --references experiments/artifacts/cdm-native-20261004/data/references.jsonl \
    --official-root data/raw/omnidocbench-cdm-replay \
    --tmp-dir /tmp/ocr-edr-cdm-replay/$condition \
    --output experiments/runs/cdm-replay/$condition
done
```

If using an isolated extracted runtime, add `--runtime-env /path/to/your/env.json` containing its PATH/library/TeX/font-map configuration. Our distribution packages were only downloaded and extracted, not installed globally. The original external-volume Ghostscript denial disappeared when actual generated PDF scratch files moved to a permitted directory; the system profile and upstream code were unchanged. Keep failed readiness attempts as diagnostic evidence rather than formula scores. Different runtime/font versions need their own readiness checks and disclosed reproduction comparison.

Compare `cases`, `summary`, `reference_readiness` and metric/source hashes with the saved evaluations; fresh metric seconds and temporary paths can differ. Reference/candidate strings are passed exactly as frozen; only the upstream algorithm's internal preprocessing runs. Source images, private panels, model weights, runtime binaries, archives and credentials are excluded. All 16 sources remain development and must be excluded from future untouched evaluation claims.
