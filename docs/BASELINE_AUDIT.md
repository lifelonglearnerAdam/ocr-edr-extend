# Note baseline audit

This audit reads the existing MonkeyOCRv2-B Note evaluation artifacts. It does not run a repair model. The machine-readable report is [note_baseline_audit.json](../experiments/tables/note_baseline_audit.json); its input hashes identify the exact snapshot.

There are 118 pages, 25 diagnostic formula elements, and 37 diagnostic table elements. The official evaluator emits 28 formula matches and 37 table matches. The importer retains both representations rather than equating their counts. Joining page identity and GT reading-order position with the exact reference yields 23 formula and 37 table inputs with a clear one-to-one prediction. Two formula elements remain unresolved and receive no guessed prediction.

| Metric | Official page average | Mean of matched samples |
|---|---:|---:|
| Formula CDM | 65.05273810% | 68.69642857% |
| Table TEDS | 73.12665319% | 72.09616353% |

The official page averages match the stored baseline summary. A mean across all matched samples weights pages with more elements more heavily and is a separate statistic.

As an accounting control, the comparison uses the baseline as both before and after: 28/28 formula and 37/37 table predictions remain unchanged; both score deltas are zero. At the default per-match metric=1 proxy threshold, there are 5 Good formula matches and 5 Good table matches. Preserve proxy is 1 and Bad-fix proxy is 0 for this identity comparison. These are bookkeeping checks, not evidence of model repair performance. VisFix is unmeasured.

To reproduce locally from the recovered mirror:

```bash
python scripts/prepare_note_eval.py \
  --source-root ../../server_mirror \
  --output data/raw/note_eval \
  --public-summary experiments/tables/note_baseline_audit.json
```

Generated element records, official matches and image hashes are evaluation-only artifacts under the ignored `data/raw/` directory. The public summary contains aggregate counts, scores and input file hashes, with no source images or reference text.
