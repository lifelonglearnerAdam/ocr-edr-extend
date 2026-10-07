# Formula/table region manifests

Each JSONL line supplies sample and parent-page identities, benchmark, modality, split, provenance flag, source region and initial parser output:

```json
{"sample_id":"synthetic-formula-1","page_id":"synthetic-page-1","benchmark":"synthetic","modality":"formula","split":"test","test_only":true,"source_image":"formula.png","prediction":"x^3","reference":"x^2"}
```

Use modality `formula` or `table`, split `train`, `dev` or `test`, and an explicit boolean `test_only`. Test records require `true`; training records require `false`. Relative images resolve beneath the manifest directory or `--source-root`. The validator checks file existence, path containment and the actual image SHA-256.

The optional reference remains outside the policy/judge Observation. It belongs in offline evaluation or independent training targets. Official benchmark match artifacts are compared separately with `compare_official_results.py`.

```bash
python scripts/validate_manifest.py --manifest examples/synthetic/regions.jsonl
python scripts/validate_manifest.py \
  --manifest data/processed/train.jsonl --purpose training \
  --held-out data/processed/held_out.jsonl
```

Training checks require a nonempty held-out manifest and reject overlapping parent pages or exact images even if sample names change. This is not a perceptual near-duplicate detector: preserve parent-page provenance and review derived/augmented copies before training. Select thresholds/checkpoints on independent development data, then freeze decisions before test evaluation.

Store raw inputs and generated trajectories in ignored data directories or the external project dataset root. Publish source/release information, hashes, configurations and aggregate results after actual experiments. Public examples are generated synthetic fixtures.

The October 6 UniMER-1M pilot uses `page_id=unimer1m-image:<annotation_index>` as the available archive-image identity, not an asserted original-document identity. Its additional image/formula screening is recorded separately. Original document grouping and pretrained-model exposure cannot be established from this archive. Direct-answer SFT targets remain separate from the strict reference-free recognition/repair input files; no optimizer step has been run on the assembled records.
