# Table SFT admission and readiness receipts — 2026-10-07

This package records preparation and software verification, not completed table
training or a quality improvement. Table optimizer steps at this archive cut: **0**.

- `admission-summary.json`: 127 documents / 406 cases after excluding the entire
  previously reviewed p0002 family. Original data remain frozen. Absolute source
  path spelling is normalized in this public copy; the original receipt hash is kept.
- `token-preflight.jsonl` / `preflight-summary.json`: every admitted training example
  fits the frozen 4096-token cap; actual document-balanced exposure/token totals
  differ between the two arms and are reported. These contain counts/identities,
  no training targets or source image bytes.
- `model-dev-reference-audit/`: all32 source/reference observations, their scope,
  panel hashes and panel-generation procedure. This is a single assistant visual
  screen without consulting model outputs, not independent human gold labeling.
  A suspected panel omission was cleared against the original raster/PDF/annotation;
  it is not counted as an annotation defect. No references or cases were changed.
- `source/admission-executed`: exact source saved at admission execution, before
  later loader additions. `source/inference-executed`: source frozen for the actual
  complete-input CPU base inference; archive existence does not mean that run has
  finished. `source/readiness`: reviewed current training/evaluation code and tests.
- `verification.json`: 117 full-environment tests, formatting, same-body formula
  optimizer extraction, resolved paired-comparison review finding, and separately
  verified remote source/model staging. Staging is not a training invocation.

Reproduction and limitations:
[readiness report](../../../docs/research/TABLE_SFT_READINESS_20261007.md) and
[frozen protocol](../../../docs/research/TABLE_SFT_SCREEN_PROTOCOL_20261007.md).
Apply the frozen inference overlay to the base commit recorded by its run before
replaying old results. Do not substitute later source and describe its hash as the
original run's hash.

Image rights remain article-specific; no source images or rendered panels are in
this package. See the earlier [PubTabNet provenance/license record](../pubtabnet-preparation-20261007/README.md).
No model weights, raw model-call text, passwords or host authentication files are published.
