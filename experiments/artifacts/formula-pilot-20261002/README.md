# Controlled formula pilot artifacts

This package contains the October 2, 2026 diagnostic described in [the result report](../../../docs/research/RESULTS_20261002.md). It is synthetic evidence for model behavior, not benchmark data or proof of a paper contribution.

- `data/inputs.jsonl` and `data/images/`: inference inputs with source hashes.
- `data/references.jsonl`: offline evaluation labels; never pass this file to inference.
- `predictions.jsonl`: unchanged baseline plus all four proposal conditions, with raw text and call traces.
- `run.json`: pinned model/code/dependencies and actual run metadata, with file paths made portable.
- `evaluation.json`, `summary.csv`: same-renderer exact-raster and normalized-string diagnostic proxies.
- `diagnostic_cases.png`: six visually inspected examples.
- `checksums.sha256`: hashes of the original exported evidence (before this README).

Use the documented evaluator to replay the scores. Evaluation writes `evaluation_renders/`; that directory is a generated cache, not evidence. Compare regenerated evaluation and CSV hashes to the checksum file. Inputs derive from four templates and must not be treated as independent benchmark pages.
