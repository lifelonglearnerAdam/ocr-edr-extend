# Native table development and coordinate compatibility — 2026-10-07

Two complete32-source runs are preserved. The unmodified selected author/export
combination produces a coordinate-frame mismatch and is retained as an integration
diagnostic, not a weak student-repair baseline. An explicitly recorded original-frame
decoder override yields mean TEDS0.982848 and13/32 reference-full-matches. This is
not a Qwen/SFT/RL gain and not a parser-unseen benchmark.

See [results](../../../docs/research/NATIVE_TABLE_RESULTS_20261007.md),
[original protocol](../../../docs/research/NATIVE_TABLE_PROTOCOL_20261007.md) and
[coordinate addendum](../../../docs/research/NATIVE_TABLE_COORDINATE_ADDENDUM_20261007.md).

- Inference/evaluation receipts preserve original file hashes and replace host-local
  source/model paths with logical roots. Raw HTML/model-call strings are not included.
- Both evaluations retain all32 sources and official numeric metrics/DOM diagnostics.
- Source snapshots preserve both actually executed wrappers; the author code remains
  identified by commit and per-file hashes. Runtime overriding is explicitly disclosed.
- `coordinate-roundtrip*.json` use real author encoding with synthetic coordinates,
  requiring no reference labels. `coordinate-condition-comparison.json` checks all32
  pairs for identical OCR outputs/structure and the expected coordinate scaling.
- `source_frame_diagnostics.json` is a reference-free descriptor, not a calibrated
  acceptance policy. Same OCR confidence vectors accompany different coordinate validity.
- Four selected source/raw-render/reference audits are scoped assistant observations,
  not random-sample prevalence estimates or independent human labels. Panel hashes and
  generation source are retained, but source images/panels are not redistributed.
- TEDS-S's formatting-node denominator and metric normalization's interpretation of
  escaped format tags are explicitly distinguished from actual raw rendering fidelity.

Official checkpoint URLs/hashes and the isolated CPU package list are recorded;
upstream did not supply immutable SHA256 manifests. No model weights, raw reference
HTML, source images, authentication files or account credentials are published.
