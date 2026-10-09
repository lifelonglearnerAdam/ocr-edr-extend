# Independent supervision preparation, 2026-10-06

This receipt package records a verified official UniMER-1M train/dev subset: 128 training and 32 development source images, with preservation and controlled-error inputs frozen before new native model calls. The pinned Nougat-LaTeX recognizer then produces all 160 native outputs without target-reference access. Released targets, source images and raw recognition records stay in ignored local data/run directories.

The assembled direct-answer dataset has 384 training and 96 development examples: one preservation, one controlled-error and one native-input example per source. No teacher generation, optimizer step or model-weight update is performed.

`dataset_receipt.json` hashes every frozen image and manifest; `independent_checks.json` records separately checked overlap and image integrity. The preparer blocks all 23,757 available UniMER-Test images, indexed references and prior inspected sources, using exact and disclosed proxy-near-duplicate checks. Original document IDs and model-pretraining exposure are unavailable; image/formula grouping does not establish original-document independence.

`supervision_summary.json` gives native coverage, costs and whitespace-removed string-equality counts. That proxy is not CDM or source-grounded correctness. The fixed first eight source/reference pairs from each split were visually inspected; not all labels are independently validated.

Read the repository protocol `docs/research/SUPERVISION_PROTOCOL_20261006.md` before reproduction. The included experiment scripts run from a source-compatible repository root and refuse to overwrite prepared outputs. Download the pinned official archives according to the upstream dataset terms and verify the recorded whole-archive hashes. `source_snapshot/` retains the invoked existing recognition and rendering code; no images, released target text, weights or credentials are bundled here.

Current checks: all 80 local tests pass with real Tectonic, table-rendering, official TEDS and official core CDM integration enabled. This verifies those software paths, not SFT performance or a benchmark gain.
