# Article-isolated table data preparation, 2026-10-07

The primary PubTabNet repository links the pinned 11.24 GB mirror archive. Whole-file SHA-256 is verified. Actual inventory is 500,777 train / 9,115 val / 9,138 test images, with annotations on train/val only. Published splits share original articles: 6,105 train/val, 6,124 train/test and 216 val/test overlaps. The full 17,794 published held-out articles are excluded from optimizer/model-dev selection; test articles are also excluded from calibration/locked evaluation.

Final roles are 128 train, 32 model-dev, 32 gate-calibration and 64 locked-evaluation articles, one table per article. Final pairwise article, byte-image and decoded-pixel overlap is zero. All bounded controlled action targets replay to the frozen canonical annotation; these are annotation-derived controls, not native parser errors or teacher trajectories.

There are 410 train, 103 model-dev, 102 calibration and 213 locked cases. Only train/model-dev targets are assembled as SFT records. No table training step is performed. Calibration/locked images are not visually inspected or used for model selection. Selection is restricted by cell count, size and controllable-error existence, so this is not a representative full benchmark sample.

A fixed first-eight training source/annotation review finds an extra bottom row in source p0002 missing from the released annotation. Frozen labels are unchanged; label review is required before this family is admitted to training. No statement that all remaining labels are correct is supported by this limited audit. Valid HTML and TEDS target selfchecks do not establish full source fidelity.

Primary annotations license is CDLA-Permissive-1.0; image rights follow the PMC article conditions. The mirror's CDLA-Sharing-1.0 card differs and that metadata conflict is retained. Raw images, annotations, targets and model outputs are excluded from this artifact. Counts, hashes, exact preparation code, audited observations and primary-source receipts are included. Model pretraining overlap is unknown. Read `docs/research/PUBTABNET_PREPARATION_20261007.md` before reproduction.
