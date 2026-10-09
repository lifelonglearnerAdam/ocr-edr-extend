# Read-only workspace and interrupted-run recovery — 2026-10-09

The original NTFS volume became read-only and its Git index/ref were unreadable.
A new ext4 checkout recovered the same published commit and readable experiment
inputs/evidence, with no repair or mutation of the source volume. See the
[recovery protocol](../../../docs/research/RESEARCH_RECOVERY_20261009.md).

The completed 381-step all-arm checkpoint is reused after hash verification. The
second arm had only 62 logged steps and no checkpoint when its process disappeared
after reboot. It is restarted from the original base under the same frozen config;
248 interrupted exposures and 334.892 seconds inner-loop time remain accounted.
The preflight/schedule and first62 restarted loss values reproduce exactly.

All12 pinned model files, the completed adapter, train/dev role identities and22
execution/statistics files were independently checked.144 full-environment tests
pass with no optional skips after restoring package-specified executable modes,
links and private TeX runtime roots. Original validation failures are retained.

An independent final verifier handles the supervisor's statistics receipt-name
mismatch by verifying all scientific outputs and recomputing document statistics.
It never relabels a training/inference failure or overwrites the original controller
receipt. Complete two-arm quality results are pending at this recovery archive stage.

This public archive contains numeric recovery facts, original-file hashes and
relative/logical identities only. Raw data, package contents, checkpoint weights,
private copy manifests, account paths, keys and credentials are not redistributed.
