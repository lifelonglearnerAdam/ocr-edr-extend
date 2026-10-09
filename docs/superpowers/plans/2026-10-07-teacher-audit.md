# Sealed teacher cohort verification implementation plan

> **For agentic workers:** Use `superpowers:executing-plans` inline for this serial audit. The existing research branch and draft PR remain the integration target; no new teacher decisions may be generated after reference access.

**Goal:** Verify the already sealed twelve training episodes and produce paired, explicitly provisional supervision views with full failure accounting.

**Architecture:** Keep generation/execution unchanged. A separate audit module verifies file and transition contracts; a CLI verifies the complete packet before opening mappings or training references, evaluates frozen final answers, and exports private paired views plus a numeric receipt.

**Tech Stack:** Python 3.10+, existing lxml/WeasyPrint renderer, pinned official OmniDocBench TEDS, unittest. No new dependencies.

**Spec:** `docs/research/INTERACTIVE_TEACHER_PILOT_20261007.md`.

## Global constraints

- Twelve unique admitted train documents only; retain all twelve, including abstentions and metric nonmatches.
- Maximum three edit attempts per case, including failed attempts; a terminal stop is mandatory.
- Frozen decisions, states and evidence are never edited after reference access.
- Mechanical validity, teacher self-check and published weak-reference agreement are separate fields.
- No independent visual judge, immutable teacher snapshot, student training, matched token budget, or generalization claim is implied by this data export.
- Provisional supervision admission is fixed **before scoring**: mechanical replay valid, terminal uncertainty `low`, valid final HTML, and both official normalized TEDS and TEDS-S full agreement within `1e-12`. Others remain in the full cohort and review queue. This conservative filter is not a correctness certificate.
- Both views use identical selected cases, sources, initial states and final answers. Token lengths/budgets are not yet matched; the existing single-step SFT kernel must not consume the trajectory export implicitly.
- Publish code, hashes and numeric/structural diagnostics; retain raw images, HTML, decisions and source-specific evidence locally.

## Task 1: Mechanical verification

**Files:** `src/ocr_edr/teacher_audit.py`, `tests/test_teacher_audit.py`.

**Interfaces:**

```python
verify_frozen_files(root: Path, manifest: dict[str, str]) -> None
replay_teacher_episode(task: dict, decisions: list[dict], events: list[dict],
                       state: dict, *, max_edit_attempts: int, renderer) -> dict
```

- [x] Add fixture transitions from a two-cell table, then tests for a successful edit/stop, stale observation, altered final state, changed render bytes, exhausted budget, post-stop events, and retained render failure.
- [x] Run `.venv-pilot/bin/python -m unittest discover -s tests -p test_teacher_audit.py -v`; observe missing implementation failures.
- [x] Verify complete file inventory and SHA256 values. Reject absolute/traversing paths, duplicates at JSON loading, missing or extra episode files.
- [x] Replay actual actions using the existing safe executor into a separate render cache. Check deterministic candidate/action results and the frozen state chain. Compare fresh PNG hashes/backend on original successful edits. Keep original render failures as rollbacks even if a fresh diagnostic render succeeds.
- [x] Re-run focused tests and the real twelve-case packet without reading references.

## Task 2: Train isolation, weak-reference metrics and paired export

**Files:** `src/ocr_edr/teacher_audit.py`, `scripts/audit_teacher_table_pilot.py`, `tests/test_teacher_audit.py`.

**Interfaces:**

```python
validate_teacher_mapping(mapping: list[dict], sources: list[dict],
                         excluded_families: set[str]) -> None
paired_teacher_views(episodes: list[dict], audits: list[dict]) -> tuple[list, list]
```

- [x] Add tests rejecting held-out roles, duplicated source documents, excluded families and held-out document/hash overlap; assert paired views have identical inputs and final answers while preserving a review-required episode in the full audit.
- [x] CLI arguments: `--packet`, `--seal`, `--dataset`, `--native-run`, `--official-root`, `--output`. Output must be new and outside the sealed packet.
- [x] Verify packet seal, prompt/task/source/event/decision hashes, executor snapshots, all twelve terminal chains and initial/final re-renders before reading `private-mapping.json` or references.
- [x] Validate source/provenance metadata and then hash/read only `train-references.jsonl`. Use pinned official normalizer symmetrically; check reference self-scores and finite `[0,1]` metrics.
- [x] Export `mechanical.json`, `evaluation.json`, `per_source.csv`, `final-only.jsonl`, `trajectory.jsonl`, `review-required.jsonl` and `run.json`. Views contain teacher targets, never reference targets or held-out data. Receipt states unmatched token budget and no training execution.
- [x] Recheck all frozen hashes after evaluation; retain output receipts for failure as well as success.

## Task 3: Evidence and publication

**Files:** research results/protocol/README/registry and `experiments/artifacts/interactive-teacher-pilot-20261007/`.

- [x] Write counts with denominators, official pre/post metrics, route breakdown, abstentions and disagreement descriptions. Explicitly distinguish raw-render repairs from metric normalization.
- [x] Archive sanitized receipts, numeric per-source results, source hashes and exact executed code snapshots. Generate and verify `sha256.json` after final edits.
- [x] Run full optional integration suite, Ruff and Black; review the diff and artifact contents for credentials, private paths, raw training content and unsupported claims.
- [x] Request code review using the existing reviewer and address material findings. Prompt-freeze/prepare→audit integration was reproduced, repaired and reviewed; 135 full-environment tests pass.
Integration target: the authorized existing draft PR #4, without merging. Git/PR readback records publication; the five-direction research goal remains active.
