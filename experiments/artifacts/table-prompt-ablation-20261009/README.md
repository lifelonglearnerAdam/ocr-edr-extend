# Fixed-checkpoint prompt ablation — 2026-10-09

Six new complete103-case conditions reuse the same base/two terminal381-step
adapters. Literal309 raw calls replay exactly; both prompt cohorts match except
the response-shape example appendix. Results and all denominators remain visible.

Descriptive all: TEDS0.994361998,31/71 full fixes,0/32 matching regressions.
All31 fixes are synthetic extra-row repairs; none of the32 number or7 span cases
are fixed. Descriptive no-preservation: TEDS0.977806481,48/71 fixes (32 extra-row,
16 numeric),14/32 regressions. First literal negative outputs remain preserved.

See the [complete explanation](../../../docs/research/TABLE_PROMPT_ABLATION_RESULTS_20261009.md).
This is post-hoc, single-seed inspected development. Removing examples also removes
90 input tokens and matches the training prompt; it does not isolate a pure
semantic example effect or establish independent native/parser transfer.

Public artifacts retain numeric evaluations, document statistics, cost/call hashes
and exact executed sources. Raw source images, target/predicted HTML and weights
remain local. `sha256.json` covers this public archive.
