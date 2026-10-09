# Initial-HTML cell-address diagnostic

Date: October 4, 2026. Freeze before inference. This is a development intervention selected after inspecting the original table failures. It cannot establish a held-out gain, a causal attention explanation or a novel method.

## Hypothesis and fixed population

The first JSON editor accepted three edits at wrong row/cell addresses and produced six invalid schemas. Test whether exposing all DOM addresses from the initial HTML improves executable local repairs. Keep every one of the original 14 inputs from four source pages, including all released-reference controls, content/span perturbations and published demo predictions. Keep the original source/reference disagreements and unknown demo-parser identity explicit. Do not alter labels, exclude failures, select cases based on the new output, or provide a reference-derived target.

The frozen input SHA-256 is `447aa0d9294b60650161dbe3678bb08c3d8b411af1e54addd84fd7f8438aa215`; the frozen reference SHA-256 is `847271cd7c76ceaf3199bed408881453985ad71d14c27fb7d9b4514b2764fc33`. Input images are the exact original polygon crops, with their previously recorded hashes. All four source pages remain development data.

## Two proposal conditions and a zero-call baseline

1. `source_only_patch`: reproduce the original prompt, image and action schema exactly. Tests compare the full prompt and messages against all 14 archived calls. Rerun all calls; compare raw outputs/actions/finals with the original run as a reproduction control.
2. `source_only_patch_indexed`: append a `<cell_map>` JSON array between the initial table and the unchanged output contract. Supply every initial cell's zero-based `row`, zero-based `cell`, tag, rowspan, colspan and plain text. Both the map and the action adapter use current DOM row/cell order. Addresses are recomputed from each input, with no target selected or highlighted. No visual bounding box or correct error location is supplied.
3. `unchanged_0`: retain the exact initial string at zero model calls.

Use the same snapshot `895c3a49bc3fa70a340399125c650a463535e71c`, Qwen2-VL-2B-Instruct, CPU bfloat16, eight threads, greedy decoding, seed 20261004, pixel bounds 100352–200704, and 512 new-token cap. There are exactly 28 calls: baseline then indexed for each case, in the existing frozen input order. Do not add retries, output-dependent prompt changes, teachers, training or a visual judge. Record exact code/model/dependency versions, serialized messages, image hashes, actual processor grids, tokens, generation time, cap hits, raw actions, schema/render failures and rollbacks.

The atomic action set remains stop, replace one cell's plain text, set one cell's positive spans, or delete one row. It cannot express every native error, notably missing-row insertion. Invalid actions and failed renders roll back under the same frozen policy. Adding the map also repeats content and increases input tokens; this comparison measures the whole explicit-map intervention, not an isolated addressing mechanism or equal token cost.

## Evaluation and decision

After inference completes, use the pinned unmodified OmniDocBench normalization and TEDS/TEDS-S at `f133a71e9e91c3621c7ce8994200a7b394a06eb3`. Retain fixed case/page/annotation pairing and all 42 output rows. Report case/page means, each variant separately, exact fixes, improved/degraded inputs, initially matching regressions, accepted changes, stops, failures, cap hits and cost. Verify 28 calls and hashes against receipts before scoring.

Inspect every new accepted change against the source and initial cell addresses; distinguish wrong-address writes, copied wrong text, schema failures, structural edits and deliberate stops. Keep these observational audit notes separate from frozen benchmark references, without manufacturing new visual accuracy labels. A metric plateau can hide a harmful cell change; TEDS matching alone does not prove image fidelity.

Continue toward verifier/SFT design only if the proposal pool contains useful source-grounded native/local repairs at a measured preservation cost. If addresses merely reduce schema failures, or the model mostly copies/stops or edits wrong content, the resulting bottleneck remains proposal/recognition quality. A positive development result still needs an untouched evaluation and a second identified parser; this prompt intervention is engineering evidence, with novelty unproven.
