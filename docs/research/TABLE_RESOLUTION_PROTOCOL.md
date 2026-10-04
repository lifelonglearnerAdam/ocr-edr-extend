# Table visual-resolution development control

Date: October 4, 2026. Freeze before model inference. This follow-up is chosen after the negative cell-address diagnostic. It tests a specific implementation prerequisite on already inspected development pages, not a held-out gain or a new method.

## Question and fixed inputs

Does a genuinely larger processed source-image grid improve the original bounded JSON repair prompt? The cell-map run prevented edits mostly by stopping; it did not establish that cell addressing was the main bottleneck. Retain all 14 original inputs from four source pages, including released-reference controls, digit/span perturbations and published demo predictions. Do not highlight errors, supply target coordinates, modify labels, select favorable outputs or add a map.

Input SHA-256: `447aa0d9294b60650161dbe3678bb08c3d8b411af1e54addd84fd7f8438aa215`. Offline reference SHA-256: `847271cd7c76ceaf3199bed408881453985ad71d14c27fb7d9b4514b2764fc33`. Reference metadata and prior audit notes remain outside model messages. All four pages remain development data with correlated repeated inputs, documented source/reference disagreements and an unidentified released-demo parser.

## Conditions, exposure and budget

Use the original `source_only_patch` prompt, one source crop, the unchanged strict action schema and adapter. Keep the snapshot `895c3a49bc3fa70a340399125c650a463535e71c`, CPU bfloat16, eight threads, seed 20261004, greedy decoding, SDPA and 512 new-token cap.

1. Low: min pixels 100352, max pixels 200704. Rerun every original call as a reproduction comparator.
2. High: min pixels 802816, max pixels 802816. This changes the processor representation even for a small crop; it is interpolation, not a higher-resolution source acquisition.
3. Unchanged: initial bytes at zero model calls, included separately with each condition's output.

There are exactly 28 model calls: all 14 low inputs in frozen order, then all 14 high inputs in that same order. Two inference roots produce 56 output rows, including two identical sets of unchanged baselines; these are not 56 independent samples. No additional model calls, retries, output-dependent prompts, training, teacher generation or learned judge are allowed. The larger condition receives more visual tokens, so this is a resolution/cost intervention, not matched input-token cost or a pure capacity test. Call-count and output caps are matched; generation times can also be affected by fixed condition order and CPU state.

Before loading model weights, run the exact pinned processor over each message and image. Record the actual grids, processed token counts, serialized message/prompt/template/image hashes and original pixel dimensions. Require a larger grid for every source family; otherwise halt without claiming a resolution contrast. The preflight observed low/high grids are t001: 28×34 / 60×70; t002: 14×42 / 36×116; t003: 14×64 / 32×130; t004: 20×28 / 56×76. Verify these again in actual model-call receipts. Freeze code, protocol and all 28 calls before inference.

## Scoring, audit and decision

Use pinned unmodified OmniDocBench normalization and TEDS/TEDS-S at `f133a71e9e91c3621c7ce8994200a7b394a06eb3` with fixed sample/page/annotation pairs and all inputs retained. Check every run/source/prompt/image hash and call count before scoring. Compare every low raw response/action/final string/grid/token count with the preceding address-run original arm; disclose differences. Do not loosen JSON parsing after seeing high outputs.

Report per-case, per-variant and page/case means; nonmatching fixes, improved/degraded cases, matching-control regressions, stops, schema/render failures, cap hits and full generation/token cost. Separately inspect every accepted high change against source and initial addresses. Correct content at a wrong address is still a harmful edit. TEDS can miss cell damage and released references can disagree with images; keep observational audits separate from frozen labels.

Continue toward SFT/verifier work only if the proposal pool contains useful source-grounded repairs at a disclosed preservation/cost tradeoff. If higher visual exposure only changes confidence, syntax or stopping without useful repairs, image bounds alone are insufficient and the next work should address proposal supervision/capability and identified-parser evaluation. Any positive development signal requires an untouched sample set and relevant comparators before a paper-quality claim.
