# Table development pilot, fixed before model inference

Date: October 4, 2026. This is a small engineering/research diagnostic on already published OmniDocBench demo records; it is not a full benchmark evaluation, a frozen test, or evidence of transfer across named parsers. The demo prediction records do not identify their parser.

## Frozen selection and source mapping

Pin official repository revision `f133a71e9e91c3621c7ce8994200a7b394a06eb3`. Replay all ten published table records through the unmodified upstream TEDS implementation on their saved normalized strings; require TEDS and TEDS-S reproduction within 1e-12 before repair inference. The current upstream preprocessing is applied equally to reference/initial/final strings in the new evaluation; its outputs can differ from normalization used by the saved historical demo result, so report newly recomputed paired baselines rather than mixing the two.

Select the first four eligible table records by SHA-256 of `20261004:img_id:gt_position`, with at most one per source page. Eligibility is released normalized reference length <=850 characters and <=32 HTML cells, complete supported HTML, and a resolvable page annotation at the same table order. Selection does not depend on baseline score or repair output. Preserve all ten eligibility decisions. Crop the official polygon bounding rectangle with floor/ceil, add 12 white pixels, and retain page hashes, crop coordinates, annotation IDs and source crop hashes. All accessed pages/regions are development data and are excluded from later frozen tests.

Each selected source gets a released-reference control, a fixed first-digit content perturbation, its original published demo prediction, and (if present) a one-unit reduction in the first rowspan/colspan greater than one. Inputs carry only IDs, source images/hashes and initial HTML. References, variant names and stored scores are evaluator-only. Correct controls mean agreement with the released annotation, which may contain errors; inspect source/annotation disagreements without changing frozen labels.

## Fixed proposal conditions

Use the same pinned Qwen2-VL-2B snapshot, CPU bfloat16, eight threads, greedy decoding, original 100352–200704 image-pixel bounds, and 512 new-token cap. Every condition receives the same initial HTML and the source image. Run:

1. Source-only full HTML rewrite.
2. Source-first/current-render-second full HTML rewrite.
3. Current-render-first/source-last full HTML rewrite.
4. Source-only single bounded edit, using exactly one JSON action: stop, replace one cell's plain text, update one cell's positive row/column spans, or delete one row.

The edit contract addresses zero-based rows and cells in the initial DOM order. No reference-derived error location is supplied. The initial table is provided as text, not a separate labeled cell map. A malformed or non-renderable response rolls back to the fixed initial HTML. A valid response does not imply correctness. If initial rendering fails, retain the sample and preserve its baseline for the two-image conditions. The bounded edit can still run using the source and text. No adaptive retries, teacher calls or training occur.

HTML rendering uses pinned WeasyPrint with fixed fonts/CSS and a one-page budget. It is evidence for the model rather than a visual-consistency verdict. The four methods use the same token cap, but image/token usage and output length differ; report actual costs. Bounded edits restrict the accessible solution space and cannot repair every native error in one action. Do not claim equal capability or a new algorithm from this control alone.

## Evaluation

Use unmodified, checksum-pinned official TEDS/TEDS-S and HTML normalization on every retained case. Keep fixed sample/page/annotation matching; no end-to-end rematching occurs. Report content and structure scores, full-set paired deltas, initially matching regressions, improved/degraded inputs, syntax/adapter/render failures, output truncation and CPU generation time. Average by source page and by case; variants of a source are correlated, so do not treat them as independent benchmark pages. No visual-correctness labels or benchmark claims are inferred from TEDS=1 alone.

## Source-metadata correction before inference

Preparation stopped before any table model call because `notes_f7f010b78016aeebd76e56d9283eb67f_49.jpg` declares 729×516 but its actual image is 516×729. The polygon at (45,567)–(450,676) fits the actual image and its direct crop was inspected: it contains the intended handwritten number/word table. There is no EXIF rotation. Preserve polygon coordinates without rescaling; explicitly allow reversed width/height metadata only when the rectangle fits the actual image, record both dimensions and the exception, and reject other mismatches. Selection, source count and prompts are unchanged.

The same reversed-metadata exception was recorded for `jiaocaineedrop_Chapter9.pdf_46.jpg`: declared 2178×1700, actual 1700×2178. Its direct polygon crop contains the full Birthday Money table. Before inference, all four final crops and all 14 initial renders were inspected. Separate audit entries flag source/reference disagreements on `t001` and `t003`; neither the selection nor the frozen labels were changed.
