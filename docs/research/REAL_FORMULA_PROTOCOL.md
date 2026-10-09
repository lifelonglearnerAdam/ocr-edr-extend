# Independent-source formula development pilot

Protocol fixed on 2026-10-03 before model inference. This study probes model behavior on real published images; it is not a full UniMER evaluation or proof of cross-parser gains.

## Data

Use the official `wanderkid/UniMER_Dataset` archive at revision `2343ddd963290469da36ca83e3a56c66e068add9`, SHA-256 `9bf370b8cac868fee84835f40dec26c477430611253e3feb681356a8149a3a90`. Follow upstream `Im2LatexDataset`: numeric image filenames index each split's annotation file. Select 8 SPE and 8 SCE images in a fixed SHA-256 order, applying only predeclared renderer/length/perturbation eligibility rules. Log every examined image and exclusion reason; do not filter on model performance.

The restriction to MathText-supported formulas and labels of at most 200 characters is a temporary engineering limitation, not a representative benchmark sample. Preserve all selected dataset IDs and source hashes. All inspected samples are development data and must be excluded from later frozen tests. Since these are published benchmark images, model pretraining overlap cannot be ruled out.

For each source make a correct-input control and one visible symbol perturbation. Keep labels in a separate evaluator file. This is a controlled-error diagnostic on independently sourced images, not native OCR-error evaluation. Separately run a pinned independent Nougat-LaTeX recognizer using its author's custom image processor to obtain native predictions; any later repair selection must report all cases and renderer failures.

## Methods

On the controlled-error cases run source-only repair, original source-first/candidate-second repair, and source-last/candidate-first repair. Fix model snapshot, decoder, image limits and budget to the image-role study. Do not optimize prompts against these correctness scores. Keep the other role conditions exploratory; no best-of selection is a deployable result.

For native recognizer outputs, preserve an unchanged baseline and compare source-only/source-first/source-last proposals on the same sources. If an initial prediction cannot render, record a failure and preserve its baseline for the two-image arms; do not remove the image from the full-set denominator. Keep fixed initial predictions across all methods.

## Evidence

Record raw strings, messages and ordered image hashes, processor grids, model/source versions, token caps, truncated outputs, runtime and per-image score proxies. Report initially correct inputs separately from incorrect inputs, edits and renderer rollbacks. Exact string and same-renderer pixel equality are conservative proxies; neither is CDM/VisFix. A model paraphrase may be visually equivalent yet fail the proxy. Inspect disagreement images before drawing conclusions.

The research decision concerns reliable source grounding and preservation in small repair policies. Rendering and GRPO are already contributions of the base OCR-EDR paper. This development pilot can motivate an extension experiment but does not establish novelty.

## Rendering update before the real batch

A bundled Tectonic 0.17.0 runtime passed its smoke compile. After the MathText subset was selected but before any repair inference, full TeX rendering became available. The source selection remains exactly the frozen MathText-supported 16 images; it is not reselected to improve scores. Real controlled and native-output runs will use the full TeX renderer for candidate evidence and offline same-renderer comparison. Retain the MathText selection bias in every result claim. Rendering success still does not imply correctness. The official CDM environment/metric has not been reproduced.
