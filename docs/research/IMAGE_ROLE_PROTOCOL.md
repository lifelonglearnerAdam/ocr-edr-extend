# Image-role diagnostic, fixed before inference

Date: 2026-10-03. This is exploratory follow-up to the October 2 pilot, not a new independent holdout or a benchmark claim.

## Question

The original two-image prompt fixes fewer errors than source-only repair. Is that associated with image order, labeling each image at its position, or simply processing more images?

Use the same pinned untrained Qwen2-VL-2B model, all 20 existing cases, fixed greedy decoding, 128 output tokens, the same image resize limits and CPU bfloat16 inference. Ground-truth files and branch labels remain inaccessible to the inference script. No parameter or prompt is selected from correctness scores during this run.

## Conditions

1. `source_only`: reproduce the one-source baseline.
2. `source_first`: reproduce the original source-then-candidate message.
3. `source_last`: reverse the two images and correctly update their textual role indices. The edit instructions and current prediction stay the same.
4. `labeled_source_first`: place a short source/candidate label immediately before each image, keeping source first.
5. `labeled_source_last`: the same interleaved labels with source last.
6. `duplicate_source`: two copies of the source, explicitly described as duplicates. This controls for a second image without conflicting visual content.

Image order with appropriate role-index changes and interleaved labels are prompt interventions, not a trained algorithm. Compare one-factor contrasts: source-first versus source-last; source-first versus labeled-source-first; labeled-source-first versus labeled-source-last. Token counts differ slightly with labels and necessarily differ from the one-image baseline; record them rather than claim exact cost matching.

## Measurements and interpretation

Keep raw output, ordered image hashes, the serialized message content, the formatted chat-template hash, image-grid shape, tokens and timing. Compare same-renderer raster proxy, normalization-sensitive exact string, incorrect-to-correct edits, correct-to-incorrect edits and copying the current candidate. Equivalent controls remain whitespace-only. References enter only a separate offline evaluator.

Reproducing October 2 source-only/source-first outputs is an implementation check. Strong order dependence would justify source-grounding controls in subsequent experiments. It would not prove the model ignores a particular image: behavior, not internal attention, is measured.

After inspection, freeze a separate protocol for independently sourced UniMER formula images. Use those accessed samples as exploratory development cases, preserve their dataset IDs/hashes, and exclude them from any later frozen test claim. Do not select cases by model correctness.
