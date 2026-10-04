# Image-only recognition control

Fixed October 4, 2026 after the October 3 repair failures were evaluated. This is an exploratory development experiment on all 16 already inspected sources, not a holdout or a prompt selected for benchmark claims.

Question: can the same untrained 2B model recognize the source at all when the potentially wrong native prediction is absent? The direct-transcription instructions differ from the repair instructions, so this experiment jointly changes task framing and the presence of the initial candidate; it cannot by itself isolate anchoring.

Use the exact same images and pinned Qwen2-VL-2B snapshot, CPU bfloat16, eight threads, greedy decoding and 128 output tokens. Run two fixed conditions: original 100352–200704 image-pixel bounds, and 100352–802816 bounds. Images smaller than the upper bound may receive identical processor grids; report actual grids/token counts. No reference or initial prediction enters either prompt. The existing native prediction is joined only in offline evaluation to measure matching-to-nonmatching transitions.

The fixed prompt is: “Transcribe the mathematical expression in the image into LaTeX. Preserve every visible symbol, subscript, superscript, fraction and delimiter. Do not solve or simplify it. Return only the expression inside <latex>...</latex>. Do not describe the image or return bounding-box coordinates.” No examples are added and no per-image retries or prompt optimization are permitted.

Keep raw strings, ordered image hashes, serialized messages, processed grids, output-cap status and CPU generation costs. Evaluate raw candidates with the frozen original TeX adapter, then separately report whole-display-wrapper normalization as a post-hoc adapter control. An unrenderable recognition is a failed recognition and receives no baseline rollback; unchanged parser output is reported separately. Exact-raster/string scores remain proxies, with the existing annotation concerns unchanged. Higher-resolution calls have different costs and must not be described as budget-matched gains over the low-resolution repair runs.
