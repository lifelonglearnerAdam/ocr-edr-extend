# Research decisions: first evidence before heavier training

Date: 2026-10-02. Target: a strong, reproducible paper; venue choice follows the eventual contribution and evidence. The October 19 milestone is a pilot and an implementation decision, not a paper submission promise.

## What the base paper already does

[OCR-EDR v1](https://arxiv.org/html/2609.03445v1) already combines source images, symbolic predictions, updated rendering, diagnosis/localization, curriculum SFT, and GRPO. Its model is Qwen3.5-9B, and its frozen benchmark covers text and formulas. Cross-parser formula refinement is already evaluated. We must not describe rendering, GRPO, preservation rewards, or cross-system testing alone as our new contribution.

The paper text was inspected directly on October 2. Its reported scores remain author-reported, not reproduced here. No author release link was found in the inspected HTML; exact-name Hugging Face searches for DocEDR models and OCRErrBench datasets returned empty lists. This is a bounded release check, not proof that no release exists anywhere. Reproduction availability remains open.

## Hypotheses and stopping rules

| Idea | Possible contribution to investigate | Essential comparison | Cheap falsification step | Decision rule |
| --- | --- | --- | --- | --- |
| H1: formula/table repair | Table structure and content preservation through bounded edits supported by visual evidence | Unchanged prediction, source-only repair, one-pass render repair, fresh/stale iteration; later an available DocEDR implementation | Controlled formula proposals now; then real table cell/row/span errors with a fixed edit budget | Continue only if paired full-set improvement survives correct-input regressions. If localization cannot support reliable edits, investigate the verifier before scaling. |
| H2: agentic RL | A measurable advantage of learned edit/render/stop decisions under a cost and regression constraint | Same student, training data, context and inference budget for SFT versus RL | Small SFT control plus reward audits before any GRPO run | Stop if apparent gains disappear when budgets match, or reward optimization increases verified regressions. GRPO by itself is already in the base paper. |
| H3: confidence cascade | Reliable selective acceptance at a measured cost under parser shifts | Frozen single judge, cascade, and independent visual/structure checks | Offline false-acceptance versus coverage curve on development cases, including confident errors | Reject a gate if its cheap confident acceptances violate the predeclared regression budget on held-out data. Do not assume vendor confidence transfers to OCR. |
| H4: teacher distillation | A small student that retains image fidelity and correct-input preservation at useful cost | Same small model before distillation and with direct-answer versus trajectory SFT | A modest train/dev-only set of teacher examples; counterfactual source controls | Continue if it improves held-out net repairs and preservation. No paid teacher generation starts without available quota and authorization. |
| H5: transfer | Evidence that the selected mechanism survives changed OCR error distributions | At least two parsers, with one parser held out from tuning; eventually the full parser matrix | Fixed train/dev parser and blinded second-parser pilot | Do not claim generality from one parser or selected Bad cases. Report all inputs, per-parser regressions and costs. Transfer is evidence, not a new algorithm by itself. |

A possible joint direction is a small, selective repair policy that explicitly controls harmful edits to tables and formulas. This is a candidate, not an established novel contribution. Compare related selective-prediction and structured-edit methods before naming it as the paper's contribution.

## Literature that changes the plan

[Table recognition judge study](https://arxiv.org/html/2607.13347v1) reports that iterative proposals can contain improvements while judge scores fail to select them reliably, with substantial preservation failures. This makes proposal quality and acceptance quality separate experiments.

[Jev rubric-judge study](https://arxiv.org/html/2609.29769v1) reports correlated errors between Jev and LLM judges; escalation does not automatically correct confident shared mistakes. [JEV-as-a-Judge](https://arxiv.org/html/2609.26550v1) studies a hosted decision interface. OCR-specific calibration, visual-input compatibility, model/version pinning, access and cost remain to be verified. No Jev calls or fees were incurred in this step.

## Immediate sequence

1. Complete the untrained 2B formula diagnostic and inspect each failure. It tests engineering and image fidelity, not benchmark quality.
2. Build real train/dev formula and table regions separate from frozen test pages, pin a full TeX/table renderer and the official evaluator, and obtain predictions from two parsers.
3. Measure proposal improvement and correct-input regression separately from a frozen judge's acceptance errors. Preserve rendering-equivalent inputs and include source counterfactuals.
4. Use those measurements to choose between better evidence/local edits, distillation, or verifier work. Only then spend resources on RL and broad transfer runs.
