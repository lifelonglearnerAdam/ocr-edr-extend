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

The original H1 claim that formula/table gains exceed text gains remains untested. This first diagnostic checks the narrower prerequisite that rendering can help repair formulas without damage.

A possible joint direction is a small, selective repair policy that explicitly controls harmful edits to tables and formulas. This is a candidate, not an established novel contribution. Compare related selective-prediction and structured-edit methods before naming it as the paper's contribution.

## Literature that changes the plan

[Table recognition judge study](https://arxiv.org/html/2607.13347v1) reports that iterative proposals can contain improvements while judge scores fail to select them reliably, with substantial preservation failures. This makes proposal quality and acceptance quality separate experiments.

[Jev rubric-judge study](https://arxiv.org/html/2609.29769v1) reports correlated errors between Jev and LLM judges; escalation does not automatically correct confident shared mistakes. [JEV-as-a-Judge](https://arxiv.org/html/2609.26550v1) studies a hosted decision interface. OCR-specific calibration, visual-input compatibility, model/version pinning, access and cost remain to be verified. No Jev calls or fees were incurred in this step.

## Immediate sequence

1. Complete the untrained 2B formula diagnostic and inspect each failure. It tests engineering and image fidelity, not benchmark quality.
2. Build real train/dev formula and table regions separate from frozen test pages, pin a full TeX/table renderer and the official evaluator, and obtain predictions from two parsers.
3. Measure proposal improvement and correct-input regression separately from a frozen judge's acceptance errors. Preserve rendering-equivalent inputs and include source counterfactuals.
4. Use those measurements to choose between better evidence/local edits, distillation, or verifier work. Only then spend resources on RL and broad transfer runs.

## October 4 checkpoint after official metric replay

The cell-map and genuinely larger-grid table controls are complete. The map preserves by stopping; the larger grids add harm and cost without repairs. Official core CDM replay is now operational, with all-reference selfchecks and exact fresh replay on the 16 inspected native formula sources. One mg content proposal improves 0.778 → 1.0, while the main repair/recognition means decrease. The original raster scores remain separate. CDM can miss one inspected subscript mismatch, and frozen reference disagreements can reward changes away from visible source details. See [the full October 4 results](RESULTS_20261004.md) and [the scoring protocol](CDM_PROTOCOL.md).

| Hypothesis | Current decision | Next experiment prerequisite |
| --- | --- | --- |
| H1 formula/table repair | Keep active; small untrained prompt interventions do not provide a net repair benefit | Improve proposal capability with independent train/dev supervision; evaluate useful local edits and preservation together |
| H2 agentic RL | Defer training until a functioning SFT proposal baseline and audited reward exist | Include structural/source checks and preservation; then compare SFT and RL with the same data, model and budget |
| H3 selective verifier/cascade | Keep active; metric/render/consensus agreement is insufficient alone | Separate proposal quality from acceptance errors; calibrate risk versus coverage/cost on untouched data |
| H4 small-model distillation | Keep active; the unit correction is a candidate capability signal, not enough training evidence | Train/dev-only licensed trajectories with source-grounded positives and preservation examples; no paid teacher calls without quota and authorization |
| H5 parser transfer | Unproven: native formula repair uses Nougat-LaTeX; published table demo parser remains unknown | Native outputs from a second identified parser, with one parser held out from tuning |

Do not train on these inspected development pages and call them untouched evaluation. Do not name rendering, GRPO, local edits, preservation or transfer alone as novel; the prior-art overlap remains unresolved. The next useful work is proposal supervision and independently checked acceptance, with a frozen data split and bounded compute, before scaling RL.

## October 6 supervision-data checkpoint

The official UniMER-1M training archive is downloaded and whole-file verified. A model-output-independent subset is frozen: 128 train and 32 dev source images, with preservation and controlled-symbol examples. Every source also has a new pinned Nougat-LaTeX image-only output; none is filtered by recognition quality. The assembled direct-answer records number 384 train and 96 dev. The complete 23,757-image UniMER-Test archive and prior inspected sources are screened out using exact checks and disclosed near-duplicate proxies.

This completes a limited data-preparation prerequisite for H1/H4, not an SFT result. Targets are released annotations, not teacher-generated trajectories; original document identifiers and pretraining exposure remain unavailable. No optimizer update, teacher call, learned acceptance or new repair evaluation occurs. See [the protocol](SUPERVISION_PROTOCOL_20261006.md) and [the stage report](RESULTS_20261006.md). The next step is a bounded, verified SFT implementation with dev excluded from optimization and preservation reported alongside controlled/native-input repairs. H2 remains deferred until that baseline and reward checks exist.

A subsequent campus RTX 4090 [single-step resource check](GPU_READINESS_20261006.md) verifies assistant-token masking, finite LoRA gradients and one temporary adapter update on two training examples. Peak allocated memory is 4.6844 GiB; no dev input or saved checkpoint is involved. This supplies runtime feasibility evidence only and does not change the above H1/H4 effectiveness or H2 decision.

## October 6 integrated experiment update

The [unified design](INTEGRATED_RESEARCH_20261006.md) combines the five hypotheses without claiming their combination is novel. Two direct-target LoRA arms now complete the fixed 192-step development screen; all three inference conditions retain 96 inputs from 32 sources. Both SFTs reach full core-CDM matches on the 32 controlled errors, while they fix no native metric nonmatch and introduce one source-confirmed extra-superscript regression. The two arms differ on only 2/96 final strings, so explicit preservation-pair benefit is not established. See [the complete scope and results](SFT_SCREEN_RESULTS_20261006.md); the earlier preparation and smoke notes remain dated stage records.

A new primary-source check corrects H3's interface assumption: `jev-1.13.0` is text-only, and its Choice confidence normalizes pmax by option count. The [current cascade](JEV_INTERFACE_CHECK_20261006.md) must use independently produced visual/structural evidence with full cost accounting and a simple-rule comparator. No hosted judge/teacher call is performed. H2/RL, teacher trajectories, independent acceptance/calibration, table supervision and held-out-parser evidence remain incomplete. Native source fidelity and selection, rather than controlled score saturation, are now the critical next comparisons.

## October 7 native parser and article-isolated table checkpoint

The frozen direct-SFT student is now tested on all 32 same development sources with a second identified parser, LaTeX-OCR. Official core-CDM mean increases 0.899375 → 0.98990625, with 6 full repairs, 4 partial improvements and no metric-matching regression; complete source panels preserve residual wrong symbols and unresolved glyph/reference concerns. This is a positive parser-shift development signal alongside the unchanged negative Nougat-input result, not independent locked transfer evidence. Read [the second-parser scope and findings](SECOND_PARSER_RESULTS_20261007.md) before making H5 claims.

PubTabNet's original article IDs expose 6,105 published train/val, 6,124 train/test and 216 val/test overlaps. The project excludes all published held-out articles from optimizer/model-dev data and prepares 128 train, 32 model-dev, 32 gate-calibration and 64 locked source articles with zero final pairwise document/image intersections. Controlled targets replay; real training-source inspection finds an omitted bottom row in reference p0002, flagged before any table training. Locked/calibration images stay uninspected by humans and unused for selection. See [table preparation and label limits](TABLE_DATA_RESULTS_20261007.md).

H1/H4 now have a bounded real-native formula repair signal and table supervision inputs; full native-table capability and teacher trajectories remain unproven. H3 needs independent visual evidence and calibration, H2 needs audited reward/acceptance before GRPO, and H5 needs untouched source groups plus independent confirmation. Do not train or retune on the existing development audit and label it a test. No paid teacher/Jev call or new table optimizer update occurs in this checkpoint.
