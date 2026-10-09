# Official core CDM replay of frozen formula outputs

Date: October 4, 2026. Commit this evaluation protocol and runner before scoring the existing development outputs. No new model, teacher, training, or inference call is part of this experiment.

## Question and unchanged data

Same-renderer exact-raster comparisons can miss useful partial corrections or disagree over styling. Re-evaluate every saved native/parser, repair and image-only output using the unmodified official OmniDocBench `cdm_metrics` core callable. Separate full matches, partial improvement, harmful changes and failed/empty predictions. This is fixed-pair offline metric replay, not the full OmniDocBench preprocessing/matching pipeline or a benchmark result. Do not replace earlier raster scores or use offline references to generate or select new candidates.

The 16 UniMER sources were selected on October 3 using label-length and MathText-renderability restrictions and have all been inspected. They remain development forever. All existing source/reference disagreements, original output failures, cap hits and selection limitations remain recorded; do not silently relabel, omit, or retry them.

Use the native reference file from the image-only package. Its 16 reference strings match the original source annotations by family. Before scoring, verify every checksum in both source evidence packages and require these SHA-256 values:

| File | SHA-256 | Output rows |
| --- | --- | ---: |
| `unimer-development-20261003/native-tex/predictions.jsonl` | `fcfc9f4fe7fc3418cef33564e0c8b5699c09ce2895f058bb942c87a1fe685a99` | 64 |
| `unimer-development-20261003/native-tex-adapter-audit/predictions.jsonl` | `dcdf8da6d829ff3b97ea7621d6f7c7e23301f21fc22bd89349868f7b9474762d` | 128 |
| `unimer-development-20261003/native-tex-agreement-audit/predictions.jsonl` | `cfdb7dedd1a07a6ca9a2925786038a441cf88f1e16c87fef5aeb18a2ef6de939` | 48 |
| `image-only-20261004/evaluation/predictions.jsonl` | `a672a0bc2558cd0d2599101049ebc93d7036b80bc03caa6071b06028b3c8655b` | 80 |
| `image-only-20261004/data/references.jsonl` | `c0c2981823913dba5ed1c48618bf32dd7038516aa2b50cc0e4e639d64b1bd603` | 16 references |

There are 320 output records across these four evaluations, including repeated baselines and post-hoc policies. They represent 16 source images, not 320 independent samples. All arms in an evaluation must cover all references, have matching family IDs, and agree on the initial parser string. Each prediction file is evaluated separately; never hide duplicate outputs by merging favorable arms.

## Metric integrity and readiness

Pin OmniDocBench to `f133a71e9e91c3621c7ce8994200a7b394a06eb3`. Verify every Python file in its CDM subtree against the pinned Git tree before importing it under an isolated namespace. Download/extraction and Python/environment setup may change runtime availability; the upstream algorithm, token preprocessing, symbol matching and sources are untouched.

Reset NumPy's random seed to 20261004 for every metric call. Use exact frozen initial/final/reference strings with the upstream core callable's internal preprocessing only. No additional formula normalizer or post-hoc edit is introduced. Retain existing normalized/contract/audit outputs as distinct saved conditions.

Run four readiness controls before each evaluation: identical and equivalent-bracing `x^2+1` must score 1; `x^3+1` must receive partial credit; malformed `\frac{` must score zero with a successfully rendered reference. Then self-score every reference and require F1=1 and a nonzero reference token count. If any reference fails, stop that evaluation with a failed receipt and retain the diagnostic calls; do not produce an aggregate that silently interprets unavailable rendering as a zero prediction score. Require a stable reference-token count in every subsequent pair.

Cache identical exact reference/prediction pairs within each evaluation, resetting the seed for uncached pairs. Cache only evaluation work, not model calls. Record every unique metric call, token matches, scores and seconds. Failed/empty predictions retain zero and stay in the denominator; `pred_tokens=0` is reported as unrenderable or empty, not assigned an unsupported precise failure cause.

The working isolated runtime uses TeX Live 2025/Debian pdfTeX 1.40.28, ImageMagick 7.1.2-18 Q16 and system Ghostscript 10.06.0; Python dependencies are pinned in `requirements-cdm.txt`. Debian/Ubuntu packages were downloaded and extracted in the project cache, not installed globally. Distribution archives/font-map receipts preserve hashes. The system Ghostscript profile denies this external volume; metric scratch files use a permitted temporary directory, without changing that profile or the upstream code. Every evaluation records runtime, source and data hashes. No unsupported CJK-font claim is made.

## Reporting and decision

Report case and source mean F1, paired deltas, CDM-matching/nonmatching initial counts, matching regressions, nonmatching full fixes, partial improvements/degradations, final matches, byte changes and failed/empty initial/final predictions. CDM=1 means agreement under this metric and frozen annotation; it is not proof of literal image fidelity or semantic correctness. Retain source/reference observational audits separately.

Compare the main native-parser repair arms and image-only outputs first. Existing adapter/contract and agreement policies are post-hoc development controls. The two image-only pixel settings had identical actual grids and outputs, so CDM cannot make them a genuine resolution experiment. No new accept/reject policy is selected for a held-out claim from these results.

Inspect changed cases with score increases, particularly the unit correction previously obscured by the raster proxy, against the source and saved reference; also inspect full-match regressions. Archive all scores, raw frozen inputs, readiness controls, source/runtime/data receipts, code snapshots and checksums in a portable package. Fresh replay must reproduce every case and summary under the pinned runtime, with any stochastic/runtime discrepancy disclosed.

Use these findings to decide what proposal capability/supervision to investigate next. Official metric readiness does not establish novelty, a learned verifier, parser transfer or training gains. A paper claim still requires untouched evaluation, identified independent parsers, relevant baselines, and matched training/inference budgets.
