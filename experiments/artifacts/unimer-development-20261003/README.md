# Independent formula development evidence, October 3, 2026

This package preserves a bounded development study on 16 published UniMER images (8 SPE, 8 SCE). It contains fixed model outputs, evaluator results, provenance, selection/exclusion logs, and reference-free inference inputs. It does not contain the benchmark source images or model weights.

These inspected images are permanently development data and must be excluded from later frozen test claims. Selection required MathText-supported annotations of at most 200 characters before full TeX became available. This is neither a representative UniMER evaluation nor CDM/VisFix. Native errors and controlled one-symbol perturbations are separate conditions on the same sources.

- `data/`: 32 controlled cases, all examined eligibility records, archive-to-annotation mapping, offline labels.
- `native-data/`: all 16 native predictions from the independent Nougat-LaTeX recognizer; labels remain separate.
- `nougat-native/`: pinned model and custom processor provenance, prediction strings, runtime and length-cap status.
- `inference_source/`: exact clean-commit driver/proposer bytes for controlled/native repair, verified against each inference receipt.
- `reproduction.json`: all case/summary records from both score replays and all four post-hoc controls reproduce exactly.
- `controlled-tex/`, `native-tex/`: fixed Qwen2-VL-2B outputs and all-case exact-raster/string proxy evaluation.
- `*-adapter-audit/`: post-hoc replay of the same outputs through whole-environment normalization and strict output-contract controls. Decisions use outputs only; labels are opened afterward. No new model calls.
- `*-agreement-audit/`: post-hoc fixed agreement gates over the saved outputs. All required proposal costs are charged, including rejected calls. Agreement is not a correctness certificate.

Replaying proxy scores does not require source images:

```bash
.venv-pilot/bin/python scripts/evaluate_formula_pilot.py \
  --run experiments/artifacts/unimer-development-20261003/controlled-tex \
  --references experiments/artifacts/unimer-development-20261003/data/references.jsonl \
  --renderer tectonic
.venv-pilot/bin/python scripts/evaluate_formula_pilot.py \
  --run experiments/artifacts/unimer-development-20261003/native-tex \
  --references experiments/artifacts/unimer-development-20261003/native-data/references.jsonl \
  --renderer tectonic
```

Reconstruct inference images from the official archive at dataset revision `2343ddd963290469da36ca83e3a56c66e068add9` with SHA-256 `9bf370b8cac868fee84835f40dec26c477430611253e3feb681356a8149a3a90`. The source IDs and archive paths are listed in `data/sources.jsonl`. Numeric image filenames index annotation rows; ZIP enumeration order is not the mapping. Run `scripts/make_unimer_development.py` into a new temporary directory and compare every source image hash and JSONL byte hash. The native-data image paths use the same source images.

The frozen reference labels have not been changed after inspection. `annotation_audit.jsonl` lists visual concerns separately; these concerns qualify interpretation of the proxy counts. See [the protocol](../../../docs/research/REAL_FORMULA_PROTOCOL.md) and [the findings](../../../docs/research/RESULTS_20261003.md).
