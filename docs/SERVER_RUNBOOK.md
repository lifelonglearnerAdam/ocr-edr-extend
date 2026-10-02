# Server runbook

Campus network access is required for the lab SSH servers. Use your account's SSH configuration or authenticate interactively; keep passwords and private keys outside this repository. The prepared checkout uses `/data/yxliu/projects/ocr-edr-extend`.

## CPU checks and Note import

The existing evaluation environment provides Python 3.10 and PyYAML. System Python 3.8 is too old for this package.

```bash
cd /data/yxliu/projects/ocr-edr-extend
PY=/data/yxliu/envs/omnidocbench-v1.6/bin/python
SOURCE=/data/yxliu/datasets/ocr-edr-note-eval-20261002

"$PY" -m unittest discover -s tests -v
"$PY" scripts/demo_loop.py
"$PY" scripts/prepare_note_eval.py \
  --source-root "$SOURCE" \
  --output data/raw/note_eval
```

The evaluation snapshot is copied from the verified local mirror into the account's own dataset directory; the original shared files under `/data/hzhang` are not readable by this account. `SNAPSHOT_SHA256.json` records each copied input's hash and its test-only status. New artifacts are written into this checkout's ignored data and experiment directories. The importer's default `--remote-root /data/hzhang` rebases crop paths from the original manifest onto the supplied source root.

## Official evaluation

The account has an OmniDocBench v1.6 checkout at `/data/yxliu/projects/OmniDocBench`. Its observed revision at setup was `8a575f05466d2cc6c34a5a65b7ce5fb958fc184c`; it may differ from the evaluator that produced the stored Note baseline. Pin a common evaluator and environment for before/after comparisons.

```bash
cd /data/yxliu/projects/ocr-edr-extend
PY=/data/yxliu/envs/omnidocbench-v1.6/bin/python
SOURCE=/data/yxliu/datasets/ocr-edr-note-eval-20261002

"$PY" scripts/eval_omnidocbench.py \
  --config configs/eval/omnidocbench_formula_table.yaml \
  --official-repo /data/yxliu/projects/OmniDocBench \
  --python "$PY" \
  --gt "$SOURCE/OmniDocBench_note/OmniDocBench_note.json" \
  --pred-dir "$SOURCE/baselines/monkeyocrv2_b_note/markdowns" \
  --dry-run
```

Remove `--dry-run` to execute after loading the working CDM environment and verifying `pdflatex`, `kpsewhich`, ImageMagick and Ghostscript. Each invocation creates a new run directory containing native configuration, annotation/configuration hashes, evaluator revision and exit code. The official evaluator's relative output directory is inside that run directory. Supply a distinct repaired Markdown directory with `--pred-dir` for the after run.

```bash
"$PY" scripts/compare_official_results.py \
  --before experiments/runs/official/BEFORE_RUN/result \
  --after experiments/runs/official/AFTER_RUN/result \
  --output experiments/tables/paired_result.json
```

## Model and renderer integration

The first target remains `Qwen/Qwen2-VL-2B-Instruct`. A cache directory exists at `/data/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct`; check the complete snapshot and model dependencies before loading. No model weights are copied into Git.

Implement `Policy.act` and `Judge.assess` using `Observation`: source image, current markup and current rendering only. The example demo judge is scripted and must be replaced for real experiments. `CommandRenderer` accepts an executable argument list containing `{input}`, `{output}` and optionally `{modality}`; it requires a fresh PNG. Renderer and judge failures return the initial prediction with an explicit rollback status.

There are eight RTX 3090 GPUs with 24 GiB each. Check `nvidia-smi` immediately before reserving a GPU; free memory observed during setup does not establish future availability. Start with inference and a small resource measurement, then decide whether distributed training is necessary. Independent train/dev data and teacher API quota are still prerequisites for the planned training work.
