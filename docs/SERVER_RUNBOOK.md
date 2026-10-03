# Standalone project server setup

## Platform split

| Role | Host | Environment | Use for |
| --- | --- | --- | --- |
| Documented Windows workstation | `D:\recovered_final\optimization\ocr-edr-team` | System Python 3.10+ **or** a **Windows** venv (`\.venv\Scripts\python.exe`) | Docs, light scripts, PR review, `unittest` without Torch |
| Ubuntu / GPU research host (GPT-managed) | `~/d/recovered_final/optimization/ocr-edr-team` and campus paths below | POSIX venv (`bin/python`) | Pilot inference, TeX/table rendering, official evaluator, training |

**Do not reuse a POSIX venv on Windows.** Linux-created `.venv/` directories use `bin/` + `lib/` (no `Scripts\python.exe`). A Windows checkout that sees `bin/` instead of `Scripts\` must create a native venv; the Linux venv is not portable. Conversely, do not point POSIX tooling at a Windows venv.

Clone-specific caches (`data/raw/model-cache`, `experiments/runs`) stay machine-local and Git-ignored. Share **artifacts** under `experiments/artifacts/` with hashes, not live venvs or model weights.

## Ubuntu / campus GPU host

The local POSIX checkout is `~/d/recovered_final/optimization/ocr-edr-team`. The dedicated campus-server checkout is `/data/yxliu/projects/ocr-edr-extend-paper`; its CPU environment is `/data/yxliu/envs/ocr-edr-extend-core`.

```bash
cd /data/yxliu/projects/ocr-edr-extend-paper
PY=/data/yxliu/envs/ocr-edr-extend-core/bin/python
"$PY" -m unittest discover -s tests -v
"$PY" scripts/demo_loop.py
"$PY" scripts/validate_manifest.py --manifest examples/synthetic/regions.jsonl
```

Raw project inputs are staged separately in `/data/yxliu/datasets/ocr-edr-extend-paper`. Dataset acquisition and real-model integration remain pending; synthetic fixtures only verify software.

The existing official OmniDocBench checkout/environment can provide dependencies for a pinned evaluation while this project's manifests and outputs remain separate:

```bash
"$PY" scripts/eval_omnidocbench.py \
  --config configs/eval/omnidocbench_formula_table.yaml \
  --official-repo /data/yxliu/projects/OmniDocBench \
  --python /data/yxliu/envs/omnidocbench-v1.6/bin/python \
  --baseline paddleocr_vl \
  --gt /path/to/this-projects/OmniDocBench.json \
  --pred-dir /path/to/this-projects/paddleocr_predictions \
  --dry-run
```

Supply this project's actual annotation and prediction paths. Remove `--dry-run` after configuring official CDM dependencies: TeX, ImageMagick and Ghostscript. New run directories record configuration/annotation hashes, evaluator revision and exit status.

The student target remains `Qwen/Qwen2-VL-2B-Instruct`. CPU scripts launch no training jobs. Check GPU/process availability and a small memory profile before reserving inference or GRPO resources. Keep SSH credentials/private keys outside Git; campus access is required.

## Windows workstation checks

```powershell
cd D:\recovered_final\optimization\ocr-edr-team
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\demo_loop.py
```

- Use `python.exe` / `py` only after confirming it is the Windows venv, not a WSL or conda env that lacks `Scripts\`.
- Prefer `git -c http.proxy= -c https.proxy= <cmd>` when the local HTTP proxy is down; keep secrets and campus keys out of Git.
- `requirements-pilot.txt` (Torch + Qwen) is **optional on Windows**. Full pilot inference belongs on the Ubuntu/GPU host; CI remains the portable quality gate (pytest/unittest on 3.10 and 3.12).
- When a run is produced on Linux and copied here, keep `run.json` `source_revision` / file hashes intact and re-evaluate offline with `scripts/evaluate_formula_pilot.py` before editing artifacts.
