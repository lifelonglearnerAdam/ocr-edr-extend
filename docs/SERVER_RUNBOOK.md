# Standalone project server setup

The local checkout is `~/d/recovered_final/optimization/ocr-edr-team`. The dedicated campus-server checkout is `/data/yxliu/projects/ocr-edr-extend-paper`; its CPU environment is `/data/yxliu/envs/ocr-edr-extend-core`.

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
