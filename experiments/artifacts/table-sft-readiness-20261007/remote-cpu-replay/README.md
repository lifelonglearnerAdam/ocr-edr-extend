# Independent staged-runtime CPU replay

Supplement to the original readiness archive; its frozen files and initial hash
manifest are unchanged. This directory has its own supplementary manifest.

`cpu_mask_preflight.py` is the exact executed procedure. It uses the checked-out
project processor/mask/schedule functions on the admitted table records, with CUDA
hidden and no model or optimizer instantiated. All 406 processed-token counts,
assistant-mask boundaries and image-grid records match the earlier local receipt.
The resulting JSONL SHA256 is
`047f6862b51073a389ff8bd6dc83c3e6630cf8efe418e70a2dd57167f7d0ebb1`,
the same as the already archived `../token-preflight.jsonl`.

`run.json` records actual versions and both exposure totals; `model-verification.json`
records separate verification of the staged 12-file pinned model. No GPU forward,
backward, optimizer step or checkpoint was produced. This does not prove identical
future floating-point training across the two runtimes.

Run on an independently staged admitted dataset from its source checkout:

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 python3 cpu_mask_preflight.py \
  --project /path/to/ocr-edr-extend \
  --dataset /path/to/pubtabnet-four-roles-20261007 \
  --model /path/to/895c3a49bc3fa70a340399125c650a463535e71c \
  --output /path/to/new-preflight-output
```

Use the verified project interpreter/dependency overlay where needed. No host
address, username, password, raw target records or source images are included.
