# Local NF4 training readiness — 2026-10-07

Two predeclared train-source boundary examples completed optimizer steps on the
local 8 GiB RTX 5070 Laptop GPU, with an independent process verifying all 112
saved adapter tensors. This is bounded execution evidence, not an accuracy result
or completion of the two 381-step table experiments.

- `full-logits/` retains the first NF4 run's OOM at step zero, exact executed code,
  configuration and preflight. Full-sequence vocabulary cross-entropy requested an
  additional 1.53 GiB at the longest input.
- `projected-loss/` retains the second run, including **failed** final status. It
  completed two optimizer steps but the same-process reload preflight lacked the
  required 7 GiB free. Peak allocated/reserved memory was 3.7803/4.65625 GiB.
- `standalone-reload/` verifies the saved checkpoint in a fresh CUDA process, with
  112 tensors exactly matching and 2.2946 GiB peak allocated. Its receipt binds the
  original failed training receipt without rewriting it.
- `dependencies/` records fixed official source URLs, retrieval times and PyPI wheel
  SHA256 checks. PEFT/bitsandbytes were installed in a separate overlay; the existing
  PyTorch/Transformers environment was retained.
- `verification.json` records 140 passing full-environment tests, mathematical-loss
  and tiny-Qwen gradient/generation parity checks, cross-precision pairing rejection,
  formatting checks and read-only code review. Tests did not use the active GPU.
- Exact generation-time code snapshots are retained separately from later CLI
  improvements. Default BF16 remains a distinct profile with its 16 GiB guard.

The projected loss omits only vocabulary projections for ignored labels. All image
and text context still goes through the transformer, with the same supervised next
tokens and mean denominator. The independent full NF4 contrast retains all 127
training documents and same-precision base inference; its results are not filled in
early here. See [results](../../../docs/research/TABLE_NF4_READINESS_RESULTS_20261007.md)
and [full protocol](../../../docs/research/TABLE_NF4_SCREEN_PROTOCOL_20261007.md).

No input images, target HTML, checkpoint weights, credentials or actual account paths
are published in this archive. Host paths in receipts are replaced by logical roots;
original hashes remain recorded. `sha256.json` covers all public files in this folder.
