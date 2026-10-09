# Interactive teacher train pilot — 2026-10-07

The current assistant generated twelve real, training-only episodes before opening
this cohort's published references: nine edits and twelve terminal stops. All twelve
episodes replay; nine edited renders and twelve initial renders reproduce the frozen
PNG hashes. This is data collection and verification, not student distillation.

Official normalized TEDS full matches rise from 9/12 to 10/12, **while mean TEDS falls
from 0.995074353 to 0.972659189**. One source-visible row split disagrees with the
published merged-row DOM. The disagreement and three teacher abstentions remain in
the full denominator and review queue. No independent visual adjudication is claimed.

See the [results](../../../docs/research/INTERACTIVE_TEACHER_RESULTS_20261007.md) and
[generation protocol](../../../docs/research/INTERACTIVE_TEACHER_PILOT_20261007.md).

- `cohort-seal.json`, `generation-packet.json` and `events-sanitized.jsonl` bind all
  twelve sources, decisions, HTML states and render evidence using hashes. These are
  local integrity records, not third-party timestamps or proof of image viewing.
- `teacher-instructions.txt` is the exact common prompt. The recorded interactive
  model label is not an immutable model snapshot; teacher tokens/cost are unknown.
- `audit/` preserves every source's numerical result and mechanical checks.
  `paired-view-index.json` identifies the eight conservatively admitted pairs.
  Both private views have the same sources, initial HTML and teacher final answers;
  training token budgets remain unmatched and no student updates were performed.
- The eight paired documents comprise three no-edit cases, four formatting repairs
  and one numerical repair. This pilot does not establish broad repair capability.
- `native-pool/` records all 127 admitted train-source parser calls, zero failures,
  and 347.0468 s summed outer pipeline time. It uses the explicit original-frame
  SLANet coordinate override, with public PubTabNet pretraining-overlap limitations.
- `source/generation/` preserves exact generation-time code, including the original
  prepare script whose prompt was frozen by a separately recorded manual tool step.
  The current prepare CLI now freezes that prompt directly; a seal CLI packages the
  formerly manual manifest step. The original packet and snapshots were not rewritten.
- `source/audit/` and `source/native-pool/` match the corresponding executed hashes.
  `audit-history.json` records the first audit and its replacement after formatting
  and interface checks; metrics and paired target files are byte-identical.
- `verification.json` records 135 passing full-environment tests with no optional
  skips, formatting checks and read-only code review. It is not visual label review.

Raw source images, full HTML, per-case teacher decisions/evidence text, private
mapping, model weights and authentication material remain outside this public
archive. Local roots in receipts are replaced by logical placeholders; hashes of
original private files are retained. `sha256.json` covers every published file here.
