# Shared research dashboard implementation plan

> Execute inline; use the existing read-only reviewer for evidence/provenance and UI checks. User explicitly requests timely GitHub publication and a shareable, updatable HTML report.

**Goal:** Publish a readable single-file Chinese research report combining verified project experiments and the two supplied senior PDFs, with repeatable data refresh and an online URL.

**Architecture:** Maintain a public, sanitized JSON snapshot, a standalone HTML template, and a standard-library builder. A separate updater derives quantitative sections from completed hash-bound experiment receipts. The site polls published JSON for newer snapshots; local experiment monitoring updates and publishes only meaningful changes, never raw model calls or credentials.

**Inputs:** `qwen复现.pdf` (3 pages) and `相关工作.pdf` (2 pages), read and rendered locally; the full verified NF4 run, frozen statistics, formula SFT/second-parser, native table, teacher pilot and prompt controls. PDFs contain a personal Windows path, so publish only vetted figure crops and verified transcriptions, with original PDF SHA256/page provenance.

**Output paths:** `docs/site/index.html`, `docs/site/research-data.json`, `docs/site/assets/`; `scripts/build_research_dashboard.py`, `scripts/update_research_dashboard.py`; scientific evidence remains in `experiments/artifacts/`.

## Acceptance criteria

- Clear overview explains research question, five-direction pipeline, division of contributions and what the outcomes mean.
- Peer results are explicitly labeled supplied-report evidence, not independently reproduced. Formula verifier percentages have n=1800, condition and metric definitions; unspecified revisions/hardware/splits remain unknown.
- Full NF4 negative results and uncertainty are visible, with all103 cases/32 documents, both381-step arms,12/32 regressions, costs and document-level intervals. Never hide negatives behind training loss.
- Include earlier formula/native-parser/teacher evidence with boundaries, pending judge/RL/transfer and actionable next comparisons.
- Responsive, accessible navigation, expandable detail, clear source links, update time and download. No external font/script/CDN dependency; HTML works offline with embedded charts/images/data.
- Generated data safely escapes HTML/script injection and rejects nonfinite/incorrect/partial metrics. Public outputs contain no raw HTML, private paths, training images or credentials.
- Desktop/mobile browser QA covers tabs, filter, download, source links and layout; public live URL is read back after publishing.
- Every scientific update is reflected in GitHub and the hosted page using the same generated snapshot; timestamp does not imply directly streaming GPU state. Preserve the main branch/other collaborator changes and do not merge PR4.

## Steps

1. [x] Read all5 supplied PDF pages and extract source-supported values; identify safe crops.
2. [x] Freeze public peer provenance and current result snapshot; publish the completed negative NF4 evidence.
3. [x] Test secure serialization and evidence-bound update behavior, then implement the generator and responsive template.
4. [x] Build, review quantitative content and inspect actual desktop/mobile browser render.
5. [x] Push feature work and publish the reviewed static site using an official Pages deployment workflow from the authorized research branch; enable/read back the public page under the user's publication request.
6. [x] Connect experiment snapshot refresh and repeat publication; include prompt-ablation output only after actual completion. Keep the five-direction research goal active.
