# Table NF4 analysis protocol verification

The original-document bootstrap and audit selection protocol was fixed before
NF4 model-development inference began. See the [protocol](../../../docs/research/TABLE_NF4_ANALYSIS_PROTOCOL_20261007.md).

- Fixed comparisons: base vs unchanged, both SFT arms vs base, and all vs no explicit preservation.
- Original PMC documents are resampled as pairs; variants within each document are averaged first.
- Intervals are descriptive, single-seed development summaries, not confirmatory or risk certificates.
- Proxy utility weights 1/2/4 and complete inference/render cost denominators are fixed.
- Four focused tests include hand-computed unequal-document weighting, pairing/score/cost rejection,
  order-independent replay and a complete 32-document/103-case synthetic CLI fixture.
- Full environment: 144 tests passed with no optional skips. Core environment: 144 discovered,
  47 optional skips. Independent read-only review found no blocking statistical issues.

No model scores, synthetic values presented as real results, source images, HTML or weights
are published in this protocol-only receipt. Live full training results remain pending separately.
