# Diagnosis SFT recovery evidence, October9

Original optimizer logged6steps/24exposures, then no progress for50+minutes despite a live CPU-spinning process and idle GPU. System journal records12suspend/resume operations. This supports an external-interruption explanation; exact CUDA failure mechanism is unproven because stack attachment was unavailable under unchanged OS policy.

Only owned training/pipeline were stopped. Original training receipt/log/preflight/schedule remain byte-identical; the old waiter's status file was naturally updated after first observation and its final saved hash is retained. No optimizer checkpoint existed. The retry is a fresh full381-step fixed-protocol run, not a6-step continuation. Original24exposures and observed cost remain separate.

Matching base/config/targets/admission/runtime/token budgets, byte-identical preflight/schedule and exact first6loss/gradient checks are appended. The replay check is a progress receipt, not final quality evidence. Temporary idle:sleep inhibition protects only this authorized GPU task. No power settings, authentication, safety guards or other users'processes were changed.

[Recovery methods and limits](../../../docs/research/TABLE_DIAGNOSIS_RECOVERY_20261009.md). Class/location/refinement scores remain pending until every frozen stage and independent mechanical audit is complete.
