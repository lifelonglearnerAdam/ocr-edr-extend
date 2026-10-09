# Separately trained table diagnosis and fixed-refiner experiment

User authorizes continued autonomous research, ablations and prompt publication; keep the full five-direction goal active. This is a bounded component stage within the existing integrated design, not a new completed method.

1. Test strict three-field diagnosis conversion/parsing, no answer-text leakage, stale/source binding, alternate legal-region displacement. Implement `src/ocr_edr/table_diagnosis.py`; extend existing table messages/adaptation with an optional locally constructed, schema-checked diagnosis. Unhinted existing behavior must remain byte-identical.
2. Implement `scripts/train_table_diagnosis.py` on the existing fixed-schedule kernel and a dedicated YAML; train-only admission127/406, fixed381steps, no target/reference dev reads. Freeze source/config/model/prompt hashes and budget before GPU work. Reuse independent checkpoint reload checker.
3. Implement diagnosis inference (103controlled+32native) without reference access. Freeze all135calls before offline gold scoring or oracle packet preparation.
4. Prepare hash-bound learned/displaced/oracle packets. Oracle only has projected class/location, cannot disclose replacement text/span. Run fixed all refiner341newcalls; record all failures and full source coverage; reuse original135unhintedcalls after identity checks.
5. Freeze outputs then offline score separate diagnosis and downstream tables. Preserve native/reference ambiguity and compute paired document contrasts. Public archives sanitize text/images/paths; report costs and exploration limits.
6. Reviewer audits actual inputs/model/receipts/full denominators, then full tests, formatting, browser report figures, push and verified online readback. The auto observer should only publish reviewed site changes.
