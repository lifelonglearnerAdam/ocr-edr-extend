# Independent formula supervision preparation

Protocol date: 2026-10-06. This continues the proposal-supervision prerequisite in H1/H4. It prepares a small train/dev dataset and native-parser inputs; it does not establish SFT, RL, distillation or benchmark improvement.

## Source and provenance

Use only `UniMER-1M.zip` from `wanderkid/UniMER_Dataset`, revision `2343ddd963290469da36ca83e3a56c66e068add9`. Verify the complete 1,983,641,138-byte archive against its published LFS SHA-256 `c2563aba157f470a9d7d2084aac58fb32607c58ce0992eb4a89d767d23be44dd` before extraction. The publisher labels UniMER-1M as training and UniMER-Test as evaluation; the dataset card declares Apache-2.0 and separately explains that HME100K material requires manual acquisition. Do not infer that every component has unrestricted rights from a single card field.

The available archive contains 986,122 image members and a 1,061,791-line annotation file. Those counts differ from the card's nominal total. Record available members, nonblank annotations and valid image-to-label mapping rather than silently assuming complete coverage. Numeric image stems index the author's annotation list.

Sources read on 2026-10-06:

- <https://huggingface.co/datasets/wanderkid/UniMER_Dataset/blob/2343ddd963290469da36ca83e3a56c66e068add9/README.md>
- <https://huggingface.co/api/datasets/wanderkid/UniMER_Dataset/tree/2343ddd963290469da36ca83e3a56c66e068add9?recursive=true>
- <https://github.com/opendatalab/UniMERNet/tree/5a2c80d96b1d2dba447ff18d873e5fb73ba03c35>

## Fixed selection and split

1. Order available training image members by SHA-256 of `20261006-supervision:<archive_path>` before any new model output is generated.
2. Require a nonempty released formula of 8–200 characters and one defined single-symbol perturbation: one digit in a braced exponent/subscript, otherwise a plus/minus operator. This is a deliberately restricted capability pilot, not representative sampling.
3. Block all 23,757 available UniMER-Test images and their indexed references, as well as previously inspected project source images/formulas. Never turn uninspected benchmark items into training examples.
4. Check file hashes, decoded RGB pixel hashes and conservative formula keys (whitespace removed only). Also reject a candidate when its 64-bit image difference hash is within Hamming distance 2 of a held-out/selected image with an aspect ratio within 2%. This is a screening proxy with both false positives and false negatives, not proof of semantic or document-level independence.
5. Require released and corrupted formulas to render under the existing pinned Tectonic adapter and have different exact raster signatures. Log every exclusion; never select on model quality.
6. Assign entire source families by a deterministic formula-key hash: modulo 5 equals zero goes to dev, otherwise train. Stop at 128 train and 32 dev sources. Do not reuse formula keys, controlled-input keys, exact or proxy-near-duplicate images across families or splits.

Original source-document/page identifiers are not provided by this archive. Archive image identity and formula/image grouping are available; original-document independence and pretrained-model exposure remain unverified. Record that limitation explicitly.

## Supervision and inference boundary

For each selected source create one preservation example and one controlled repair example. The preservation target copies the candidate exactly; the repair target is the released reference. These are annotation-derived direct-answer targets, not teacher-generated or native-error trajectories. Train/dev references and variant labels stay in offline files; model inference inputs carry only generic IDs, image paths/hashes and the candidate string.

Freeze image-only inputs before running the existing pinned Nougat-LaTeX recognizer. Retain every native response and cap/failure record without selecting on error severity. Native outputs can later supply a separate supervision arm, but annotation disagreements require source inspection. Hold the dev split out of optimizer updates; the already inspected October 3/4 samples remain diagnostic history rather than fresh evaluation.

## Verification and limitations

Verify archive and manifest hashes, quotas, pair counts, source-to-label indexing, no train/dev overlap, full benchmark blocking, image containment and reference-free inference schemas. Independently recheck written images and manifests after the preparation process. Keep raw images/annotations/model outputs in ignored local data/run directories. Write an aggregate receipt and a source-hash-matched preparation script for reproducibility.

No weights are changed in this preparation stage. A functioning SFT baseline, teacher quota, calibrated acceptance, native outputs from a second identified parser and matched-budget SFT/RL experiments remain later requirements. Exact raster differences and string targets are not CDM/VisFix or certificates of source correctness.
