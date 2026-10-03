# Project scope

The source is `optimization/ocr-edr-team`, with research notes from `optimization/research/ocr-edr-extend`. The GitHub repository is `lifelonglearnerAdam/ocr-edr-extend`.

This project develops a small-model extension of OCR-EDR for formula/table diagnosis and repair across OCR systems. Its hypotheses concern updated visual evidence, Agentic RL, teacher distillation and Jev cascades that reduce judge cost while preserving correct inputs.

The separate `noteresearch` project covers MonkeyOCR Note tables and formulas. Its diagnostic records, baseline scores, reviews and training decisions belong to that work. MonkeyOCR may appear here as one benchmark parser alongside PaddleOCR-VL, MinerU and DeepSeek-OCR.

This project requires its own benchmark manifest, parser predictions, frozen evaluation protocol and run metadata, plus independently sourced train/dev data. Initial code checks use synthetic fixtures and make no OCR-performance claim.

The base paper already uses rendering feedback and GRPO according to the supplied research notes. Research claims must establish what the formula/table extension, smaller model, judge cascade or transfer protocol contributes beyond that baseline.
