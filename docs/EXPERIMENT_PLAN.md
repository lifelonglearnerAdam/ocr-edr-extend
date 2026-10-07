# Initial experiments through October 19

The first milestone is a formula/table correction pilot and implementation plan, following the supplied meeting notes. The CPU foundation implements state transitions, renderer interfaces, region validation and delegation to the official evaluator. It has no trained repair policy or measured repair gains yet.

1. Confirm available releases/licenses for OCR-EDR, DocEDR/OCRErrBench and Jev; pin OmniDocBench and its CDM/TEDS environment. Saved literature notes are inputs to this verification.
2. Prepare independently sourced train/dev regions and a frozen benchmark manifest. Start with two parsers, then extend to the four-system matrix. Report modality, parser and Good/Bad branches separately.
3. Connect real LaTeX and HTML/Markdown table renderers, the Qwen2-VL-2B policy and a frozen visual judge. Check rendering-equivalent outputs, failures and preservation before batch inference.
4. Compare unchanged outputs, one-pass repair, multiple turns without updated rendering and the full loop at a common token/turn budget. Report full-set metrics, Bad-subset gains, Good-subset regressions and cost.
5. Add teacher trajectories and curriculum SFT using train/dev data; compare SFT and GRPO at a fixed student and dataset. Existing training YAML is a proposed recipe, not an implemented trainer.
6. Evaluate Jev directly and as a calibrated cascade, including escalation, false acceptance, quality and cost. Choose the gate on development data before frozen test evaluation.

The brief's initial success criterion is improvement on at least two OCR systems, with a student of at most 3B parameters and reproducible costs. Estimate uncertainty at the parent-page level rather than treating same-page crops as independent observations. Validate novelty against the base paper's rendering and GRPO contributions.

## October 6 execution update

The dated initial plan above is retained as context. The unified five-direction design and factorized comparisons are now in `docs/research/INTEGRATED_RESEARCH_20261006.md` and `configs/research/five_direction_ablation.yaml`. Two direct-target LoRA arms complete 192 steps, with fixed terminal checkpoints and no dev optimizer use; all 96 dev cases per model are generated and officially core-CDM scored. Controlled errors improve, native errors do not, and one extra-superscript harm is visually confirmed. The results require independent acceptance/source supervision rather than immediately expanding RL.

Official Jev documentation now establishes a text-only hosted interface. Treat Jev as a calibrated router on provenance-tagged evidence, with an independent visual module and same-evidence rule comparator; charge all evidence-generation costs. Teacher access, trajectory targets, verified acceptance/calibration, table training, a second identified parser and untouched evaluation remain pending. These tasks remain part of the full project, not omitted scope.
