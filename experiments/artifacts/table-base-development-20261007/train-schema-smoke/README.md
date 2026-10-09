# Train-only prompt-format feasibility check

The first admitted training source, p0001, supplies its three existing controlled
variants. Both conditions use the same base model/runtime and one reference-free
source image. One condition appends four literal JSON response shapes, with indices
and text explicitly marked as placeholders. No reference or target action is read.

Observed contract validity is 0/3 for the descriptive prompt and 3/3 with examples;
the latter outputs are all `stop`, so there are no edits. This is one independent
source document, not a development quality score or evidence of visual repair.
There are six calls and zero optimizer steps.

`run.json` and the exact driver preserve the recipe; `call-accounting.jsonl` omits
HTML, prompts containing source text and raw completions. Imported project code
was at the preparation revision `6d3f16cce0a4c86c1db32eb86798e13133ec6913`;
the older shared modules are also preserved in the readiness inference overlay.
Subsequent full-development literal-example inference has its own frozen protocol
and does not replace the original 103-call negative result.
