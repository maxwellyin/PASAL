# `scripts` Guide

This directory contains the public-facing runnable entry points for the PASAL repository.

## Main Entry Points

- `prompt_tuning.py`
  - Prompt-tune the source-domain model.

- `adapt_pasal.py`
  - Main PASAL adaptation pipeline using generated QA pairs.

- `eval.py`
  - Evaluate a PASAL checkpoint on a target-domain test split.

## Additional Variants

- `finetune_baseline.py`
  - Standard seq2seq fine-tuning baseline without prompt tuning.

- `adapt_sf.py`
  - Source-free adaptation using an external extractive QA model for pseudo labels.

- `adapt_prompt_sf.py`
  - Prompt-only source-free adaptation.

- `adapt_sf_with_qg.py`
  - Question generation plus extractive QA pseudo-labeling.

- `train_zero_prompt_source.py`
  - Source training from a zeroed, frozen prompt initialization.

- `train_dann.py`
  - Domain-adversarial training variant.

- `eval_dann.py`
  - Evaluate a DANN checkpoint.

## Notes

- Shared implementation lives in [`pasal/`](../pasal/).
- Legacy historical experiment files remain under [`legacy/v4/`](../legacy/v4/) but are no longer the primary project interface.
