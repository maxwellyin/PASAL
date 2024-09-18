# `v4` Experiment Guide

This directory contains the original experiment drivers used during development of PASAL. The code is organized as standalone research scripts rather than a unified training package.

The repository has now been partially modernized: most script files are thin CLI entry points, while shared training and pseudo-labeling logic lives in `utils/`.

## Recommended Entry Points

- `prompt-tuning.py`
  - Source-domain prompt tuning.
  - Trains only the learnable prompt embedding while freezing the T5 backbone.
  - Supports CLI flags such as `--source-domain`, `--data-root`, and `--cuda-visible-devices`.

- `model-sf-after-p-sf.py`
  - Main PASAL-style target adaptation pipeline.
  - Builds pseudo question-answer pairs on the target domain and performs self-training.
  - Supports CLI flags such as `--target-domain`, `--checkpoint-path`, `--train-subset`, and `--cuda-visible-devices`.

- `eva.py`
  - Evaluates the prompt-tuned model on the default target-domain test split.
  - Supports `--target-domain` and `--checkpoint-path`.

- `eva_cmd.py --target-domain <domain>`
  - Thin compatibility wrapper over `eva.py`.

## Example Commands

```bash
python prompt-tuning.py --source-domain SQuAD --cuda-visible-devices 0
python model-sf-after-p-sf.py --target-domain NaturalQuestionsShort --train-subset 10000
python eva.py --target-domain NewsQA
```

## Other Experimental Variants

- `finetuning.py`
  - Standard seq2seq fine-tuning baseline without prompt tuning.

- `sf.py`
  - Source-free adaptation using an external extractive QA model for pseudo labels.

- `sf_prompt.py`
  - Prompt-only adaptation on pseudo-labeled target data.

- `sf_with_qg.py`
  - Combines question generation with an external extractive QA model.

- `msf.py`, `m-train-on-source.py`, `psf-after-msf.py`, `psf-msf-multiple.py`, `sf_multiple.py`, `sf_simple.py`
  - Additional variants explored during experimentation.

- `domain_train.py`, `eva2.py`
  - Domain-adversarial branch based on `PromptT5DANN`.

## Utilities

- `utils/model.py`
  - Customized T5 implementation with learnable prompt embeddings and an optional domain classifier.

- `utils/runtime.py`
  - Shared path resolution, checkpoint naming, and runtime helpers.

- `utils/experiments.py`
  - Shared dataset preprocessing, parameter-freezing strategies, trainer construction, and self-training helpers.

- `utils/pseudo_labeling.py`
  - Shared pseudo-label generation strategies for prompt QA, extractive QA, and question generation.

- `utils/util.py`
  - Shared constants, preprocessing helpers, prompt padding, and EM/F1 metrics.

- `utils/qg.py`
  - Utilities for question generation and ROUGE-based evaluation.

- `utils/ae.py`
  - Utilities for answer extraction and EM/F1 scoring.

- `utils/tokenizer.py`
  - Helper script for saving a tokenizer with the `<hl>` token used by question generation.

## Notes

- The main entry-point scripts are now parameterized through `argparse`.
- Many historical experiment files are now thin wrappers over shared utilities, so script names remain stable without duplicating training logic.
- Dataset locations are expected under `../../largeQA/data/`.
- Several legacy scripts still assume a local multi-GPU environment and preserve the original hard-coded setup.
- The repository preserves the original research workflow, so consistency with the paper took priority over framework-level refactoring.
