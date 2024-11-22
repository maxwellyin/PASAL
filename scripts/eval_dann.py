import argparse
from pathlib import Path

from datasets import DatasetDict, load_from_disk
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Tokenizer

from pasal.model import PromptT5DANN
from pasal.runtime import DEFAULT_DATA_ROOT, REPO_ROOT, model_slug, resolve_data_dir
from pasal.util import (
    MODEL_CHECKPOINT,
    PROMPT_LENGTH,
    SOURCE_DOMAIN,
    T5_MAX_INPUT_LENGTH,
    TARGET_DOMAIN,
    compute_metrics,
    pad_prompt_length,
    preprocess_validation_examples,
    set_tokenizer_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a DANN PASAL checkpoint on a target-domain test split.")
    parser.add_argument("--source-domain", default=SOURCE_DOMAIN, help="Source domain used during training.")
    parser.add_argument("--target-domain", default=TARGET_DOMAIN, help="Target domain to evaluate on.")
    parser.add_argument("--model-checkpoint", default=MODEL_CHECKPOINT, help="Tokenizer/model family used for preprocessing.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Directory containing processed datasets.")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Checkpoint directory to evaluate. Defaults to the DANN model.")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device evaluation batch size.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_tokenizer_checkpoint(args.model_checkpoint)

    validation_path = resolve_data_dir(args.data_root, args.target_domain, "test")
    raw_validation = load_from_disk(str(validation_path))

    pmaxlen = T5_MAX_INPUT_LENGTH - PROMPT_LENGTH
    preprocess_fn = lambda batch: pad_prompt_length(preprocess_validation_examples(batch, max_length=pmaxlen))
    validation_dataset = raw_validation.map(preprocess_fn, batched=True, remove_columns=raw_validation.column_names)
    tokenized_datasets = DatasetDict({"validation": validation_dataset})

    model_name = model_slug(args.model_checkpoint)
    checkpoint_path = args.checkpoint_path or (REPO_ROOT / "checkpoints" / f"prompt-{model_name}-{args.source_domain}" / "dann")
    model = PromptT5DANN.from_pretrained(str(checkpoint_path))

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(REPO_ROOT / "out"),
        evaluation_strategy="epoch",
        learning_rate=5.6e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=False,
    )

    tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print(trainer.evaluate())
    print("MODEL_CHECKPOINT:", args.model_checkpoint)
    print("SOURCE_DOMAIN:", args.source_domain)
    print("TARGET_DOMAIN:", args.target_domain)
    print(f"PROMPT_LENGTH: {PROMPT_LENGTH}")
    print(f"CHECKPOINT_PATH: {checkpoint_path}")


if __name__ == "__main__":
    main()
