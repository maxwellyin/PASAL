import argparse
from pathlib import Path

from datasets import DatasetDict, load_from_disk
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Tokenizer

from pasal.model import T5ForConditionalGeneration as PromptT5
from pasal.runtime import DEFAULT_DATA_ROOT, REPO_ROOT, maybe_set_cuda_visible_devices, model_slug, resolve_data_dir
from pasal.util import (
    MODEL_CHECKPOINT,
    PROMPT_LENGTH,
    SOURCE_DOMAIN,
    T5_MAX_INPUT_LENGTH,
    compute_metrics,
    pad_prompt_length,
    preprocess_training_examples,
    preprocess_validation_examples,
    set_tokenizer_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prompt-tune the source-domain PASAL model.")
    parser.add_argument("--source-domain", default=SOURCE_DOMAIN, help="Source domain name.")
    parser.add_argument("--model-checkpoint", default=MODEL_CHECKPOINT, help="Hugging Face model checkpoint.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Directory containing processed datasets.")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "checkpoints", help="Directory used to save checkpoints.")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device train and eval batch size.")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES override.")
    return parser.parse_args()


def build_dataset_splits(train_path: Path, validation_path: Path):
    raw_train = load_from_disk(str(train_path))
    raw_validation = load_from_disk(str(validation_path))

    pmaxlen = T5_MAX_INPUT_LENGTH - PROMPT_LENGTH
    train_fn = lambda batch: pad_prompt_length(preprocess_training_examples(batch, max_length=pmaxlen))
    validation_fn = lambda batch: pad_prompt_length(preprocess_validation_examples(batch, max_length=pmaxlen))

    train_dataset = raw_train.map(train_fn, batched=True, remove_columns=raw_train.column_names)
    validation_dataset = raw_validation.map(validation_fn, batched=True, remove_columns=raw_validation.column_names)
    return DatasetDict({"train": train_dataset, "validation": validation_dataset})


def main():
    args = parse_args()
    maybe_set_cuda_visible_devices(args.cuda_visible_devices)
    set_tokenizer_checkpoint(args.model_checkpoint)

    source_train_path = resolve_data_dir(args.data_root, args.source_domain, "train")
    source_test_path = resolve_data_dir(args.data_root, args.source_domain, "test")
    model_name = model_slug(args.model_checkpoint)
    output_dir = args.output_root / f"prompt-{model_name}-{args.source_domain}" / "base"

    tokenized_datasets = build_dataset_splits(source_train_path, source_test_path)

    model = PromptT5.from_pretrained(args.model_checkpoint)
    for param in model.parameters():
        param.requires_grad = False
    model.encoder.prompt_embedding.requires_grad = True

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        report_to="tensorboard",
        fp16=False,
    )

    tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(trainer.evaluate())

    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    print("MODEL_CHECKPOINT:", args.model_checkpoint)
    print("SOURCE_DOMAIN:", args.source_domain)
    print(f"PROMPT_LENGTH: {PROMPT_LENGTH}")
    print(f"Saved model to: {final_dir}")


if __name__ == "__main__":
    main()
