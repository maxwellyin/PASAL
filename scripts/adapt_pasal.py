import argparse
from pathlib import Path

from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Tokenizer, pipeline

from pasal.model import T5ForConditionalGeneration as PromptT5
from pasal.runtime import (
    DEFAULT_DATA_ROOT,
    REPO_ROOT,
    maybe_set_cuda_visible_devices,
    model_slug,
    pipeline_device,
    resolve_data_dir,
)
from pasal.util import (
    MODEL_CHECKPOINT,
    PROMPT_LENGTH,
    SEED,
    SOURCE_DOMAIN,
    T5_MAX_INPUT_LENGTH,
    TARGET_DOMAIN,
    compute_metrics,
    format_for_lmqg,
    pad_prompt_length,
    preprocess_training_examples,
    preprocess_validation_examples,
    set_tokenizer_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run PASAL target-domain self-training after prompt-based source-free adaptation.")
    parser.add_argument("--source-domain", default=SOURCE_DOMAIN, help="Source domain used to name checkpoints.")
    parser.add_argument("--target-domain", default=TARGET_DOMAIN, help="Target domain to adapt to.")
    parser.add_argument("--model-checkpoint", default=MODEL_CHECKPOINT, help="Backbone checkpoint used for tokenization.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Directory containing processed datasets.")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Initial checkpoint to continue training from.")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "checkpoints", help="Directory used to save checkpoints.")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device train and eval batch size.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of self-training epochs.")
    parser.add_argument("--learning-rate", type=float, default=5.6e-5, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--train-subset", type=int, default=10000, help="Optional limit on target training examples.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed used for dataset shuffling.")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES override.")
    parser.add_argument("--qg-model", default="lmqg/flan-t5-base-squad-qg", help="Question-generation checkpoint.")
    return parser.parse_args()


def build_preprocess_fns():
    pmaxlen = T5_MAX_INPUT_LENGTH - PROMPT_LENGTH
    train_fn = lambda batch: pad_prompt_length(preprocess_training_examples(batch, max_length=pmaxlen))
    validation_fn = lambda batch: pad_prompt_length(preprocess_validation_examples(batch, max_length=pmaxlen))
    return train_fn, validation_fn


def train_model(train_dataset, validation_dataset, model, model_checkpoint, output_dir, batch_size, epochs, learning_rate, weight_decay):
    train_fn, validation_fn = build_preprocess_fns()
    train_dataset = train_dataset.map(train_fn, batched=True, remove_columns=train_dataset.column_names)
    validation_dataset = validation_dataset.map(validation_fn, batched=True, remove_columns=validation_dataset.column_names)

    model.encoder.prompt_embedding.requires_grad = False

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        save_total_limit=3,
        num_train_epochs=epochs,
        predict_with_generate=True,
        report_to="tensorboard",
        fp16=False,
    )

    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    outcome = trainer.evaluate()
    print(outcome)
    return model, outcome


def main():
    args = parse_args()
    maybe_set_cuda_visible_devices(args.cuda_visible_devices)
    set_tokenizer_checkpoint(args.model_checkpoint)

    model_name = model_slug(args.model_checkpoint)
    checkpoint_path = args.checkpoint_path or (
        args.output_root / f"prompt-{model_name}-{args.source_domain}" / "prompt-sf" / args.target_domain / "final" / "0"
    )
    output_dir = args.output_root / f"prompt-{model_name}-{args.source_domain}" / "m-sf-after-p-sf"

    train_path = resolve_data_dir(args.data_root, args.target_domain, "train")
    test_path = resolve_data_dir(args.data_root, args.target_domain, "test")

    train_dataset = load_from_disk(str(train_path)).shuffle(args.seed)
    if args.train_subset is not None:
        train_dataset = train_dataset.select(range(min(args.train_subset, len(train_dataset))))
    test_dataset = load_from_disk(str(test_path))
    raw_datasets = DatasetDict({"train": train_dataset, "test": test_dataset})

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = PromptT5.from_pretrained(str(checkpoint_path))
    device = pipeline_device()

    question_generation = pipeline("text2text-generation", args.qg_model, device=device)
    question_answerer = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

    def create_pseudo_label(example):
        generated_question = question_generation(format_for_lmqg(example))[0]["generated_text"]
        qa_input = f"{tokenizer.pad_token * PROMPT_LENGTH}question: {generated_question} context: {example['context']}"
        predicted_answer = question_answerer(qa_input)[0]["generated_text"]
        answers = {"text": [predicted_answer], "answer_start": example["answers"]["answer_start"]}
        return {"question": generated_question, "answers": answers}

    pseudo_dataset = raw_datasets["train"].map(create_pseudo_label, load_from_cache_file=False)
    filtered_dataset = pseudo_dataset.filter(lambda example: example["answers"]["text"][0] in example["context"])
    print(f"pseudo_set length: {len(pseudo_dataset)}")
    print(f"filtered pseudo_set length: {len(filtered_dataset)}")

    model, outcome = train_model(
        train_dataset=filtered_dataset,
        validation_dataset=raw_datasets["test"],
        model=model,
        model_checkpoint=args.model_checkpoint,
        output_dir=output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    final_dir = output_dir / "0"
    model.save_pretrained(str(final_dir))
    print("MODEL_CHECKPOINT:", args.model_checkpoint)
    print("SOURCE_DOMAIN:", args.source_domain)
    print("TARGET_DOMAIN:", args.target_domain)
    print(f"PROMPT_LENGTH: {PROMPT_LENGTH}")
    print(f"INPUT_CHECKPOINT: {checkpoint_path}")
    print(f"OUTPUT_CHECKPOINT: {final_dir}")
    print(outcome)


if __name__ == "__main__":
    main()
