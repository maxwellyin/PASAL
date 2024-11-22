from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from datasets import DatasetDict, load_from_disk
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Tokenizer

from .util import (
    PROMPT_LENGTH,
    T5_MAX_INPUT_LENGTH,
    compute_metrics,
    pad_prompt_length,
    preprocess_training_examples,
    preprocess_validation_examples,
)


def build_preprocess_fn(use_prompt: bool, is_train: bool):
    preprocess = preprocess_training_examples if is_train else preprocess_validation_examples

    if not use_prompt:
        return preprocess

    pmaxlen = T5_MAX_INPUT_LENGTH - PROMPT_LENGTH
    return lambda batch: pad_prompt_length(preprocess(batch, max_length=pmaxlen))


def map_dataset(dataset, use_prompt: bool, is_train: bool):
    preprocess_fn = build_preprocess_fn(use_prompt=use_prompt, is_train=is_train)
    return dataset.map(preprocess_fn, batched=True, remove_columns=dataset.column_names)


def load_split_dataset(split_path: Path | str, use_prompt: bool, is_train: bool):
    dataset = load_from_disk(str(split_path))
    return map_dataset(dataset, use_prompt=use_prompt, is_train=is_train)


def load_train_validation_splits(train_path: Path | str, validation_path: Path | str, use_prompt: bool):
    train_dataset = load_split_dataset(train_path, use_prompt=use_prompt, is_train=True)
    validation_dataset = load_split_dataset(validation_path, use_prompt=use_prompt, is_train=False)
    return DatasetDict({"train": train_dataset, "validation": validation_dataset})


def load_raw_target_datasets(train_path: Path | str, test_path: Path | str, seed: int, train_subset: Optional[int] = None):
    train_dataset = load_from_disk(str(train_path)).shuffle(seed)
    if train_subset is not None:
        train_dataset = train_dataset.select(range(min(train_subset, len(train_dataset))))
    test_dataset = load_from_disk(str(test_path))
    return DatasetDict({"train": train_dataset, "test": test_dataset})


def set_trainable_parameters(model, strategy: str):
    if strategy == "prompt_only":
        for param in model.parameters():
            param.requires_grad = False
        model.encoder.prompt_embedding.requires_grad = True
        return

    if strategy == "freeze_prompt":
        for param in model.parameters():
            param.requires_grad = True
        model.encoder.prompt_embedding.requires_grad = False
        return

    if strategy == "frozen":
        for param in model.parameters():
            param.requires_grad = False
        return

    if strategy == "all":
        for param in model.parameters():
            param.requires_grad = True
        return

    raise ValueError(f"Unknown training strategy: {strategy}")


def zero_prompt_embedding(model):
    prompt_embedding = model.encoder.prompt_embedding.data
    model.encoder.prompt_embedding.data = prompt_embedding.new_zeros(prompt_embedding.shape)


def build_training_args(
    output_dir: Path | str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
):
    return Seq2SeqTrainingArguments(
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


def train_seq2seq_model(
    *,
    model,
    model_checkpoint: str,
    output_dir: Path | str,
    train_dataset,
    validation_dataset,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
):
    training_args = build_training_args(
        output_dir=output_dir,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
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
    return model, outcome


def self_train_once(
    *,
    raw_datasets,
    pseudo_label_fn: Callable,
    pseudo_filter_fn: Optional[Callable],
    model,
    model_checkpoint: str,
    output_dir: Path | str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    trainable_strategy: str,
    use_prompt: bool,
    load_from_cache_file: bool = False,
):
    pseudo_dataset = raw_datasets["train"].map(pseudo_label_fn, load_from_cache_file=load_from_cache_file)
    filtered_dataset = pseudo_dataset.filter(pseudo_filter_fn) if pseudo_filter_fn else pseudo_dataset

    train_dataset = map_dataset(filtered_dataset, use_prompt=use_prompt, is_train=True)
    validation_dataset = map_dataset(raw_datasets["test"], use_prompt=use_prompt, is_train=False)
    set_trainable_parameters(model, trainable_strategy)
    model, outcome = train_seq2seq_model(
        model=model,
        model_checkpoint=model_checkpoint,
        output_dir=output_dir,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    return model, outcome, pseudo_dataset, filtered_dataset
