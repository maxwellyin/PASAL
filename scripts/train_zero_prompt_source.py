import argparse
from pathlib import Path

from pasal.experiments import load_train_validation_splits, set_trainable_parameters, train_seq2seq_model, zero_prompt_embedding
from pasal.model import T5ForConditionalGeneration as PromptT5
from pasal.runtime import DEFAULT_DATA_ROOT, REPO_ROOT, maybe_set_cuda_visible_devices, model_slug, resolve_data_dir
from pasal.util import MODEL_CHECKPOINT, PROMPT_LENGTH, SOURCE_DOMAIN, set_tokenizer_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train the prompt model on the source domain with a zeroed frozen prompt.")
    parser.add_argument("--source-domain", default=SOURCE_DOMAIN, help="Source domain name.")
    parser.add_argument("--model-checkpoint", default=MODEL_CHECKPOINT, help="Backbone checkpoint.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Directory containing processed datasets.")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "checkpoints", help="Directory used to save checkpoints.")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device train and eval batch size.")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=5.6e-5, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--run-name", default="12", help="Subdirectory used for checkpoint naming.")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES override.")
    return parser.parse_args()


def main():
    args = parse_args()
    maybe_set_cuda_visible_devices(args.cuda_visible_devices)
    set_tokenizer_checkpoint(args.model_checkpoint)

    train_path = resolve_data_dir(args.data_root, args.source_domain, "train")
    test_path = resolve_data_dir(args.data_root, args.source_domain, "test")
    tokenized_datasets = load_train_validation_splits(train_path, test_path, use_prompt=True)

    model_name = model_slug(args.model_checkpoint)
    output_dir = args.output_root / f"prompt-{model_name}-{args.source_domain}" / args.run_name
    model = PromptT5.from_pretrained(args.model_checkpoint)
    zero_prompt_embedding(model)
    set_trainable_parameters(model, "freeze_prompt")
    model, outcome = train_seq2seq_model(
        model=model,
        model_checkpoint=args.model_checkpoint,
        output_dir=output_dir,
        train_dataset=tokenized_datasets["train"],
        validation_dataset=tokenized_datasets["validation"],
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    print(outcome)
    print("MODEL_CHECKPOINT:", args.model_checkpoint)
    print("SOURCE_DOMAIN:", args.source_domain)
    print(f"PROMPT_LENGTH: {PROMPT_LENGTH}")
    print(f"Saved model to: {final_dir}")


if __name__ == "__main__":
    main()
