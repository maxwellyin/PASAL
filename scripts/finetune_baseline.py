import argparse
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM

from pasal.experiments import load_train_validation_splits, train_seq2seq_model
from pasal.runtime import DEFAULT_DATA_ROOT, REPO_ROOT, maybe_set_cuda_visible_devices, model_slug, resolve_data_dir
from pasal.util import MODEL_CHECKPOINT, SOURCE_DOMAIN, set_tokenizer_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Run the standard seq2seq fine-tuning baseline.")
    parser.add_argument("--source-domain", default=SOURCE_DOMAIN, help="Source domain name.")
    parser.add_argument("--model-checkpoint", default=MODEL_CHECKPOINT, help="Backbone checkpoint.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Directory containing processed datasets.")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "checkpoints", help="Directory used to save checkpoints.")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device train and eval batch size.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=5.6e-5, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES override.")
    return parser.parse_args()


def main():
    args = parse_args()
    maybe_set_cuda_visible_devices(args.cuda_visible_devices)
    set_tokenizer_checkpoint(args.model_checkpoint)

    train_path = resolve_data_dir(args.data_root, args.source_domain, "train")
    test_path = resolve_data_dir(args.data_root, args.source_domain, "test")
    tokenized_datasets = load_train_validation_splits(train_path, test_path, use_prompt=False)

    model_name = model_slug(args.model_checkpoint)
    output_dir = args.output_root / f"{model_name}-{args.source_domain}" / "base"
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)
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
    print(f"Saved model to: {final_dir}")


if __name__ == "__main__":
    main()
