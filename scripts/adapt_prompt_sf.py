import argparse
from pathlib import Path

from transformers import AutoTokenizer

from pasal.experiments import load_raw_target_datasets, self_train_once
from pasal.model import T5ForConditionalGeneration as PromptT5
from pasal.pseudo_labeling import filter_answer_in_context, make_prompt_qa_pseudo_labeler
from pasal.runtime import DEFAULT_DATA_ROOT, REPO_ROOT, maybe_set_cuda_visible_devices, model_slug, resolve_data_dir
from pasal.util import MODEL_CHECKPOINT, PROMPT_LENGTH, SEED, SOURCE_DOMAIN, TARGET_DOMAIN, set_tokenizer_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Prompt-only source-free adaptation on pseudo-labeled target data.")
    parser.add_argument("--source-domain", default=SOURCE_DOMAIN, help="Source domain used to name checkpoints.")
    parser.add_argument("--target-domain", default=TARGET_DOMAIN, help="Target domain to adapt to.")
    parser.add_argument("--model-checkpoint", default=MODEL_CHECKPOINT, help="Backbone checkpoint used for tokenization.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Directory containing processed datasets.")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Initial checkpoint to continue training from.")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "checkpoints", help="Directory used to save checkpoints.")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device train and eval batch size.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of self-training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--train-subset", type=int, default=10000, help="Optional limit on target training examples.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed used for dataset shuffling.")
    parser.add_argument("--qg-model", default="lmqg/flan-t5-base-squad-qg", help="Question-generation checkpoint.")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES override.")
    return parser.parse_args()


def main():
    args = parse_args()
    maybe_set_cuda_visible_devices(args.cuda_visible_devices)
    set_tokenizer_checkpoint(args.model_checkpoint)

    model_name = model_slug(args.model_checkpoint)
    checkpoint_path = args.checkpoint_path or (
        args.output_root / f"prompt-{model_name}-{args.source_domain}" / "base" / "final"
    )
    output_dir = args.output_root / f"prompt-{model_name}-{args.source_domain}" / "prompt-sf" / args.target_domain

    train_path = resolve_data_dir(args.data_root, args.target_domain, "train")
    test_path = resolve_data_dir(args.data_root, args.target_domain, "test")
    raw_datasets = load_raw_target_datasets(train_path, test_path, seed=args.seed, train_subset=args.train_subset)

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = PromptT5.from_pretrained(str(checkpoint_path))
    pseudo_label_fn = make_prompt_qa_pseudo_labeler(tokenizer, model, args.qg_model)
    model, outcome, pseudo_dataset, filtered_dataset = self_train_once(
        raw_datasets=raw_datasets,
        pseudo_label_fn=pseudo_label_fn,
        pseudo_filter_fn=filter_answer_in_context,
        model=model,
        model_checkpoint=args.model_checkpoint,
        output_dir=output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        trainable_strategy="prompt_only",
        use_prompt=True,
    )

    final_dir = output_dir / "final" / "0"
    model.save_pretrained(str(final_dir))
    print(f"pseudo_set length: {len(pseudo_dataset)}")
    print(f"filtered pseudo_set length: {len(filtered_dataset)}")
    print("SOURCE_DOMAIN:", args.source_domain)
    print("TARGET_DOMAIN:", args.target_domain)
    print(f"PROMPT_LENGTH: {PROMPT_LENGTH}")
    print(f"INPUT_CHECKPOINT: {checkpoint_path}")
    print(f"OUTPUT_CHECKPOINT: {final_dir}")
    print(outcome)


if __name__ == "__main__":
    main()
