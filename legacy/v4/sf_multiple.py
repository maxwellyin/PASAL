import argparse
from pathlib import Path

from transformers import AutoTokenizer

from utils.experiments import load_raw_target_datasets, self_train_once
from utils.model import T5ForConditionalGeneration as PromptT5
from utils.pseudo_labeling import filter_answer_in_context, make_prompt_qa_pseudo_labeler
from utils.runtime import DEFAULT_DATA_ROOT, V4_ROOT, maybe_set_cuda_visible_devices, model_slug, resolve_data_dir
from utils.util import MODEL_CHECKPOINT, PROMPT_LENGTH, SEED, SOURCE_DOMAIN, TARGET_DOMAIN, set_tokenizer_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Repeated model self-training using generated QA pseudo labels.")
    parser.add_argument("--source-domain", default=SOURCE_DOMAIN, help="Source domain used to name checkpoints.")
    parser.add_argument("--target-domain", default=TARGET_DOMAIN, help="Target domain to adapt to.")
    parser.add_argument("--model-checkpoint", default=MODEL_CHECKPOINT, help="Backbone checkpoint used for tokenization.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Directory containing processed datasets.")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Initial checkpoint to continue training from.")
    parser.add_argument("--output-root", type=Path, default=V4_ROOT / "checkpoint", help="Directory used to save checkpoints.")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device train and eval batch size.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of self-training epochs.")
    parser.add_argument("--learning-rate", type=float, default=5.6e-5, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--train-subset", type=int, default=10000, help="Optional limit on target training examples.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed used for dataset shuffling.")
    parser.add_argument("--rounds", type=int, default=1, help="Number of adaptation rounds.")
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
    output_dir = args.output_root / f"prompt-{model_name}-{args.source_domain}" / "sf"

    train_path = resolve_data_dir(args.data_root, args.target_domain, "train")
    test_path = resolve_data_dir(args.data_root, args.target_domain, "test")
    raw_datasets = load_raw_target_datasets(train_path, test_path, seed=args.seed, train_subset=args.train_subset)

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = PromptT5.from_pretrained(str(checkpoint_path))
    outcomes = []

    for round_index in range(args.rounds):
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
            trainable_strategy="freeze_prompt",
            use_prompt=True,
        )
        outcomes.append(outcome)
        model.save_pretrained(str(output_dir / str(round_index)))
        print(f"round={round_index} pseudo_set length: {len(pseudo_dataset)}")
        print(f"round={round_index} filtered pseudo_set length: {len(filtered_dataset)}")
        print(outcome)

    for outcome in outcomes:
        print(outcome)


if __name__ == "__main__":
    main()
