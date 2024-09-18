import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import load_from_disk
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, get_scheduler

from utils.experiments import map_dataset
from utils.model import PromptT5DANN
from utils.runtime import DEFAULT_DATA_ROOT, V4_ROOT, maybe_set_cuda_visible_devices, model_slug, resolve_data_dir
from utils.util import MODEL_CHECKPOINT, SEED, SOURCE_DOMAIN, TARGET_DOMAIN, set_tokenizer_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Domain-adversarial training for PASAL.")
    parser.add_argument("--source-domain", default=SOURCE_DOMAIN, help="Source domain name.")
    parser.add_argument("--target-domain", default=TARGET_DOMAIN, help="Target domain name.")
    parser.add_argument("--model-checkpoint", default=MODEL_CHECKPOINT, help="Backbone checkpoint used for tokenization.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Directory containing processed datasets.")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Initial checkpoint to continue training from.")
    parser.add_argument("--output-root", type=Path, default=V4_ROOT / "checkpoint", help="Directory used to save checkpoints.")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device batch size.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--base-learning-rate", type=float, default=2e-5, help="Learning rate for the QA model.")
    parser.add_argument("--domain-learning-rate", type=float, default=1e-3, help="Learning rate for the domain classifier.")
    parser.add_argument("--train-subset", type=int, default=10000, help="Optional limit on aligned source/target examples.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed used for dataset shuffling.")
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
    output_dir = args.output_root / f"prompt-{model_name}-{args.source_domain}" / "dann"

    source_train_path = resolve_data_dir(args.data_root, args.source_domain, "train")
    target_train_path = resolve_data_dir(args.data_root, args.target_domain, "train")
    source_train = load_from_disk(str(source_train_path)).shuffle(seed=args.seed)
    target_train = load_from_disk(str(target_train_path)).shuffle(seed=args.seed)
    train_subset = min(len(source_train), len(target_train), args.train_subset)

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = PromptT5DANN.from_pretrained(str(checkpoint_path))
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    source_train = map_dataset(source_train.select(range(train_subset)), use_prompt=True, is_train=True)
    target_train = map_dataset(target_train.select(range(train_subset)), use_prompt=True, is_train=True)
    source_train.set_format("torch")
    target_train.set_format("torch")

    source_loader = DataLoader(source_train, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    target_loader = DataLoader(target_train, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)

    params_base = [value for name, value in model.named_parameters() if "domain_classifier" not in name]
    params_domain = [value for name, value in model.named_parameters() if "domain_classifier" in name]
    optimizer_base = AdamW(params_base, lr=args.base_learning_rate)
    optimizer_domain = Adam(params_domain, lr=args.domain_learning_rate)

    accelerator = Accelerator()
    model, optimizer_base, optimizer_domain, source_loader, target_loader = accelerator.prepare(
        model, optimizer_base, optimizer_domain, source_loader, target_loader
    )

    num_training_steps = args.epochs * len(source_loader)
    lr_scheduler_base = get_scheduler(
        "linear", optimizer=optimizer_base, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    lr_scheduler_domain = get_scheduler(
        "linear", optimizer=optimizer_domain, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    model.encoder.prompt_embedding.requires_grad = False
    model.use_domain_loss = True
    model.train()
    progress_bar = tqdm(range(num_training_steps))

    for _ in range(args.epochs):
        for source_batch, target_batch in zip(source_loader, target_loader):
            batch_size = source_batch["attention_mask"].shape[0]
            device = source_batch["attention_mask"].device

            source_batch["domain_label"] = torch.zeros(batch_size, device=device, dtype=torch.long)
            target_batch["domain_label"] = torch.ones(batch_size, device=device, dtype=torch.long)

            source_loss = model(**source_batch).loss
            target_loss = model(**target_batch).loss
            accelerator.backward(source_loss + target_loss)

            optimizer_base.step()
            lr_scheduler_base.step()
            optimizer_base.zero_grad()

            optimizer_domain.step()
            lr_scheduler_domain.step()
            optimizer_domain.zero_grad()
            progress_bar.update(1)

    accelerator.wait_for_everyone()
    accelerator.unwrap_model(model).save_pretrained(str(output_dir), save_function=accelerator.save)
    accelerator.print(f"Saved model to: {output_dir}")


if __name__ == "__main__":
    main()
