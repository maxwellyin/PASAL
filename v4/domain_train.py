# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# %%
import torch, shutil
import numpy as np
from tqdm.auto import tqdm
from torch.optim import AdamW, Adam
from datasets import DatasetDict, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, pipeline, default_data_collator, get_scheduler, DataCollatorForSeq2Seq
from utils.util import preprocess_training_examples, preprocess_validation_examples, compute_metrics, pad_prompt_length, SOURCE_DOMAIN, TARGET_DOMAIN, MODEL_CHECKPOINT, PROMPT_LENGTH, T5_MAX_INPUT_LENGTH, SEED, BATCH_SIZE
import os
from accelerate import Accelerator
from utils.model import PromptT5DANN
# %%
SOURCE_TRAIN_DATA = f"../../largeQA/data/{SOURCE_DOMAIN}_train"
SOURCE_TEST_DATA = f"../../largeQA/data/{SOURCE_DOMAIN}_test"
TARGET_TRAIN_DATA = f"../../largeQA/data/{TARGET_DOMAIN}_train"
TARGET_TEST_DATA = f"../../largeQA/data/{TARGET_DOMAIN}_test"
VAL_SPLIT = "test"
NUM_EXAMPLES = 10000
LR_BASE = 2e-5
LR_DOMAIN = 1e-3

model_name = MODEL_CHECKPOINT.split("/")[-1]
model_checkpoint = f"./checkpoint/prompt-{model_name}-{SOURCE_DOMAIN}/base/final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
model = PromptT5DANN.from_pretrained(model_checkpoint)
# %%
source_train_datasets = load_from_disk(SOURCE_TRAIN_DATA).shuffle(seed=SEED)
target_train_datasets = load_from_disk(TARGET_TRAIN_DATA).shuffle(seed=SEED)

train_num = min(len(source_train_datasets), len(target_train_datasets))
train_num = 10000
# %%
pmaxlen = T5_MAX_INPUT_LENGTH - PROMPT_LENGTH
func1 = lambda x: pad_prompt_length(preprocess_training_examples(x, max_length=pmaxlen))
func2 = lambda x: pad_prompt_length(preprocess_validation_examples(x, max_length=pmaxlen))

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

source_train_dataset = source_train_datasets.map(
    func1,
    batched=True,
    remove_columns=source_train_datasets.column_names,
)

source_train_dataset.set_format("torch")
source_train_dataloader = DataLoader(
    source_train_dataset.select(range(train_num)), ##截断，因为map后可能超过一千
    shuffle=True,
    collate_fn=data_collator,
    batch_size=BATCH_SIZE,
)
# %%
target_train_dataset = target_train_datasets.map(
    func1,
    batched=True,
    remove_columns=target_train_datasets.column_names,
)

target_train_dataset.set_format("torch")
target_train_dataloader = DataLoader(
    target_train_dataset.select(range(train_num)),
    shuffle=True,
    collate_fn=data_collator,
    batch_size=BATCH_SIZE,
)
# %%
params_base = [v for k, v in model.named_parameters() if "domain_classifier" not in k]
params_domain = [v for k, v in model.named_parameters() if "domain_classifier" in k]
optimizer_base = AdamW(params_base, lr= LR_BASE)
optimizer_domain = Adam(params_domain, lr= LR_DOMAIN)
# %%
accelerator = Accelerator()
model, optimizer_base, optimizer_domain, source_train_dataloader, target_train_dataloader = accelerator.prepare(
    model, optimizer_base, optimizer_domain, source_train_dataloader, target_train_dataloader
)
# %%
num_train_epochs = 3
num_update_steps_per_epoch = len(source_train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler_base = get_scheduler(
        "linear",
        optimizer=optimizer_base,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

lr_scheduler_domain = get_scheduler(
        "linear",
        optimizer=optimizer_domain,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
# %%
progress_bar = tqdm(range(num_training_steps))

model.encoder.prompt_embedding.requires_grad = False
model.use_domain_loss = True
model.train()
for epoch in range(num_train_epochs):       
    for step, (source_batch, target_batch) in enumerate(zip(source_train_dataloader, target_train_dataloader)):
        
        batch_size = source_batch['attention_mask'].shape[0]
        source_batch['domain_label'] = torch.zeros(batch_size).long().cuda()
        source_outputs = model(**source_batch)
        source_loss = source_outputs.loss

        target_batch['domain_label'] = torch.ones(batch_size).long().cuda()
        target_outputs = model(**target_batch)
        target_loss = target_outputs.loss


        accelerator.backward(source_loss+target_loss)

        optimizer_base.step()
        lr_scheduler_base.step()
        optimizer_base.zero_grad()

        optimizer_domain.step()
        lr_scheduler_domain.step()
        optimizer_domain.zero_grad()
        progress_bar.update(1)
# %%
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(f"./checkpoint/prompt-{model_name}-{SOURCE_DOMAIN}/dann", save_function=accelerator.save)
# %%
accelerator.print(f"{os.path.basename(__file__)} finished.")
# %%
