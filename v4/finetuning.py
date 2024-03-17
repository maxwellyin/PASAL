# %%
import torch
import os
from tqdm.auto import tqdm
from datasets import load_from_disk, DatasetDict
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, T5Tokenizer
from utils.util import preprocess_training_examples, preprocess_validation_examples, compute_metrics, SOURCE_DOMAIN, MODEL_CHECKPOINT
# %%
TRAIN_DATA = f"../../largeQA/data/{SOURCE_DOMAIN}_train"
TEST_DATA = f"../../largeQA/data/{SOURCE_DOMAIN}_test"
model_checkpoint = MODEL_CHECKPOINT
# %%
raw_train_datasets = load_from_disk(TRAIN_DATA)
raw_validation_datasets = load_from_disk(TEST_DATA)
# %%
train_datasets = raw_train_datasets.map(preprocess_training_examples, batched=True, remove_columns=raw_train_datasets.column_names)
validation_datasets = raw_validation_datasets.map(preprocess_validation_examples, batched=True, remove_columns=raw_validation_datasets.column_names)

tokenized_datasets = DatasetDict({'train': train_datasets, 'validation': validation_datasets})
# %%
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
# %%
batch_size = 4
num_train_epochs = 8
# Show the training loss with every epoch\
logging_steps = len(tokenized_datasets["train"]) // batch_size

model_name = model_checkpoint.split("/")[-1]
output_dir=f"./checkpoint/{model_name}-{SOURCE_DOMAIN}",

args = Seq2SeqTrainingArguments(
    output_dir=f"./checkpoint/tmp/finetune/{model_name}-{SOURCE_DOMAIN}",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    report_to="tensorboard",
    fp16=False, # Overflows with fp16
)
# %%
tokenizer = T5Tokenizer.from_pretrained(MODEL_CHECKPOINT)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
# %%
trainer.train()
# %%
print(trainer.evaluate())
# %%
# after training
model.save_pretrained(f"./checkpoint/{model_name}-{SOURCE_DOMAIN}/base/final")
print("MODEL_CHECKPOINT:", MODEL_CHECKPOINT)
print("SOURCE_DOMAIN:", SOURCE_DOMAIN)
print(f"{os.path.basename(__file__)} finished.")