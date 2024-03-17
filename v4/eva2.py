# %%
import torch
import os
from tqdm.auto import tqdm
from datasets import load_from_disk, DatasetDict
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, T5Tokenizer
from utils.util import preprocess_training_examples, preprocess_validation_examples, compute_metrics, pad_prompt_length, SOURCE_DOMAIN, TARGET_DOMAIN, MODEL_CHECKPOINT, PROMPT_LENGTH, T5_MAX_INPUT_LENGTH
from utils.model import PromptT5DANN
# %%
TEST_DATA = f"../../largeQA/data/{TARGET_DOMAIN}_test"
model_name = MODEL_CHECKPOINT.split("/")[-1]
model_checkpoint = f"./checkpoint/prompt-{model_name}-{SOURCE_DOMAIN}/dann"
# %%
raw_validation_datasets = load_from_disk(TEST_DATA)
# %%
pmaxlen = T5_MAX_INPUT_LENGTH - PROMPT_LENGTH
func2 = lambda x: pad_prompt_length(preprocess_validation_examples(x, max_length=pmaxlen))

validation_datasets = raw_validation_datasets.map(func2, batched=True, remove_columns=raw_validation_datasets.column_names)

tokenized_datasets = DatasetDict({'validation': validation_datasets})
# %%
model = PromptT5DANN.from_pretrained(model_checkpoint)
# %%
batch_size = 4
num_train_epochs = 8
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"./out",
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    fp16=False,
)
# %%
tokenizer = T5Tokenizer.from_pretrained(MODEL_CHECKPOINT)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    args,
    # train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
# %%
print(trainer.evaluate())
# %%
print("MODEL_CHECKPOINT:", MODEL_CHECKPOINT)
print("SOURCE_DOMAIN:", SOURCE_DOMAIN)
print(f"PROMPT_LENGTH: {PROMPT_LENGTH}")
print(f"{os.path.basename(__file__)} finished.")
# %%
