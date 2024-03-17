# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# %%
import shutil
import os, torch
from tqdm.auto import tqdm
from datasets import load_from_disk, DatasetDict
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, T5Tokenizer
from utils.util import preprocess_training_examples, preprocess_validation_examples, compute_metrics, pad_prompt_length, SOURCE_DOMAIN, MODEL_CHECKPOINT, PROMPT_LENGTH, T5_MAX_INPUT_LENGTH
from utils.model import T5ForConditionalGeneration as PromptT5
# %%
TRAIN_DATA = f"../../largeQA/data/{SOURCE_DOMAIN}_train"
TEST_DATA = f"../../largeQA/data/{SOURCE_DOMAIN}_test"
model_name = MODEL_CHECKPOINT.split("/")[-1]
model_checkpoint = MODEL_CHECKPOINT
# %%
raw_train_datasets = load_from_disk(TRAIN_DATA)
raw_validation_datasets = load_from_disk(TEST_DATA)
# %%
pmaxlen = T5_MAX_INPUT_LENGTH - PROMPT_LENGTH
func1 = lambda x: pad_prompt_length(preprocess_training_examples(x, max_length=pmaxlen))
func2 = lambda x: pad_prompt_length(preprocess_validation_examples(x, max_length=pmaxlen))

train_datasets = raw_train_datasets.map(func1, batched=True, remove_columns=raw_train_datasets.column_names)
validation_datasets = raw_validation_datasets.map(func2, batched=True, remove_columns=raw_validation_datasets.column_names)

tokenized_datasets = DatasetDict({'train': train_datasets, 'validation': validation_datasets})
# %%
model = PromptT5.from_pretrained(model_checkpoint)
prompt_embedding_shape = model.encoder.prompt_embedding.data.shape
zero_tensor = torch.zeros(prompt_embedding_shape)
model.encoder.prompt_embedding.data = zero_tensor
# Set requires_grad to True only for prompt_embedding
model.encoder.prompt_embedding.requires_grad = False
# %%
batch_size = 4
num_train_epochs = 12
# Show the training loss with every epoch
args = Seq2SeqTrainingArguments(
    output_dir=f"./checkpoint/prompt-{model_name}-{SOURCE_DOMAIN}/12",
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_strategy = "epoch",
    report_to="tensorboard",
    fp16=False,
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
model.save_pretrained(f"./checkpoint/prompt-{model_name}-{SOURCE_DOMAIN}/12/final")
print("MODEL_CHECKPOINT:", MODEL_CHECKPOINT)
print("SOURCE_DOMAIN:", SOURCE_DOMAIN)
print(f"PROMPT_LENGTH: {PROMPT_LENGTH}")
print(f"{os.path.basename(__file__)} finished.")
# %%
