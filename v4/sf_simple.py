# 不用prompt
# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# %%
import torch, shutil
import numpy as np
from tqdm.auto import tqdm
from torch.optim import AdamW, Adam
from datasets import DatasetDict, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, T5Tokenizer, pipeline, T5ForConditionalGeneration
from utils.util import preprocess_training_examples, preprocess_validation_examples, compute_metrics, pad_prompt_length, SOURCE_DOMAIN, TARGET_DOMAIN, MODEL_CHECKPOINT, PROMPT_LENGTH, T5_MAX_INPUT_LENGTH, SEED, NUM_LOOPS
from os import path, makedirs
from accelerate import Accelerator
# %%
TRAIN_DATA = f"../../largeQA/data/{TARGET_DOMAIN}_train"
TEST_DATA = f"../../largeQA/data/{TARGET_DOMAIN}_test"
VAL_SPLIT = 'test'
THRESHOLD = 0.5
BATCH_SIZE = 8
USE_FINETUNED_MODEL = False
model_name = MODEL_CHECKPOINT.split("/")[-1]
model_checkpoint = f"./checkpoint/{model_name}-{SOURCE_DOMAIN}/base/final" if USE_FINETUNED_MODEL else MODEL_CHECKPOINT
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
# %%
train_datasets = load_from_disk(TRAIN_DATA)
train_datasets = train_datasets.select(range(10000)).shuffle(SEED)
test_dataset = load_from_disk(TEST_DATA)
raw_datasets = DatasetDict({
    'train': train_datasets,
    'test': test_dataset})

# %%
def train(train_dataset, validation_dataset, model, output_dir):
    train_datasets = train_dataset.map(preprocess_training_examples, batched=True, remove_columns=train_dataset.column_names)
    validation_datasets = validation_dataset.map(preprocess_validation_examples, batched=True, remove_columns=validation_dataset.column_names)

    batch_size = BATCH_SIZE
    num_train_epochs = 1
    # Show the training loss with every epoch
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
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

    tokenizer = T5Tokenizer.from_pretrained(MODEL_CHECKPOINT)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_datasets,
        eval_dataset=validation_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(trainer.evaluate())
    return model

# %%
def main(i:int, model, output_dir):
    qa_checkpoint = "deepset/roberta-base-squad2"
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_checkpoint)
    question_answerer = pipeline("question-answering", model=qa_checkpoint, tokenizer=qa_tokenizer, device=torch.cuda.is_available()-1)

    def create_pseudo_label(example):
        out = question_answerer(question=example['question'], context=example['context'])
        answers = {'text': [out['answer']], 'answer_start': [out['start']]}
        return {'answers': answers, 'score': out['score']}

    pseudo_set = raw_datasets['train'].map(create_pseudo_label)
    pseudo_set2 = pseudo_set.filter(lambda example: example['score']>THRESHOLD)
    print(f"pseudo_set2 length: {len(pseudo_set2)}")

    model = train(pseudo_set2, raw_datasets['test'], model, output_dir)
    return model
# %%
outcomes = []
output_dir = f"./checkpoint/{model_name}-{SOURCE_DOMAIN}/msf/{TARGET_DOMAIN}/" if USE_FINETUNED_MODEL else f"./checkpoint/{model_name}-{SOURCE_DOMAIN}/msf-no-finetune-on-source/{TARGET_DOMAIN}/"

for i in range(1):
    model = main(i, model, output_dir)
    model.save_pretrained(f"{output_dir}{i}")

# %%
