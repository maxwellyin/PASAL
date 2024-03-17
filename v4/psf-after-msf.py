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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, T5Tokenizer, pipeline
from utils.util import preprocess_training_examples, preprocess_validation_examples, compute_metrics, pad_prompt_length, SOURCE_DOMAIN, TARGET_DOMAIN, MODEL_CHECKPOINT, PROMPT_LENGTH, T5_MAX_INPUT_LENGTH, SEED, format_for_lmqg
from os import path, makedirs
from accelerate import Accelerator
from utils.model import T5ForConditionalGeneration as PromptT5
# %%
TRAIN_DATA = f"../../largeQA/data/{TARGET_DOMAIN}_train"
TEST_DATA = f"../../largeQA/data/{TARGET_DOMAIN}_test"
VAL_SPLIT = 'test'
BATCH_SIZE = 8
model_name = MODEL_CHECKPOINT.split("/")[-1]
model_checkpoint = f"./checkpoint/{model_name}-{SOURCE_DOMAIN}/msf-no-finetune-on-source/{TARGET_DOMAIN}/0/"

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
model = PromptT5.from_pretrained(model_checkpoint)
# %%
pmaxlen = T5_MAX_INPUT_LENGTH - PROMPT_LENGTH
func1 = lambda x: pad_prompt_length(preprocess_training_examples(x, max_length=pmaxlen))
func2 = lambda x: pad_prompt_length(preprocess_validation_examples(x, max_length=pmaxlen))
# %%
train_datasets = load_from_disk(TRAIN_DATA)
train_datasets = train_datasets.select(range(10000)).shuffle(SEED)
test_dataset = load_from_disk(TEST_DATA)
raw_datasets = DatasetDict({
    'train': train_datasets,
    'test': test_dataset})

# %%
def train(train_dataset, validation_dataset, model):
    train_dataset2 = train_dataset.map(func1, batched=True, remove_columns=train_dataset.column_names)
    validation_datasets = validation_dataset.map(func2, batched=True, remove_columns=validation_dataset.column_names)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Set requires_grad to True only for prompt_embedding
    model.encoder.prompt_embedding.requires_grad = True

    batch_size = BATCH_SIZE
    num_train_epochs = 1
    # Show the training loss with every epoch
    args = Seq2SeqTrainingArguments(
        output_dir=f"./checkpoint/prompt-{model_name}-{SOURCE_DOMAIN}/prompt-sf/{TARGET_DOMAIN}",
        evaluation_strategy="epoch",
        learning_rate=1e-3,
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
        train_dataset=train_dataset2,
        eval_dataset=validation_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    outcome = trainer.evaluate()
    print(outcome, '\n')
    return model, outcome

# %%
def main(i:int, model):
    question_generation = pipeline("text2text-generation", "lmqg/flan-t5-base-squad-qg", device=torch.cuda.is_available()-1)

    qa_model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    question_answerer = pipeline("text2text-generation", model=qa_model, tokenizer=tokenizer, device=torch.cuda.is_available()-1)

    def create_pseudo_label(example):
        qg_input = format_for_lmqg(example)
        output_qg = question_generation(qg_input)[0]['generated_text']
        qa_input = f"question: {output_qg} context: {example['context']}"
        out = question_answerer(qa_input)[0]['generated_text']
        answers = {'text': [out], 'answer_start': example['answers']['answer_start']}
        return {'question': output_qg, 'answers': answers}

    pseudo_set = raw_datasets['train'].map(create_pseudo_label, load_from_cache_file=False)
    pseudo_set2 = pseudo_set.filter(lambda example: example['answers']['text'][0] in example['context'])
    print(f"pseudo_set2 length: {len(pseudo_set2)}")

    model, outcome = train(pseudo_set2, raw_datasets['test'], model)
    return model, outcome
# %%
outcomes = []
for i in range(1):
    model, outcome = main(i, model)
    outcomes.append(outcome)
    model.save_pretrained(f"./checkpoint/prompt-{model_name}-{SOURCE_DOMAIN}/prompt-sf/{TARGET_DOMAIN}/final/{i}")
for outcome in outcomes:
    print(outcome)
# %%
