import collections
import numpy as np
from tqdm.auto import tqdm
from transformers import T5Tokenizer
from nltk.tokenize import sent_tokenize
import os

SEED = 8888
PROMPT_LENGTH = 100
T5_MAX_INPUT_LENGTH = 512
NUM_LOOPS = 10

# "SQuAD" "NaturalQuestionsShort" "HotpotQA" "NewsQA" "BioASQ" "TriviaQA-web" "SearchQA"
SOURCE_DOMAIN = "SQuAD"
TARGET_DOMAIN = "NaturalQuestionsShort"

# google/t5-v1_1-base t5-base google/flan-t5-base google/flan-t5-xxl lmqg/flan-t5-base-squad-qg
# MODEL_CHECKPOINT = "lmqg/flan-t5-base-squad-qg"
MODEL_CHECKPOINT = "google/flan-t5-base"
# MODEL_CHECKPOINT = "t5-base"

tokenizer = T5Tokenizer.from_pretrained(MODEL_CHECKPOINT)

def preprocess_training_examples(batch, max_length=512, stride=256):
    batch_size = len(batch['question'])
    new_batch = {"id": [], "input_ids": [], "attention_mask": [], "labels": []}
    
    for idx in range(batch_size):
        question = batch["question"][idx]
        context = batch["context"][idx]
        answer_start = batch["answers"][idx]["answer_start"][0]
        answer_text = batch["answers"][idx]["text"][0]

        # Tokenize question and context separately with prefixes
        tokenized_question = tokenizer.tokenize(f"question: {question}")
        tokenized_context = tokenizer.tokenize(f"context: {context}")

        if len(tokenized_question) + len(tokenized_context) + 1 <= max_length:
            # If the combined length is within the limit, tokenize the input
            model_inputs = tokenizer(
                f"question: {question} context: {context}",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
            )

            # Tokenize answer
            labels = tokenizer(answer_text)

            # Add the tokenized inputs and labels to the new batch
            new_batch["id"].append(batch["id"][idx])
            new_batch["input_ids"].append(model_inputs["input_ids"])
            new_batch["attention_mask"].append(model_inputs["attention_mask"])
            new_batch["labels"].append(labels["input_ids"])
        else:
            start_idx = 0
            while start_idx < len(context):
                end_idx = min(start_idx + max_length - len(tokenized_question) - 1, len(context))
                split_context = context[start_idx:end_idx]
                split_answer_start = answer_start - start_idx
                split_answer_end = split_answer_start + len(answer_text)

                # Tokenize question and split_context
                model_inputs = tokenizer(
                    f"question: {question} context: {split_context}",
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                )

                if split_answer_start >= 0 and split_answer_end <= len(split_context):
                    # Tokenize answer
                    labels = tokenizer(answer_text)
                else:
                    # Set labels to -100 if answer is not present in the split_context
                    labels = {"input_ids": [-100] * len(tokenizer.tokenize(answer_text))}

                # Add the tokenized inputs and labels to the new batch
                new_batch["id"].append(batch["id"][idx])
                new_batch["input_ids"].append(model_inputs["input_ids"])
                new_batch["attention_mask"].append(model_inputs["attention_mask"])
                new_batch["labels"].append(labels["input_ids"])

                start_idx += stride

    return new_batch

def preprocess_validation_examples(batch, max_length=512, stride=256):
    batch_size = len(batch['question'])
    new_batch = {"id": [], "input_ids": [], "attention_mask": [], "labels": []}
    
    for idx in range(batch_size):
        question = batch["question"][idx]
        context = batch["context"][idx]
        answer_start = batch["answers"][idx]["answer_start"][0]
        answer_text = batch["answers"][idx]["text"][0]

        # Tokenize question and context separately with prefixes
        tokenized_question = tokenizer.tokenize(f"question: {question}")
        tokenized_context = tokenizer.tokenize(f"context: {context}")

        if len(tokenized_question) + len(tokenized_context) + 1 <= max_length:
            # If the combined length is within the limit, tokenize the input
            model_inputs = tokenizer(
                f"question: {question} context: {context}",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
            )

            # Tokenize answer
            labels = tokenizer(answer_text)

            # Add the tokenized inputs and labels to the new batch
            new_batch["id"].append(batch["id"][idx])
            new_batch["input_ids"].append(model_inputs["input_ids"])
            new_batch["attention_mask"].append(model_inputs["attention_mask"])
            new_batch["labels"].append(labels["input_ids"])
        else:
            start_idx = 0
            while start_idx < len(context):
                end_idx = min(start_idx + max_length - len(tokenized_question) - 1, len(context))
                split_context = context[start_idx:end_idx]
                split_answer_start = answer_start - start_idx
                split_answer_end = split_answer_start + len(answer_text)

                # Tokenize question and split_context
                model_inputs = tokenizer(
                    f"question: {question} context: {split_context}",
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                )

                if split_answer_start >= 0 and split_answer_end <= len(split_context):
                    # Tokenize answer
                    labels = tokenizer(answer_text)

                    # Add the tokenized inputs and labels to the new batch
                    new_batch["id"].append(batch["id"][idx])
                    new_batch["input_ids"].append(model_inputs["input_ids"])
                    new_batch["attention_mask"].append(model_inputs["attention_mask"])
                    new_batch["labels"].append(labels["input_ids"])

                start_idx += stride

    return new_batch

def pad_prompt_length(batch, prompt_length=PROMPT_LENGTH):
    padded_batch = {
        # "id": [],
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    for idx, input_ids in enumerate(batch["input_ids"]):
        pad_ids = [tokenizer.pad_token_id] * prompt_length
        padded_input_ids = pad_ids + input_ids
        padded_attention_mask = [1] * prompt_length + batch["attention_mask"][idx]

        # padded_batch["id"].append(batch["id"][idx])
        padded_batch["input_ids"].append(padded_input_ids)
        padded_batch["attention_mask"].append(padded_attention_mask)
        padded_batch["labels"].append(batch["labels"][idx])

    return padded_batch




def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    import re
    import string

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))

def exact_match_score(prediction, truth):
    return int(normalize_answer(prediction) == normalize_answer(truth))

def f1_score(prediction, truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()

    common_tokens = set(pred_tokens) & set(truth_tokens)
    common_token_count = sum([pred_tokens.count(t) for t in common_tokens])

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        # If either the prediction or the truth is an empty string, return 0
        return 0

    precision = common_token_count / len(pred_tokens)
    recall = common_token_count / len(truth_tokens)

    if precision + recall == 0:
        return 0

    return 2 * (precision * recall) / (precision + recall)

def _compute_metrics(predictions, ground_truths):
    f1_sum = 0
    em_sum = 0
    total_count = len(predictions)

    for i in range(total_count):
        prediction = predictions[i]
        ground_truth = ground_truths[i]

        f1_sum += f1_score(prediction, ground_truth)
        em_sum += exact_match_score(prediction, ground_truth)

    f1_avg = f1_sum / total_count
    em_avg = em_sum / total_count

    return {"F1": f1_avg, "EM": em_avg}

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode predictions and labels into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Split decoded_preds and decoded_labels into sentences
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    # Compute F1 and exact match scores
    result = _compute_metrics(decoded_preds, decoded_labels)
    return {k: round(v, 4) for k, v in result.items()}

def format_for_lmqg(example):
    """
    Correctly formats the dataset example for question generation.
    """
    # Extracting the answer's text and start position
    answer_text = example['answers']['text'][0]
    answer_start = example['answers']['answer_start'][0]

    # Inserting the <hl> tags around the answer in the context
    modified_context = example['context'][:answer_start] + "<hl> " + answer_text + " <hl>" + example['context'][answer_start + len(answer_text):]

    # Removing the <P> and </P> tags from the context
    modified_context = modified_context.replace("<P>", "").replace("</P>", "").strip()

    # Creating the input string for the pipeline
    input_string = "generate question: " + modified_context

    return input_string

def question2batch(question:str, context: str):
    batch = {"question":[question], "context":[context], "answers":[{"answer_start":0, "text":""}]}

    pmaxlen = T5_MAX_INPUT_LENGTH - PROMPT_LENGTH
    batch2 = preprocess_validation_examples(batch, max_length=pmaxlen)
    batch3 = pad_prompt_length(batch2)
