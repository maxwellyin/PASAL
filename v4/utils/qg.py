from tqdm.auto import tqdm
from transformers import T5Tokenizer
import os
from nltk.tokenize import sent_tokenize
import evaluate
import numpy as np
from .util import MODEL_CHECKPOINT

if "google" in MODEL_CHECKPOINT:
    tokenizer_dir = os.path.join(os.path.dirname(__file__), 'qg-tokenizer')
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir)
else:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_CHECKPOINT)

def squad_to_question_generation(squad_example):
    context = squad_example['context']
    question = squad_example['question']
    answer_start = squad_example['answers']['answer_start'][0]
    answer_text = squad_example['answers']['text'][0]

    answer_end = answer_start + len(answer_text)

    sentence_start = context[:answer_start].rfind('.', 0, answer_start) + 2
    sentence_end = context.find('.', answer_end) + 1
    sentence = context[sentence_start:sentence_end].strip()

    sentence_answer = sentence[:answer_start - sentence_start] + '<hl>' + sentence[answer_start - sentence_start:answer_end - sentence_start] + '<hl>' + sentence[answer_end - sentence_start:]

    paragraph_sentence = context[:sentence_start] + '<hl>' + context[sentence_start:sentence_end] + '<hl>' + context[sentence_end:]

    paragraph_answer = context[:answer_start] + '<hl>' + context[answer_start:answer_end] + '<hl>' + context[answer_end:]

    question_generation_example = {
        'question': question,
        'paragraph': context,
        'answer': answer_text,
        'sentence': sentence,
        'paragraph_sentence': paragraph_sentence,
        'paragraph_answer': paragraph_answer,
        'sentence_answer': sentence_answer
    }

    return question_generation_example

def preprocess_qg_examples(batch, max_length=512, stride=256):
    batch_size = len(batch['question'])
    new_batch = {"input_ids": [], "attention_mask": [], "labels": []}
    
    generate_question_prefix = "generate question: "
    generate_question_prefix_length = len(tokenizer.tokenize(generate_question_prefix))

    for idx in range(batch_size):
        question = batch["question"][idx]
        paragraph_answer = batch["paragraph_answer"][idx]

        # Tokenize paragraph_answer without the generate question prefix
        tokenized_paragraph_answer = tokenizer.tokenize(f"{paragraph_answer}")

        if len(tokenized_paragraph_answer) + generate_question_prefix_length + 1 <= max_length:
            # If the paragraph_answer length is within the limit, tokenize the input
            model_inputs = tokenizer(
                f"{generate_question_prefix}{paragraph_answer}",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
            )

            # Tokenize question as labels
            labels = tokenizer(question)

            # Add the tokenized inputs and labels to the new batch
            new_batch["input_ids"].append(model_inputs["input_ids"])
            new_batch["attention_mask"].append(model_inputs["attention_mask"])
            new_batch["labels"].append(labels["input_ids"])
        else:
            start_idx = 0
            while start_idx < len(tokenized_paragraph_answer):
                end_idx = min(start_idx + max_length - generate_question_prefix_length - 1, len(tokenized_paragraph_answer))
                split_tokenized_paragraph_answer = tokenized_paragraph_answer[start_idx:end_idx]

                # Check if both <hl> tokens are present in the split_tokenized_paragraph_answer
                if split_tokenized_paragraph_answer.count('<hl>') == 2:
                    # Tokenize split_tokenized_paragraph_answer
                    model_inputs = tokenizer(
                        f"{generate_question_prefix}{split_tokenized_paragraph_answer}",
                        max_length=max_length,
                        padding="max_length",
                        truncation=True,
                        return_attention_mask=True,
                    )

                    # Tokenize question as labels
                    labels = tokenizer(question)

                    # Add the tokenized inputs and labels to the new batch
                    new_batch["input_ids"].append(model_inputs["input_ids"])
                    new_batch["attention_mask"].append(model_inputs["attention_mask"])
                    new_batch["labels"].append(labels["input_ids"])

                start_idx += stride

    return new_batch


rouge_score = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {key: value * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}
