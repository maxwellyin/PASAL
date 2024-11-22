import numpy as np
from .qg import tokenizer

def preprocess_answer_extraction_examples(batch, max_length=512, stride=256):
    batch_size = len(batch['question'])
    new_batch = {"input_ids": [], "attention_mask": [], "labels": []}
    
    extract_answer_prefix = "extract answer: "
    extract_answer_prefix_length = len(tokenizer.tokenize(extract_answer_prefix))

    for idx in range(batch_size):
        answer = batch["answer"][idx]
        paragraph_sentence = batch["paragraph_sentence"][idx]

        # Tokenize paragraph_sentence without the extract answer prefix
        tokenized_paragraph_sentence = tokenizer.tokenize(f"{paragraph_sentence}")

        if len(tokenized_paragraph_sentence) + extract_answer_prefix_length + 1 <= max_length:
            # If the paragraph_sentence length is within the limit, tokenize the input
            model_inputs = tokenizer(
                f"{extract_answer_prefix}{paragraph_sentence}",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
            )

            # Tokenize answer as labels
            labels = tokenizer(answer)

            # Add the tokenized inputs and labels to the new batch
            new_batch["input_ids"].append(model_inputs["input_ids"])
            new_batch["attention_mask"].append(model_inputs["attention_mask"])
            new_batch["labels"].append(labels["input_ids"])
        else:
            start_idx = 0
            while start_idx < len(tokenized_paragraph_sentence):
                end_idx = min(start_idx + max_length - extract_answer_prefix_length - 1, len(tokenized_paragraph_sentence))
                split_tokenized_paragraph_sentence = tokenized_paragraph_sentence[start_idx:end_idx]

                # Check if both <hl> tokens are present in the split_tokenized_paragraph_sentence
                if split_tokenized_paragraph_sentence.count('<hl>') == 2:
                    # Tokenize split_tokenized_paragraph_sentence
                    model_inputs = tokenizer(
                        f"{extract_answer_prefix}{split_tokenized_paragraph_sentence}",
                        max_length=max_length,
                        padding="max_length",
                        truncation=True,
                        return_attention_mask=True,
                    )

                    # Tokenize answer as labels
                    labels = tokenizer(answer)

                    # Add the tokenized inputs and labels to the new batch
                    new_batch["input_ids"].append(model_inputs["input_ids"])
                    new_batch["attention_mask"].append(model_inputs["attention_mask"])
                    new_batch["labels"].append(labels["input_ids"])

                start_idx += stride

    return new_batch



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

