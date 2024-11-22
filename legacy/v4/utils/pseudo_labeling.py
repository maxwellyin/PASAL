from __future__ import annotations

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from .runtime import pipeline_device
from .util import PROMPT_LENGTH, format_for_lmqg


def filter_answer_in_context(example):
    return example["answers"]["text"][0] in example["context"]


def filter_score_threshold(threshold):
    return lambda example: example["score"] > threshold


def make_prompt_qa_pseudo_labeler(tokenizer, qa_model, qg_model: str):
    device = pipeline_device()
    question_generation = pipeline("text2text-generation", qg_model, device=device)
    question_answerer = pipeline("text2text-generation", model=qa_model, tokenizer=tokenizer, device=device)

    def create_pseudo_label(example):
        generated_question = question_generation(format_for_lmqg(example))[0]["generated_text"]
        qa_input = f"{tokenizer.pad_token * PROMPT_LENGTH}question: {generated_question} context: {example['context']}"
        predicted_answer = question_answerer(qa_input)[0]["generated_text"]
        answers = {"text": [predicted_answer], "answer_start": example["answers"]["answer_start"]}
        return {"question": generated_question, "answers": answers}

    return create_pseudo_label


def make_nonprompt_seq2seq_qa_pseudo_labeler(tokenizer, qa_model_checkpoint: str, qg_model: str):
    device = pipeline_device()
    question_generation = pipeline("text2text-generation", qg_model, device=device)
    qa_model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_checkpoint)
    question_answerer = pipeline("text2text-generation", model=qa_model, tokenizer=tokenizer, device=device)

    def create_pseudo_label(example):
        generated_question = question_generation(format_for_lmqg(example))[0]["generated_text"]
        qa_input = f"question: {generated_question} context: {example['context']}"
        predicted_answer = question_answerer(qa_input)[0]["generated_text"]
        answers = {"text": [predicted_answer], "answer_start": example["answers"]["answer_start"]}
        return {"question": generated_question, "answers": answers}

    return create_pseudo_label


def make_extract_qa_pseudo_labeler(qa_checkpoint: str):
    device = pipeline_device()
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_checkpoint)
    question_answerer = pipeline("question-answering", model=qa_checkpoint, tokenizer=qa_tokenizer, device=device)

    def create_pseudo_label(example):
        output = question_answerer(question=example["question"], context=example["context"])
        answers = {"text": [output["answer"]], "answer_start": [output["start"]]}
        return {"answers": answers, "score": output["score"]}

    return create_pseudo_label


def make_qg_extract_qa_pseudo_labeler(qg_model: str, qa_checkpoint: str):
    device = pipeline_device()
    question_generation = pipeline("text2text-generation", qg_model, device=device)
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_checkpoint)
    question_answerer = pipeline("question-answering", model=qa_checkpoint, tokenizer=qa_tokenizer, device=device)

    def create_pseudo_label(example):
        generated_question = question_generation(format_for_lmqg(example))[0]["generated_text"]
        output = question_answerer(question=generated_question, context=example["context"])
        answers = {"text": [output["answer"]], "answer_start": [output["start"]]}
        return {"question": generated_question, "answers": answers, "score": output["score"]}

    return create_pseudo_label
