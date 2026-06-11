#!/usr/bin/env python3
"""
BARRATT task (HF version). FLIPPED
"""

import re
from base_functions import run_scale_task 

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "tasks/jsonl_data/barratt_prompts.jsonl"
TASK_FLIPPED = True


# Define valid answers for BARRATT task
VALID_ANSWERS = ['1', '2', '3', '4']


SYSTEM_PROMPT = (
    "You are a helpful assistant responding to a psychometric questionnaire. "
    "For each item, choose strictly '1', '2', '3' or '4'. "
    "Do NOT explain your reasoning. Only output the digit."
)

QUESTIONNAIRE_INSTRUCTION = (
    "People differ in the ways they act and think in different situations. "
    "This is a test to measure some of the ways in which you act and think.\n\n"
    "Read each statement and answer with the appropriate number using this scale:\n"
    "1 = Almost always/Always\n"
    "2 = Often\n"
    "3 = Occasionally\n"
    "4 = Rarely/Never\n\n"
    "Do not spend too much time on any statement. "
    "Answer quickly and honestly.\n\n"
    "Respond strictly with one number (1-4)."
)

ANSWER_PATTERN = re.compile(r"\b([1234])\b")


# ============================================================
# FUNCTION TO RUN TASK
# ============================================================


def run_task(model, tokenizer, outlines_model, model_key: str, data_file=None):
    return run_scale_task(
        model=model,
        tokenizer=tokenizer,
        outlines_model=outlines_model,
        model_key=model_key,
        data_path=(data_file or DATA_PATH),
        system_prompt=SYSTEM_PROMPT,
        questionnaire_instruction=QUESTIONNAIRE_INSTRUCTION,
        valid_answers=VALID_ANSWERS,
        answer_pattern=ANSWER_PATTERN,
        task_flipped=TASK_FLIPPED,
    )




def get_task_info():
    """Task metadata for introspection."""
    return {
        "name": "BARRATT",
        "description": "BARRATT impulsiveness scale with outlines-constrained generation",
        "valid_answers": VALID_ANSWERS,
        "output_columns": [
            "model", "item_id", "item_text", "context_mode", "flipped", "model_answer",
            "logit_1", "logit_2", "logit_3", "logit_4",
            "prob_1", "prob_2", "prob_3", "prob_4",
        ],
    }


