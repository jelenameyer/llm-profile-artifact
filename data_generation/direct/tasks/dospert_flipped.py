#!/usr/bin/env python3
"""
DOSPERT task (HF version). FLIPPED
"""

import re
from base_functions import run_scale_task 

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "tasks/jsonl_data/dospert_prompts.jsonl"
TASK_FLIPPED = True


# Define valid answers
VALID_ANSWERS = ['1', '2', '3', '4', '5']


SYSTEM_PROMPT = (
    "You are a helpful assistant responding to a psychometric questionnaire. "
    "For each item, choose strictly one integer from 1 to 5. "
    "Do NOT explain your reasoning. Only output the number."
)

QUESTIONNAIRE_INSTRUCTION = (
    "For each of the following statements, please indicate the likelihood that you would engage in the described activity or behavior if you were to find yourself in that situation. "
    "Provide a rating from 'Very likely' to 'Very unlikely', using the following scale:\n\n"
    "1 = Very likely\n"
    "2 = Likely\n"
    "3 = Not sure\n"
    "4 = Unlikely\n\n"
    "4 = Very unlikely\n\n"
    "For each statement strictly respond with one number (1 to 5). Statements: "
)

ANSWER_PATTERN = re.compile(r"\b([12345])\b")


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
        "name": "DOSPERT",
        "description": "DOSPERT scale with outlines-constrained generation",
        "valid_answers": VALID_ANSWERS,
        "output_columns": [
            "model", "item_id", "item_text", "context_mode", "flipped", "model_answer",
            *[f"logit_{ans}" for ans in VALID_ANSWERS],
            *[f"prob_{ans}" for ans in VALID_ANSWERS],
        ],
    }

