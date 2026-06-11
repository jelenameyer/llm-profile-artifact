#!/usr/bin/env python3
"""
DAST task (HF version). FLIPPED
"""

import re
from base_functions import run_scale_task 

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "tasks/jsonl_data/dast_prompts.jsonl"
TASK_FLIPPED = True


# Define valid answers
VALID_ANSWERS = ['1', '2']


SYSTEM_PROMPT = (
    "You are a helpful assistant responding to a psychometric questionnaire. "
    "For each item, choose strictly one integer 1 or 2. "
    "Do NOT explain your reasoning. Only output the number."
)

QUESTIONNAIRE_INSTRUCTION = (
    "The following questions concern information about your involvement and abuse of drugs. Drug abuse refers to (1) the use of prescribed or 'over the counter' drugs in excess of the directions and (2) any non-medical use of drugs. "
    "Carefully read each statement and provide your answer using the following scale: \n\n"
    "1 = No\n"
    "2 = Yes\n"
    "For each statement strictly respond with one number (1 or 2). Statements: "
)

ANSWER_PATTERN = re.compile(r"\b([12])\b")

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
        "name": "DAST",
        "description": "DAST scale with outlines-constrained generation",
        "valid_answers": VALID_ANSWERS,
        "output_columns": [
            "model", "item_id", "item_text", "context_mode", "flipped", "model_answer",
            *[f"logit_{ans}" for ans in VALID_ANSWERS],
            *[f"prob_{ans}" for ans in VALID_ANSWERS],
        ],
    }

