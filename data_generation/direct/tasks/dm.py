#!/usr/bin/env python3
"""
DM task (HF version).
"""

import re
from base_functions import run_scale_task 

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "tasks/jsonl_data/dm_prompts.jsonl"
TASK_FLIPPED = False


# Define valid answers
VALID_ANSWERS = ['1', '2', '3', '4']


SYSTEM_PROMPT = (
    "You are a helpful assistant responding to a psychometric questionnaire. "
    "For each item, choose strictly one integer 1 to 4. "
    "Do NOT explain your reasoning. Only output the number."
)

QUESTIONNAIRE_INSTRUCTION = (
    "The following questions concern your behavior in the last 6 months. "
    "Please indicate how often you have engaged in the following behaviors during the respective period of time. \n\n"
    "Please use the scale from 1 to 3. Please use 4 = 'not applicable' whenever the activity/behavior does not apply to you (e.g., if you do not drive a car). \n"
    "Carefully read each statement and provide your answer using the following scale: \n"
    "1 = Never\n"
    "2 = One or two times\n"
    "3 = Several times\n"
    "4 = Not applicable\n"
    "For each statement strictly respond with one number (1 to 4). Statements: "
)

ANSWER_PATTERN = re.compile(r"\b([1234])\b")

# ============================================================
# FUNCTION TO RUN TASK
# ============================================================


def run_task(model, tokenizer, outlines_model, model_key: str, data_file=None, context_modes=("no_context", "with_context")):
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
        context_modes=context_modes,
    )


def get_task_info():
    """Task metadata for introspection."""
    return {
        "name": "DM",
        "description": "Dm scale with outlines-constrained generation",
        "valid_answers": VALID_ANSWERS,
        "output_columns": [
            "model", "item_id", "item_text", "context_mode", "flipped", "model_answer",
            *[f"logit_{ans}" for ans in VALID_ANSWERS],
            *[f"prob_{ans}" for ans in VALID_ANSWERS],
        ],
    }

