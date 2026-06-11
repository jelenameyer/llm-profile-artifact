#!/usr/bin/env python3
"""
PRI task (HF version). FLIPPED

Special: Non-flipped and flipped version are not different in the code, only in the jsonl data (pri_prompts.jsonl vs pri_prompts_flipped.jsonl)
Also: Leaving out the certainty estimations in this task here!
"""

import re
from base_functions import run_scale_task 

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "tasks/jsonl_data/pri_prompts_flipped.jsonl"
TASK_FLIPPED = True


# Define valid answers
VALID_ANSWERS = ['1', '2']


SYSTEM_PROMPT = (
    "You are a helpful assistant responding to a psychometric questionnaire. "
    "For each item, choose strictly one integer 1 or 2. "
    "Do NOT explain your reasoning. Only output the number."
)

QUESTIONNAIRE_INSTRUCTION = (
    "The following questions concern decision problems in different contexts. "
    "Since people may think differently depending on the context, there are no right or wrong answers—only individual preferences. \n"
    "For each question, select the option you would prefer without thinking too long. \n"
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
        "name": "PRI",
        "description": "PRI scale with outlines-constrained generation",
        "valid_answers": VALID_ANSWERS,
        "output_columns": [
            "model", "item_id", "item_text", "context_mode", "flipped", "model_answer",
            *[f"logit_{ans}" for ans in VALID_ANSWERS],
            *[f"prob_{ans}" for ans in VALID_ANSWERS],
        ],
    }

