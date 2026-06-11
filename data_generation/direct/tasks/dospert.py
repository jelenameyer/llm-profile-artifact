#!/usr/bin/env python3
"""
DOSPERT task (HF version).
"""

import re
from base_functions import run_scale_task 

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "tasks/jsonl_data/dospert_prompts.jsonl"
TASK_FLIPPED = False


# Define valid answers
VALID_ANSWERS = ['1', '2', '3', '4', '5']


SYSTEM_PROMPT = (
    "You are a helpful assistant responding to a psychometric questionnaire. "
    "For each item, choose strictly one integer from 1 to 5. "
    "Do NOT explain your reasoning. Only output the number."
)

QUESTIONNAIRE_INSTRUCTION = (
    "For each of the following statements, please indicate the likelihood that you would engage in the described activity or behavior if you were to find yourself in that situation. "
    "Provide a rating from 'Very unlikely' to 'Very likely', using the following scale:\n\n"
    "1 = Very unlikely\n"
    "2 = Unlikely\n"
    "3 = Not sure\n"
    "4 = Likely\n\n"
    "4 = Very likely\n\n"
    "For each statement strictly respond with one number (1 to 5). Statements: "
)

ANSWER_PATTERN = re.compile(r"\b([12345])\b")


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
        "name": "DOSPERT",
        "description": "DOSPERT scale with outlines-constrained generation",
        "valid_answers": VALID_ANSWERS,
        "output_columns": [
            "model", "item_id", "item_text", "context_mode", "flipped", "model_answer",
            *[f"logit_{ans}" for ans in VALID_ANSWERS],
            *[f"prob_{ans}" for ans in VALID_ANSWERS],
        ],
    }

