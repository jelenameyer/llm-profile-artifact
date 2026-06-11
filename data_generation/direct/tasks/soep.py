#!/usr/bin/env python3
"""
SOEP task (HF version).
"""

import re
from base_functions import run_scale_task 

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "tasks/jsonl_data/soep_prompts.jsonl"
TASK_FLIPPED = False


# Define valid answers
VALID_ANSWERS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']


SYSTEM_PROMPT = (
    "You are a helpful assistant responding to a psychometric questionnaire. "
    "For each item, choose strictly one integer from 1 to 11. "
    "Do NOT explain your reasoning. Only output the number."
)

QUESTIONNAIRE_INSTRUCTION = (
    "In this survey, please indicate how willing you are to take risks in general and in specific areas of life. "
    "The scale ranges from 1 to 11, where:\n\n"
    "1 = Not at all willing to take risks\n"
    "11 = Very willing to take risks\n"
    "For each statement strictly respond with one number (1 to 11) indicating your willingness to take risks in that domain. Statements: "
)

ANSWER_PATTERN = re.compile(r"\b([1-9]|1[01])\b")

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
        "name": "SOEP",
        "description": "SOEP scale with outlines-constrained generation",
        "valid_answers": VALID_ANSWERS,
        "output_columns": [
            "model", "item_id", "item_text", "context_mode", "flipped", "model_answer",
            *[f"logit_{ans}" for ans in VALID_ANSWERS],
            *[f"prob_{ans}" for ans in VALID_ANSWERS],
        ],
    }

