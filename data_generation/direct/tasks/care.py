#!/usr/bin/env python3
"""
CARE task (HF version).
Special: No flipped version, since numbers have ordinal meaning.
"""

import re
from base_functions import run_scale_task 

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "tasks/jsonl_data/care_prompts.jsonl"
TASK_FLIPPED = False


# Define valid answers for CARE task (integer range 0-100)
VALID_ANSWERS = [str(i) for i in range(101)]


SYSTEM_PROMPT = (
    "You are a helpful assistant responding to a psychometric questionnaire. "
    "For each item, choose strictly one integer from 0 to 100. "
    "Do NOT explain your reasoning. Only output the number."
)

QUESTIONNAIRE_INSTRUCTION = (
    "The following questions concern information about your behaviour in the past six months. "
    "Carefully read each statement and provide your answer using a single number.\n\n"
    "For each statement, strictly respond with one integer from 0 to 100.\n"
    "Statements: Please indicate how often in the past six months did you:"
)

ANSWER_PATTERN = re.compile(r"\b(100|[1-9]?\d)\b")


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
        "name": "CARE",
        "description": "CARE scale with outlines-constrained generation",
        "valid_answers": VALID_ANSWERS,
        "output_columns": [
            "model", "item_id", "item_text", "context_mode", "flipped", "model_answer",
            *[f"logit_{ans}" for ans in VALID_ANSWERS],
            *[f"prob_{ans}" for ans in VALID_ANSWERS],
        ],
    }

