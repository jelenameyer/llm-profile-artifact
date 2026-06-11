#!/usr/bin/env python3
"""
SSSV task (HF version). FLIPPED
"""

import re
import pandas as pd
from base_functions import load_task_items, generate_answer_and_logits

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "tasks/jsonl_data/sssv_prompts.jsonl"
TASK_FLIPPED = True


# Define valid answers
VALID_ANSWERS = ['1', '2']


SYSTEM_PROMPT = (
    "You are a helpful assistant responding to a psychometric questionnaire. "
    "For each item, choose strictly one integer 1 or 2. "
    "Do NOT explain your reasoning. Only output the number."
)

QUESTIONNAIRE_INSTRUCTION = (
    "Each of the items contains two choices. Please indicate which of the choices most describes your likes or the way you feel. There are no right or wrong answers. "
    "It may happen that neither statement applies or that both statements apply to you. In any case, always choose one answer option! Read both statements carefully before choosing the one that best describes you. \n\n"
    "For each statement strictly respond with one number (1 or 2). Statements: "
)

ANSWER_PATTERN = re.compile(r"\b([12])\b")

def _make_sssv_item_block(row) -> str:
    return (
        f"Item {row['id']}:\n"
        f"1. {row['s2']}\n"
        f"2. {row['s1']}\n"
        "Respond with only '1' or '2'."
    )


def run_task(model, tokenizer, outlines_model, model_key: str, data_file=None):
    df = load_task_items(data_file or DATA_PATH)
    results = []

    # no-context
    for _, row in df.iterrows():
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": QUESTIONNAIRE_INSTRUCTION + "\n\n" + _make_sssv_item_block(row),
            },
        ]
        answer, logits, probs = generate_answer_and_logits(
            model, tokenizer, outlines_model, messages, VALID_ANSWERS, model_key, ANSWER_PATTERN
        )
        results.append({
            "model": model_key,
            "item_id": row["id"],
            "item_text": f"1) {row['s2']} || 2) {row['s1']}",
            "context_mode": "no_context",
            "flipped": TASK_FLIPPED,
            "model_answer": answer,
            **logits,
            **probs,
        })

    # with-context
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i, row in df.iterrows():
        if i == 0:
            user_content = QUESTIONNAIRE_INSTRUCTION + "\n\n" + _make_sssv_item_block(row)
        else:
            user_content = _make_sssv_item_block(row)
        messages.append({"role": "user", "content": user_content})

        answer, logits, probs = generate_answer_and_logits(
            model, tokenizer, outlines_model, messages, VALID_ANSWERS, model_key, ANSWER_PATTERN
        )
        messages.append({"role": "assistant", "content": answer})
        results.append({
            "model": model_key,
            "item_id": row["id"],
            "item_text": f"1) {row['s2']} || 2) {row['s1']}",
            "context_mode": "with_context",
            "flipped": TASK_FLIPPED,
            "model_answer": answer,
            **logits,
            **probs,
        })

    return pd.DataFrame(results)


def get_task_info():
    """Task metadata for introspection."""
    return {
        "name": "SSSV",
        "description": "SSSV scale with outlines-constrained generation",
        "valid_answers": VALID_ANSWERS,
        "output_columns": [
            "model", "item_id", "item_text", "context_mode", "flipped", "model_answer",
            *[f"logit_{ans}" for ans in VALID_ANSWERS],
            *[f"prob_{ans}" for ans in VALID_ANSWERS],
        ],
    }
