#!/usr/bin/env python3
"""
GABS task (HF version). FLIPPED
Special: overwrite base function, to include different scale ranges for first item vs. rest of the items
"""
import re
import pandas as pd
from base_functions import load_task_items, make_item_block, generate_answer_and_logits

# ============================================================
# CONFIG
# ============================================================


DATA_PATH = "tasks/jsonl_data/gabs_prompts.jsonl"
TASK_FLIPPED = True

SYSTEM_PROMPT = (
    "You are a helpful assistant responding to a psychometric questionnaire. "
    "For each item, choose strictly one integer. "
    "Do NOT explain your reasoning. Only output the number."
)

QUESTIONNAIRE_INSTRUCTION = (
    "In this survey, please answer the following questions concerning your gambling behaviour. \n"
    "Item 0 uses: 1 = No, 2 = Yes.\n"
    "Items 1-15 use: 1 = Complete disagreement, 2 = Disagreement, 3 = Agreement, 4 = Complete agreement.\n"
    "For each item, respond with only one number."
)

PATTERN_12 = re.compile(r"\b([12])\b")
PATTERN_14 = re.compile(r"\b([1234])\b")

VALID_ANSWERS_ITEM0 = ["1", "2"]
VALID_ANSWERS_MAIN = ["1", "2", "3", "4"]

# --- helper ---
def _scale_for_row(row):
    if int(row["id"]) == 0:
        return ["1", "2"], PATTERN_12
    return ["1", "2", "3", "4"], PATTERN_14


# =========================================================================================
# FUNCTION TO RUN TASK - overwrites base functions since needs different scales for item 0.
# =========================================================================================

def run_task(model, tokenizer, outlines_model, model_key: str, data_file=None):
    df = load_task_items(data_file or DATA_PATH)
    results = []

    # no-context
    for _, row in df.iterrows():
        valid_answers, answer_pattern = _scale_for_row(row)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": QUESTIONNAIRE_INSTRUCTION + "\n\n" + make_item_block(row)},
        ]
        answer, logits, probs = generate_answer_and_logits(
            model, tokenizer, outlines_model, messages, valid_answers, model_key, answer_pattern
        )
        results.append({
            "model": model_key,
            "item_id": row["id"],
            "item_text": row["item"],
            "context_mode": "no_context",
            "flipped": TASK_FLIPPED,
            "valid_answers": ",".join(valid_answers),
            "model_answer": answer,
            **logits, **probs
        })

    # with-context
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i, row in df.iterrows():
        user_content = QUESTIONNAIRE_INSTRUCTION + "\n\n" + make_item_block(row) if i == 0 else make_item_block(row)
        messages.append({"role": "user", "content": user_content})
        valid_answers, answer_pattern = _scale_for_row(row)
        answer, logits, probs = generate_answer_and_logits(
            model, tokenizer, outlines_model, messages, valid_answers, model_key, answer_pattern
        )
        messages.append({"role": "assistant", "content": answer})
        

        results.append({
            "model": model_key,
            "item_id": row["id"],
            "item_text": row["item"],
            "context_mode": "with_context",
            "flipped": TASK_FLIPPED,
            "valid_answers": ",".join(valid_answers),
            "model_answer": answer,
            **logits, **probs
        })

    return pd.DataFrame(results)


def get_task_info():
    """Task metadata for introspection."""
    return {
        "name": "GABS",
        "description": "GABS scale with outlines-constrained generation and different answer ranges per items",
        "valid_answers": {
    "item_0": VALID_ANSWERS_ITEM0,
    "items_1_15": VALID_ANSWERS_MAIN,
  },
  "output_columns": [
    "model", "item_id", "item_text", "context_mode", "flipped", "valid_answers", "model_answer",
    "logit_1", "logit_2", "logit_3", "logit_4",
    "prob_1", "prob_2", "prob_3", "prob_4",
  ],
    }
