#!/usr/bin/env python3
"""
MPL task (HF version).

Two-option behavioral choices with per-item randomized letter labels.
The output keeps a canonical mapping so results remain interpretable:
which underlying option (s1 vs s2) was chosen.
"""

import random
import re
import string
import pandas as pd
from base_functions import load_task_items, generate_answer_and_logits


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "tasks/jsonl_data/mpl_prompts.jsonl"
RANDOMIZATION_SEED = 13
TASK_FLIPPED = False

SYSTEM_PROMPT = (
    "You are a helpful assistant responding to decision tasks. "
    "For each item, choose exactly one option label. "
    "Do NOT explain your reasoning. Only output the label."
)

QUESTIONNAIRE_INSTRUCTION = (
    "You will be presented with several pairs of lotteries in each trial. Each lottery offers specific chances of winning or losing points. Each point equates to 0.075 CHF or 0.05 EUR.\n"
    "Your task is to choose between the two lotteries in each round. Each choice affects your potential earnings.\n"
    "The money earned in this study will be added to or subtracted from your starting bonus of 15 CHF or 10 EUR. In the two most extreme cases, you can either double or entirely lose this amount. "
    "No immediate feedback will be provided regarding the outcomes of your choices. \n"
    "Respond strictly with one letter label shown for the current item."
)

# --- helpers ---

def _labels_for_item(gamble_id: str):
    """
    Deterministic randomization per gamble_id.
    Ensures reproducibility across reruns.
    """
    rng = random.Random(f"{RANDOMIZATION_SEED}:{gamble_id}")
    return rng.sample(list(string.ascii_uppercase), 2)


def _make_item_block(row, label_s1: str, label_s2: str) -> str:
    return (
        f"Item {row['id']}:\n"
        f"Lottery {label_s1}: {str(row['s1']).strip()}\n"
        f"Lottery {label_s2}: {str(row['s2']).strip()}\n"
        f"Respond with only '{label_s1}' or '{label_s2}'."
    )


def _build_record(model_key, row, context_mode, answer, logits, probs, label_s1, label_s2):
    if answer == label_s1:
        chosen_option = "s1"
    elif answer == label_s2:
        chosen_option = "s2"
    else:
        chosen_option = "ERROR"

    return {
        "model": model_key,
        "gamble_id": row["gamble_id"],
        "source_row_id": row["id"],
        "item_text": f"s1) {str(row['s1']).strip()} || s2) {str(row['s2']).strip()}",
        "context_mode": context_mode,
        "flipped": TASK_FLIPPED,
        "label_s1": label_s1,
        "label_s2": label_s2,
        "model_answer_label": answer,
        "model_answer_option": chosen_option,
        "logit_s1": logits.get(f"logit_{label_s1}"),
        "logit_s2": logits.get(f"logit_{label_s2}"),
        "prob_s1": probs.get(f"prob_{label_s1}"),
        "prob_s2": probs.get(f"prob_{label_s2}"),
    }


def run_task(model, tokenizer, outlines_model, model_key: str, data_file=None, context_modes=("no_context", "with_context")):
    df = load_task_items(data_file or DATA_PATH)
    results = []

    # Precompute deterministic label mapping so both context modes use identical labels.
    label_map = {}
    for _, row in df.iterrows():
        gamble_id = str(row["gamble_id"])
        label_s1, label_s2 = _labels_for_item(gamble_id)
        label_map[gamble_id] = (label_s1, label_s2)

    # ========================================================
    # MODE 1: NO CONTEXT
    # ========================================================
    for _, row in df.iterrows():
        label_s1, label_s2 = label_map[str(row["gamble_id"])]
        valid_answers = [label_s1, label_s2]
        answer_pattern = re.compile(rf"\b({re.escape(label_s1)}|{re.escape(label_s2)})\b")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": QUESTIONNAIRE_INSTRUCTION + "\n\n" + _make_item_block(row, label_s1, label_s2),
            },
        ]

        answer, logits, probs = generate_answer_and_logits(
            model, tokenizer, outlines_model, messages, valid_answers, model_key, answer_pattern
        )
        results.append(
            _build_record(model_key, row, "no_context", answer, logits, probs, label_s1, label_s2)
        )

    if "with_context" not in context_modes:
        return pd.DataFrame(results)

    # ========================================================
    # MODE 2: WITH CONTEXT
    # ========================================================
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i, row in df.iterrows():
        label_s1, label_s2 = label_map[str(row["gamble_id"])]
        valid_answers = [label_s1, label_s2]
        answer_pattern = re.compile(rf"\b({re.escape(label_s1)}|{re.escape(label_s2)})\b")

        if i == 0:
            user_content = QUESTIONNAIRE_INSTRUCTION + "\n\n" + _make_item_block(row, label_s1, label_s2)
        else:
            user_content = _make_item_block(row, label_s1, label_s2)
        messages.append({"role": "user", "content": user_content})

        answer, logits, probs = generate_answer_and_logits(
            model, tokenizer, outlines_model, messages, valid_answers, model_key, answer_pattern
        )
        messages.append({"role": "assistant", "content": answer})
        results.append(
            _build_record(model_key, row, "with_context", answer, logits, probs, label_s1, label_s2)
        )

    return pd.DataFrame(results)


def get_task_info():
    """Task metadata for introspection."""
    return {
        "name": "MPL",
        "description": "MPL gamble-choice task with randomized option labels and canonical s1/s2 mapping",
        "valid_answers": "randomized letters per item (mapped back to s1/s2 in output)",
        "output_columns": [
            "model",
            "gamble_id",
            "source_row_id",
            "item_text",
            "context_mode",
            "flipped",
            "label_s1",
            "label_s2",
            "model_answer_label",
            "model_answer_option",
            "logit_s1",
            "logit_s2",
            "prob_s1",
            "prob_s2",
        ],
    }
