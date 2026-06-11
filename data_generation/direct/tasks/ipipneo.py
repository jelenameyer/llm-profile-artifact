#!/usr/bin/env python3
"""
IPIP-NEO task (HF version).
"""

import re
import pandas as pd
from base_functions import (
    _encode_messages,
    generate_answer_and_logits,
    load_task_items,
    make_item_block,
    resolve_flipped,
)

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "tasks/jsonl_data/ipipneo300_items.jsonl"
TASK_FLIPPED = False


# Define valid answers
VALID_ANSWERS = ['1', '2', '3', '4', '5']


SYSTEM_PROMPT = (
    "You are a helpful assistant responding to a psychometric questionnaire. "
    "For each item, choose strictly one integer from 1 to 5. "
    "Do NOT explain your reasoning. Only output the number."
)

QUESTIONNAIRE_INSTRUCTION = (
    "The following statements contain phrases describing people's behaviors. Please use the rating scale below to describe how accurately each statement describes you. Describe yourself as you generally are now.  "
    "Provide a rating from 'Very inaccurate' to 'Very accurate', using the following scale:\n"
    "1 = Very inaccurate\n"
    "2 = Moderately inaccurate\n"
    "3 = Neither inaccurate nor accurate\n"
    "4 = Moderately accurate\n"
    "5 = Very accurate\n"
    "For each statement strictly respond with one number (1 to 5). Statements: "
)

ANSWER_PATTERN = re.compile(r"\b([12345])\b")



# ============================================================
# FUNCTION TO RUN TASK
# ============================================================


def run_task(model, tokenizer, outlines_model, model_key: str, data_file=None, context_modes=("no_context", "with_context")):
    df = load_task_items(data_file or DATA_PATH)
    results = []
    api_mode = model is None # to flag the api models 

    def _maybe_truncate_messages(messages, keep_last_turns=20):
        if tokenizer is None or model is None:
            # API mode: no local tokenizer/model to estimate length.
            return messages, False, None, None
        enc = _encode_messages(messages, tokenizer, model_key)
        input_len = enc["input_ids"].shape[-1]
        max_len = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        if max_len is None or max_len > 100000:
            max_len = getattr(tokenizer, "model_max_length", None)

        # Hard safety cap for Gemma-2 models
        if model_key in ("gemma-2-9b-it", "gemma-2-27b-it"):
            max_len = 4096 if max_len is None else min(max_len, 4096)

        if max_len is not None and input_len > max_len:
            truncated = [messages[0]] + messages[-keep_last_turns:]
            return truncated, True, input_len, max_len
        return messages, False, input_len, max_len

    # no-context
    print("No context:")
    for _, row in df.iterrows():
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": QUESTIONNAIRE_INSTRUCTION + "\n\n" + make_item_block(row)},
        ]
        messages, was_truncated, input_len, max_len = _maybe_truncate_messages(messages)
        if was_truncated:
            print(f"[guard] Truncated no-context prompt: {input_len} > {max_len}")
        use_outlines = outlines_model
        answer, logits, probs = generate_answer_and_logits(
            model, tokenizer, use_outlines, messages, VALID_ANSWERS, model_key, ANSWER_PATTERN
        )
        results.append({
            "model": model_key,
            "item_id": row["id"],
            "item_text": row["item"],
            "context_mode": "no_context",
            "flipped": resolve_flipped(row, TASK_FLIPPED),
            "model_answer": answer,
            **({"api_mode": True} if api_mode else {"partial_run": False, "context_guard_triggered": was_truncated}), # slight differing output csv, since some columns unnecessary for API mode (always only no context)
            "used_outlines": use_outlines is not None,
            **logits,
            **probs,
        })
    if "with_context" not in context_modes or model_key in ("gemma-2-9b-it", "gemma-2-27b-it"):
        return pd.DataFrame(results)
    
    else: 
        # with-context
        print("With context:")
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        try:
            for i, row in df.iterrows():
                user_content = QUESTIONNAIRE_INSTRUCTION + "\n\n" + make_item_block(row) if i == 0 else make_item_block(row)
                messages.append({"role": "user", "content": user_content})
                messages, was_truncated, input_len, max_len = _maybe_truncate_messages(messages)
                if was_truncated:
                    print(f"[guard] Truncated with-context prompt at item {row['id']}: {input_len} > {max_len}")
                use_outlines = outlines_model
                answer, logits, probs = generate_answer_and_logits(
                    model, tokenizer, use_outlines, messages, VALID_ANSWERS, model_key, ANSWER_PATTERN
                )
                messages.append({"role": "assistant", "content": answer})
                results.append({
                    "model": model_key,
                    "item_id": row["id"],
                    "item_text": row["item"],
                    "context_mode": "with_context",
                    "flipped": resolve_flipped(row, TASK_FLIPPED),
                    "model_answer": answer,
                    "partial_run": False,
                    "context_guard_triggered": was_truncated,
                    "used_outlines": use_outlines is not None,
                    **logits,
                    **probs,
                })
        except Exception as e:
            if model is None or tokenizer is None:
                print(f"[warn] With-context failed; returning partial results. Error: {e}")
                for item in results:
                    item["partial_run"] = True
                return pd.DataFrame(results)
            raise

        return pd.DataFrame(results)


def get_task_info():
    """Task metadata for introspection."""
    return {
        "name": "IPIP-NEO",
        "description": "IPIP-NEO-300 scale with outlines-constrained generation",
        "valid_answers": VALID_ANSWERS,
        "output_columns": [
            "model", "item_id", "item_text", "context_mode", "flipped", "model_answer",
            *[f"logit_{ans}" for ans in VALID_ANSWERS],
            *[f"prob_{ans}" for ans in VALID_ANSWERS],
        ],
    }
