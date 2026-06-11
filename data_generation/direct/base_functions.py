#!/usr/bin/env python3
"""
base functions to run all tasks with model manager

Runs BOTH:
1) no-context mode (each item independently)
2) with-context mode (growing conversation)
(extra: for api models, no context mode only)

Results are labeled so they can be compared later.
Using outlines for valid answer generation + separate logit extraction. 
(extra: for api models only model answer without outlines and no logits.)
"""

import json
import re
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
import outlines



# ============================================================
# UTILITIES
# ============================================================

def clear_generator_cache():
    _generator_cache.clear()


def load_task_items(DATA_PATH) -> pd.DataFrame:
    rows = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def get_flipped_flag(row) -> bool:
    """
    Return normalized boolean flipped flag from a task row.
    Defaults to False when absent.
    """
    value = row.get("flipped", False)
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return value
    return False


def resolve_flipped(row, task_flipped=None) -> bool:
    """
    Resolve flipped flag.
    - If task_flipped is set (True/False), use it for all rows in this task run.
    - Otherwise fall back to row-level 'flipped' field.
    """
    if task_flipped is None:
        return get_flipped_flag(row)
    return bool(task_flipped)


def make_item_block(row) -> str:
    return (
        f"Item {row['id']}:\n"
        f"{row['item']}\n"
    )


def _supports_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def prepare_prompt(messages, tokenizer, model_name):
    """Convert messages to prompt string based on model type."""
    if (
        "bloomz" in model_name.lower()
        or model_name.lower().startswith("gemma-2")
        or not _supports_chat_template(tokenizer)
    ):
        # Plain text mode
        return "\n".join(m["content"] for m in messages)
    else:
        # Chat template mode
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


# ============================================================
# LOGIT EXTRACTION (separate from generation)
# ============================================================

def _encode_messages(messages, tokenizer, model_name):
    """Tokenize using same path as generation."""
    # Convert to string first (like generation does)
    if (
        "bloomz" in model_name.lower()
        or model_name.lower().startswith("gemma-2")
        or not _supports_chat_template(tokenizer)
    ):
        prompt = "\n".join(m["content"] for m in messages)
    else:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    # Then tokenize the string (like outlines does)
    return tokenizer(prompt, return_tensors="pt")


def _build_answer_regex_pattern(valid_answers):
    """Build regex alternatives used by outlines generation."""
    variants = []
    for ans in valid_answers:
        variants.extend(_answer_variants(ans))
    # Prefer longer strings first (e.g., "\n10" before "1") for stable matching.
    variants = sorted(set(variants), key=len, reverse=True)
    # Keep regex unanchored for Outlines compatibility across providers.
    return rf"(?:{'|'.join(re.escape(v) for v in variants)})"


def _answer_variants(answer):
    """Allowed surface forms for one answer alternative."""
    base = str(answer)
    variants = [base, " " + base, "\n" + base]
    seen = set()
    out = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _candidate_variants(answer, valid_answers):
    """Generate answer variants aligned with outlines regex alternatives."""
    if answer not in valid_answers:
        return []
    return _answer_variants(answer)

def _score_answer_variant(
    model,
    base_input_ids,
    base_attention_mask,
    prefix_last_logits,
    prefix_last_log_probs,
    variant_token_ids,
):
    """
    Score one candidate variant using robust full-sequence teacher forcing.

    Returns:
        tuple: (mean_token_logit, sum_token_logprob)
    """
    device = model.device
    token_logits = [prefix_last_logits[variant_token_ids[0]].item()]
    token_logprobs = [prefix_last_log_probs[variant_token_ids[0]].item()]

    if len(variant_token_ids) > 1:
        variant_ids = torch.tensor([variant_token_ids], device=device, dtype=torch.long)
        full_input_ids = torch.cat([base_input_ids, variant_ids], dim=-1)
        append_attention = torch.ones(
            (1, variant_ids.shape[-1]), device=device, dtype=base_attention_mask.dtype
        )
        full_attention_mask = torch.cat([base_attention_mask, append_attention], dim=-1)

        with torch.inference_mode():
            outputs = model(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                use_cache=False,
            )

        cont_logits = outputs.logits[0, -len(variant_token_ids):-1, :]
        cont_log_probs = F.log_softmax(cont_logits, dim=-1)

        for j in range(1, len(variant_token_ids)):
            tok_id = variant_token_ids[j]
            pos = j - 1
            token_logits.append(cont_logits[pos, tok_id].item())
            token_logprobs.append(cont_log_probs[pos, tok_id].item())

    mean_token_logit = float(sum(token_logits) / len(token_logits))
    sum_token_logprob = float(sum(token_logprobs))
    return mean_token_logit, sum_token_logprob


def extract_logits_and_probs(model, tokenizer, messages, valid_answers, model_name):
    """
    Score all valid answers from the answer position.

    Scores valid answers from the answer position using exact answer strings
    (aligned with outlines regex alternatives).

    For multi-token answers, scores are computed over the full token sequence.
    
    Returns:
        tuple: (logits_dict, probs_dict)
    """
    enc = _encode_messages(messages, tokenizer, model_name)
    base_input_ids = enc["input_ids"].to(model.device)
    if "attention_mask" in enc:
        base_attention_mask = enc["attention_mask"].to(model.device)
    else:
        base_attention_mask = torch.ones_like(base_input_ids, device=model.device)

    with torch.inference_mode():
        prefix_outputs = model(
            input_ids=base_input_ids,
            attention_mask=base_attention_mask,
            use_cache=False,
        )

    prefix_last_logits = prefix_outputs.logits[0, -1, :]
    prefix_last_log_probs = F.log_softmax(prefix_last_logits, dim=-1)

    answer_logit_scores = []
    answer_sequence_scores = []

    for ans in valid_answers:
        variant_logits = []
        variant_seq_scores = []

        for variant in _candidate_variants(ans, valid_answers):
            variant_token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(variant_token_ids) == 0:
                continue
            mean_tok_logit, seq_logprob = _score_answer_variant(
                model=model,
                base_input_ids=base_input_ids,
                base_attention_mask=base_attention_mask,
                prefix_last_logits=prefix_last_logits,
                prefix_last_log_probs=prefix_last_log_probs,
                variant_token_ids=variant_token_ids,
            )
            variant_logits.append(mean_tok_logit)
            variant_seq_scores.append(seq_logprob)

        if not variant_seq_scores:
            raise ValueError(f"Answer '{ans}' has no valid tokenization variants")

        # Logit-style summary for reporting: best variant's mean token logit.
        best_idx = max(range(len(variant_seq_scores)), key=lambda i: variant_seq_scores[i])
        answer_logit_scores.append(variant_logits[best_idx])

        # Probability mass across variants: logsumexp over variant sequence logprobs.
        answer_sequence_scores.append(torch.logsumexp(torch.tensor(variant_seq_scores), dim=0).item())

    answer_logits = torch.tensor(answer_logit_scores)
    answer_probs = F.softmax(torch.tensor(answer_sequence_scores), dim=0)
    
    # Build dictionaries
    logits_dict = {
        f'logit_{ans}': answer_logits[i].item()
        for i, ans in enumerate(valid_answers)
    }
    
    probs_dict = {
        f'prob_{ans}': answer_probs[i].item()
        for i, ans in enumerate(valid_answers)
    }
    
    return logits_dict, probs_dict


# ============================================================
# GENERATION WITH OUTLINES (OPTIMIZED WITH CACHING)
# ============================================================

# Global cache for generators (per model + answer pattern)
_generator_cache = {}

def get_or_create_generator(outlines_model, valid_answers):
    """
    Get cached generator or create new one.
    Avoids recreating generator for same answer pattern.
    """    
    # Create cache key from model id + answer pattern
    cache_key = (
        id(outlines_model._model) if hasattr(outlines_model, "_model") else id(outlines_model),
        tuple(valid_answers),
    )
    
    if cache_key not in _generator_cache:
        regex_pattern = _build_answer_regex_pattern(valid_answers)
        generator = outlines.Generator(
            outlines_model,
            output_type=outlines.regex(regex_pattern),
        )
        _generator_cache[cache_key] = generator
    
    return _generator_cache[cache_key]

def generate_with_outlines(outlines_model, messages, tokenizer, valid_answers, model_name):
    """Generate using outlines with cached generator"""
    prompt = prepare_prompt(messages, tokenizer, model_name)
    generator = get_or_create_generator(outlines_model, valid_answers)
    
    # Forcing greedy generation explicitly
    answer = generator(
        prompt,
        do_sample=False,
        temperature=None,  # Explicitly disable temperature
        top_p=None,        # Disable nucleus sampling
        top_k=None,        # Disable top-k
        max_new_tokens=8,
    )

    clean_answer = answer.strip()
    if clean_answer in valid_answers:
        return clean_answer

    raise ValueError(f"Outlines produced invalid constrained output: '{clean_answer}'")


def _extract_valid_answer_from_text(text, valid_answers):
    """Extract earliest exact valid answer occurrence from text."""
    candidates = sorted(valid_answers, key=len, reverse=True)
    matches = []
    for ans in candidates:
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(ans)}(?![A-Za-z0-9])")
        m = pattern.search(text)
        if m:
            matches.append((m.start(), ans))
    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    return matches[0][1]


def generate_fallback(model, tokenizer, messages, model_name, ANSWER_PATTERN=None, valid_answers=None):
    """
    Fallback generation without outlines (unconstrained).
    
    Returns:
        str: The generated answer (may be invalid)
    """
    enc = _encode_messages(messages, tokenizer, model_name)
    
    enc = {k: v.to(model.device) for k, v in enc.items()}
    if "attention_mask" not in enc:
        enc["attention_mask"] = torch.ones_like(enc["input_ids"], device=model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=5,
            do_sample=False,
            temperature=None,  # Greedy
            use_cache=False,
        )
    
    generated_tokens = outputs[0][enc["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Try to clean answer
    clean_text = text.strip()
    if ANSWER_PATTERN is not None:
        m = ANSWER_PATTERN.search(clean_text)
        if m:
            return m.group(1)
    if valid_answers is not None:
        extracted = _extract_valid_answer_from_text(clean_text, valid_answers)
        if extracted is not None:
            return extracted
    return clean_text if clean_text else "ERROR"

# ============================================================
# UNIFIED GENERATION FUNCTION
# ============================================================

def generate_answer_and_logits(
    model,
    tokenizer,
    outlines_model,
    messages,
    valid_answers,
    model_name,
    ANSWER_PATTERN=None
):
    """
    Main generation function that:
    1. Generates answer (with outlines if available, else fallback)
    2. Extracts logits for all valid answers
    
    Returns:
        tuple: (answer, logits_dict, probs_dict)
    """
    # Step 1: Generate answer
    if outlines_model is not None:
        try:
            answer = generate_with_outlines(
                outlines_model, messages, tokenizer, valid_answers, model_name
            )
        except Exception as e:
            print(f"Outlines failed ({e}), using fallback")
            answer = generate_fallback(
                model, tokenizer, messages, model_name, ANSWER_PATTERN, valid_answers
            )
            print(f"  → Fallback generated: {answer}")
    else:
        answer = generate_fallback(
            model, tokenizer, messages, model_name, ANSWER_PATTERN, valid_answers
        )
        print(f"Unconstrained generated: {answer}")
    
    # Step 2: Extract logits (always, regardless of generation method)
    logits_dict, probs_dict = extract_logits_and_probs(
        model, tokenizer, messages, valid_answers, model_name
    )

    # Step 3: Hard safety clamp: ensure final answer is always one of valid_answers.
    if answer not in valid_answers:
        best_answer = max(valid_answers, key=lambda a: probs_dict[f"prob_{a}"])
        print(f"Invalid answer '{answer}' -> clamped to highest-prob valid answer '{best_answer}'")
        answer = best_answer
    
    return answer, logits_dict, probs_dict



# ============================================================
# TASK ENTRY POINT
# ============================================================


def run_scale_task(
    model,
    tokenizer,
    outlines_model,
    model_key,
    data_path,
    system_prompt,
    questionnaire_instruction,
    valid_answers,
    answer_pattern=None,
    task_flipped=None,
    context_modes=("no_context", "with_context"),
):
    df = load_task_items(data_path)
    results = []

    # no-context
    print("No context:")
    for _, row in df.iterrows():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": questionnaire_instruction + "\n\n" + make_item_block(row)},
        ]
        answer, logits, probs = generate_answer_and_logits(
            model, tokenizer, outlines_model, messages, valid_answers, model_key, answer_pattern
        )
        results.append({
            "model": model_key,
            "item_id": row["id"],
            "item_text": row["item"],
            "context_mode": "no_context",
            "flipped": resolve_flipped(row, task_flipped),
            "model_answer": answer,
            "partial_run": False,
            **logits,
            **probs,
        })

    if "with_context" not in context_modes:
        return pd.DataFrame(results)

    # with-context
    print("With context:")
    messages = [{"role": "system", "content": system_prompt}]
    try:
        for i, row in df.iterrows():
            user_content = questionnaire_instruction + "\n\n" + make_item_block(row) if i == 0 else make_item_block(row)
            messages.append({"role": "user", "content": user_content})
            answer, logits, probs = generate_answer_and_logits(
                model, tokenizer, outlines_model, messages, valid_answers, model_key, answer_pattern
            )
            messages.append({"role": "assistant", "content": answer})

            results.append({
                "model": model_key,
                "item_id": row["id"],
                "item_text": row["item"],
                "context_mode": "with_context",
                "flipped": resolve_flipped(row, task_flipped),
                "model_answer": answer,
                "partial_run": False,
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
