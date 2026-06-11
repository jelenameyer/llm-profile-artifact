#!/usr/bin/env python3
"""
Shared utilities for psychological task evaluation modules.
Common functions used across different task implementations.
"""

import re
from typing import Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging


def detect_chat_tokens(tokenizer: AutoTokenizer) -> Tuple[str, str]:
    """
    Detect user and assistant tokens from tokenizer's chat template.
    Returns:
        Tuple[str, str]: (USER_TOKEN, ASSISTANT_TOKEN)
    """
    tpl = getattr(tokenizer, "chat_template", None)
    if tpl and isinstance(tpl, str):
            user_match = re.search(r"<\|im_start\|>user", tpl)
            assist_match = re.search(r"<\|im_start\|>assistant", tpl)
            user_tok = user_match.group(0) if user_match else "<|user|>"
            assist_tok = assist_match.group(0) if assist_match else "<|assistant|>"
            return user_tok, assist_tok
    
    # Fallback to standard tokens
    return "", "<<"

# Handle space and letter split case
def get_token_sequence(token_list):
    # if it's one token: just return it
    if len(token_list) == 1:
        return token_list
    # if it's split into space + letter: return both tokens
    elif len(token_list) == 2:
        return token_list
    else:
        logging.warning(f"Unexpected tokenization length: {token_list}")
        return token_list[-1:]  # fallback to last token
