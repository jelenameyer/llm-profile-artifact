#!/usr/bin/env python3
"""
Base Task Module - Shared logic for all survey tasks.
Individual tasks inherit from this and only need to specify configuration.
"""
# ------------- packages -----
import json
import torch
import pandas as pd
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from abc import ABC, abstractmethod


# ------------- helper function -----
def encode_without_chat_template(tokenizer, text):
    """
    Temporarily set the model's chat template to an empty string during
    tokenization, so no extra chat/thinking tokens are inserted, then restore it.
    """
    orig_tpl = tokenizer.chat_template
    tokenizer.chat_template = ""
    enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    tokenizer.chat_template = orig_tpl
    return enc



class BaseSurveyTask(ABC):
    """
    Base class for survey tasks that extract logprobs from LLM responses.
    
    Subclasses only need to define:
    - DATA_FILE: path to the JSONL file
    - TASK_NAME: human-readable task name
    - ANSWER_RANGE: tuple of (min, max) answer options (e.g., (1, 5) or (1, 11))
    - Optional: override methods for special behavior
    """
    
    # Subclasses must define these
    DATA_FILE: str = None
    TASK_NAME: str = None
    ANSWER_RANGE: Tuple[int, int] = None  # e.g., (1, 5) for 1-5 scale
    
    def __init__(self):
        if self.DATA_FILE is None or self.TASK_NAME is None or self.ANSWER_RANGE is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define DATA_FILE, TASK_NAME, and ANSWER_RANGE"
            )
    
    @staticmethod
    def detect_chat_tokens(tokenizer: AutoTokenizer, entry_idx: int) -> Tuple[str, str]:
        """
        Returns (USER_TOK, ASSIST_TOK) automatically.
        Works with HF chat models using chat_template as a string.
        Falls back to "" / "<<".
        """
        tpl = getattr(tokenizer, "chat_template", None)
        if tpl and isinstance(tpl, str):
            user_match = re.search(r"<\|im_start\|>user", tpl)
            assist_match = re.search(r"<\|im_start\|>assistant", tpl)
            user_tok = user_match.group(0) if user_match else "<|user|>"
            assist_tok = assist_match.group(0) if assist_match else "<|assistant|>"
            return user_tok, assist_tok

        if entry_idx == 0:
            logging.warning('Did not find specific assistant and user tokens, probably no chat models, using none!!')
        return "", "<<"

    def get_candidate_encodings(self, tokenizer: AutoTokenizer) -> List[List[int]]:
        """
        Get token encodings for all candidate answers.
        Returns list of token ID lists for each number in range.
        """
        min_ans, max_ans = self.ANSWER_RANGE
        return [
            tokenizer.encode(str(i), add_special_tokens=False) 
            for i in range(min_ans, max_ans + 1)
        ]
    
    def format_chat_text(self, text: str, tokenizer: AutoTokenizer, entry_idx: int) -> str:
        """
        Convert questionnaire text with << >> placeholders into chat format.
        Can be overridden if different formatting needed.
        
        Default format:
            1. Question text <<answer>>
        
        Becomes:
            <|user|> 1. Question text <<
            <|assistant|> answer
        """
        USER_TOK, ASSIST_TOK = self.detect_chat_tokens(tokenizer, entry_idx)
        
        lines = text.splitlines()
        rebuilt = []
        i = 0
        for ln in lines:
            # Match pattern: number. question text <<answer>>
            m = re.match(r"(\d+)\.\s*(.*)<<(\d+)>>", ln)
            if m:
                qnum, qtext, ans = m.groups()
                i+=1
                if i == 1:
                    if ASSIST_TOK == "<<":
                        rebuilt.append(f"{qnum}. {qtext}") #line 1 is already part of instruction, no further USER token necessary.
                    else:
                        rebuilt.append(f"{qnum}. {qtext} <<") #line 1 is already part of instruction, no further USER token necessary.
            
                else:
                    if ASSIST_TOK == "<<":
                        rebuilt.append(f"{USER_TOK} {qnum}. {qtext}") #if not 1, add user token, since it follows an ASSISTANT response.
                    else:
                        rebuilt.append(f"{USER_TOK} {qnum}. {qtext} <<") #if not 1, add user token, since it follows an ASSISTANT response.
                rebuilt.append(f"{ASSIST_TOK} {ans.strip()}")
            else:
                rebuilt.append(ln)
            
        return USER_TOK +"\n".join(rebuilt)
    
    def compute_logprobs(
        self, 
        text: str, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        model_key: str,
        entry_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Main logic: compute logprobs for all candidate answers at each << >> position.
        
        Args:
            text: Raw questionnaire text with << >> markers
            model: The loaded model
            tokenizer: The loaded tokenizer
            model_key: String identifier for the model
            
        Returns:
            List of dicts with logprobs for each answer position
        """
        # Format text for chat
        chat_text = self.format_chat_text(text, tokenizer, entry_idx)
        
        # Encode
        enc = encode_without_chat_template(tokenizer, chat_text)
        input_ids = enc.input_ids.to(model.device)
        offsets = enc.offset_mapping[0].tolist()
        
        # Get assistant token pattern
        _, ASSIST_TOK = self.detect_chat_tokens(tokenizer, entry_idx)
        
        # Match assistant responses (handles both 1-digit and 2-digit numbers)
        pattern = re.compile(rf"{re.escape(ASSIST_TOK)}\s*(\d{{1,2}})")
        
        # Compute logprobs
        with torch.inference_mode():
            out = model(input_ids)
            logprobs = torch.nn.functional.log_softmax(out.logits, dim=-1)[0]
        
        # Get candidate encodings
        candidate_encs = self.get_candidate_encodings(tokenizer)
        min_ans, max_ans = self.ANSWER_RANGE
        
        results = []
        
        # Process each answer position
        for m in pattern.finditer(chat_text):
            human_answer = m.group(1)
            span_lo, span_hi = m.span(1)
            
            # Find token indices overlapping with this number span
            tok_indices = [
                i for i, (lo, hi) in enumerate(offsets)
                if not (hi <= span_lo or lo >= span_hi)
            ]
            
            if not tok_indices:
                logging.warning(
                    f"No token overlap for answer {human_answer} at span {span_lo}-{span_hi}"
                )
                continue
            
            try:
                # First token position (anchor)
                tok_idx = tok_indices[0]
                
                # Compute logprobs for each candidate
                lp_candidates = {}
                for k, enc in zip(range(min_ans, max_ans + 1), candidate_encs):
                    if len(enc) == 1:
                        # Single-token number
                        lp = logprobs[tok_idx][enc[0]].item()
                    else:
                        # Multi-token number: sum logprobs (approximation)
                        lp = 0.0
                        for j, t in enumerate(enc):
                            if tok_idx + j < logprobs.size(0):
                                lp += logprobs[tok_idx + j][t].item()
                    
                    lp_candidates[str(k)] = lp
                
                results.append(dict(human_number=human_answer, **lp_candidates))
                
            except (StopIteration, IndexError) as e:
                logging.warning(f"Could not process span {span_lo}-{span_hi}: {e}")
                continue
        
        return results
    
    def process_entry(
        self, 
        entry: Dict[str, Any], 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        model_key: str,
        entry_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Process a single entry from the JSONL file.
        Can be overridden for custom processing logic.
        
        Args:
            entry: Dictionary from JSONL file
            model: The loaded model
            tokenizer: The loaded tokenizer
            model_key: String identifier for the model
            
        Returns:
            List of result rows
        """
        rows = []
        
        # Compute logprobs
        spans = self.compute_logprobs(entry["text"], model, tokenizer, model_key, entry_idx)
        
        # Add metadata to each result
        for i, s in enumerate(spans, 1):
            s["model"] = model_key
            s["item"] = i
            s["participant"] = entry.get("participant", "")
            s["flipped"] = entry.get("flipped", "")
            s["experiment"] = entry.get("experiment", "")
            rows.append(s)
        
        return rows
    
    def run_task(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        model_key: str,
        test_mode: bool = False,
        data_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Main task runner function - called by the model manager.
        
        Args:
            model: The loaded model
            tokenizer: The loaded tokenizer
            model_key: String identifier for the model
            test_mode: Whether to run in test mode (fewer entries, more logging)
            data_file: Optional override for data file path
            
        Returns:
            pandas.DataFrame: Results with logprobs
        """
        data_path = data_file or self.DATA_FILE
        logging.info(f"Starting {self.TASK_NAME} task for model: {model_key}")
        
        all_rows = []
        
        try:
            # Load data
            with open(data_path) as f:
                entries = [json.loads(line) for line in f]
            
            # Limit entries in test mode
            if test_mode:
                entries = entries[:2]
                logging.info(f"Test mode: processing only first 2 entries")
            
            # Process each entry
            for entry_idx, entry in enumerate(entries):
                if entry_idx % 500 == 0:
                    logging.info(
                        f"Processing {self.TASK_NAME} entry {entry_idx + 1}/{len(entries)}"
                    )
                
                try:
                    rows = self.process_entry(entry, model, tokenizer, model_key, entry_idx)
                    all_rows.extend(rows)
                    
                except Exception as e:
                    logging.error(
                        f"Error processing {self.TASK_NAME} entry {entry_idx}: {e}"
                    )
                    continue
            
            # Create and return DataFrame
            if all_rows:
                df = pd.DataFrame(all_rows)
                logging.info(
                    f"{self.TASK_NAME} task completed. "
                    f"Generated {len(all_rows)} rows of results."
                )
                return df
            else:
                logging.warning(
                    f"No results generated for {self.TASK_NAME} task on {model_key}"
                )
                return pd.DataFrame()
                
        except Exception as e:
            logging.error(f"Error in {self.TASK_NAME} task for model {model_key}: {e}")
            return pd.DataFrame()
    
    def get_task_info(self) -> Dict[str, Any]:
        """Return information about this task."""
        min_ans, max_ans = self.ANSWER_RANGE
        answer_cols = [str(i) for i in range(min_ans, max_ans + 1)]
        
        return {
            "name": self.TASK_NAME,
            "description": f"{self.TASK_NAME} questionnaire evaluation using logprobs",
            "output_columns": ["model", "item", "participant", "flipped", 
                             "experiment", "human_number"] + answer_cols,
            "data_file": self.DATA_FILE,
            "answer_range": self.ANSWER_RANGE
        }


# Convenience function for module-style interface
def create_task_runner(task_class):
    """
    Creates a module-style run_task function from a task class.
    This maintains backward compatibility with the existing code.
    """
    task_instance = task_class()
    
    def run_task(model, tokenizer, model_key, test_mode=False, data_file=None):
        return task_instance.run_task(model, tokenizer, model_key, test_mode, data_file)
    
    def get_task_info():
        return task_instance.get_task_info()
    
    return run_task, get_task_info