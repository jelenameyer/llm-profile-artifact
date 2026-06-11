#!/usr/bin/env python3
"""
LOT Task Module - Lotteries Task evaluation using chat template approach.
This module can be imported and run by the model manager.
"""

import json
import torch
import pandas as pd
import re
import logging
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

from utilsBehaviouralTasks import detect_chat_tokens, get_token_sequence
from base_task import encode_without_chat_template

# Task-specific configuration
DATA_FILE = "survey_data/prompts_lot_with_prob_numbers.jsonl"  

def parse_lot_data(text: str) -> Dict[str, Any]:
    """
    Parse LOT text and extract all information.
    """
    # Extract pump and stop keys from instructions
    box_1 = re.search(r'Problem 1:\nLottery ([A-Z]):', text)
    box_2 = re.search(r'Lottery ([A-Z]): gain 90 points ', text)
    
    if not box_1 or not box_2:
        raise ValueError("Could not identify box keys from instructions")
    
    box_1_key = box_1.group(1)
    box_2_key = box_2.group(1)
    
    # Extract instructions
    instructions_end = text.find('\n\nProblem 1:')
    instructions = text[:instructions_end].strip() + " For each statement, following the '<<' brackets, strictly respond with one letter." if instructions_end != -1 else ""

    # Split into Problem sections
    Problem_pattern = r'(?m)^Problem\s+\d+(?:\.\d+)*:.*?(?=^Problem\s+\d+(?:\.\d+)*:|\Z)'
    Problem_matches = re.findall(Problem_pattern, text, re.DOTALL)

    problems = []
    for problem_text in Problem_matches:
        # Extract the problem number from within the text
        problem_num = re.search(r'Problem\s+(\d+(?:\.\d+)*):', problem_text)
        problem_num = problem_num.group(1)

        if re.fullmatch(r'\d+', problem_num):
            problems.append({
                "problem_num": int(problem_num),
                "problem_text": problem_text
            })
        else: 
            problems.append({
                "problem_num": float(problem_num),
                "problem_text": problem_text
            })
    return {
        "instructions": instructions,
        "box_1_key": box_1_key,
        "box_2_key": box_2_key,
        "problems": problems
    }

def build_chat_sequence_with_decisions(parsed_data: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Build a single chat sequence containing ALL decisions with proper chat template assistant tokens.
    This allows one forward pass to get logprobs for all decisions.
    """
    USER_TOK, ASSIST_TOK = detect_chat_tokens(tokenizer)
    
    instructions = parsed_data["instructions"]
    box_1_key = parsed_data["box_1_key"]
    box_2_key = parsed_data["box_2_key"]
    problems = parsed_data["problems"]

    # Build one continuous chat sequence with all decisions
    chat_parts = [f"{USER_TOK} {instructions}"]
    decision_points = []
    
    for problem in problems:
        problem_num = problem["problem_num"]
        problem_text = problem["problem_text"]
        problem_start = re.search(rf'Problem {problem_num}:.*?You chose', problem_text, re.DOTALL)
        problem_start = problem_start.group()
        
        # extract human decison from problem
        human_decision = re.search(r'<<([A-Z])>>', problem_text, re.DOTALL)
        human_decision = human_decision.group(1)
        # Build the sequence for this problem with proper assistant token
        if problem_num == 1:
            if ASSIST_TOK == "<<":
                problem_sequence = [f" {problem_start} {ASSIST_TOK} {human_decision}"]
            else:
                problem_sequence = [f" {problem_start} <<{ASSIST_TOK} {human_decision}"]
        else:
            if ASSIST_TOK == "<<":
                problem_sequence = [f"{USER_TOK} {problem_start} {ASSIST_TOK} {human_decision}"]
            else:
                problem_sequence = [f"{USER_TOK} {problem_start} <<{ASSIST_TOK} {human_decision}"]

        # Store decision info (we'll find token positions later)
        decision_points.append({
            "problem_num": problem_num,
            "decision_key": human_decision
        })
        
        # Add this balloon to the overall sequence
        chat_parts.extend(problem_sequence)
    
    # Join everything into one sequence
    full_text = "".join(chat_parts)
    # Now tokenize the full sequence
    tokenized = encode_without_chat_template(tokenizer, full_text)
    input_ids = tokenized.input_ids
    offsets = tokenized.offset_mapping[0].tolist()

    # Find all assistant token occurrences in the text and map to token positions
    assistant_positions = []
    
    # Use regex to find all assistant token positions in the original text
    assist_pattern = re.escape(ASSIST_TOK) + r'\s*([' + re.escape(box_1_key) + re.escape(box_2_key) + r'])'
    
    for match in re.finditer(assist_pattern, full_text):
        assist_start = match.start()
        decision_char = match.group(1)
        
        # Find the token position that corresponds to this text position
        # Look for token that overlaps with the assistant token start
        for tok_idx, (start_pos, end_pos) in enumerate(offsets):
            if start_pos <= assist_start < end_pos:
                assistant_positions.append({
                    "token_position": tok_idx,
                    "decision_char": decision_char,
                    "text_position": assist_start
                })
                break
    
    # Match assistant positions to decision points
    if len(assistant_positions) != len(decision_points):
        logging.warning(f"Mismatch: found {len(assistant_positions)} assistant tokens but {len(decision_points)} decisions")
        # Take the minimum to avoid index errors
        min_len = min(len(assistant_positions), len(decision_points))
        assistant_positions = assistant_positions[:min_len]
        decision_points = decision_points[:min_len]
    
    # Add token positions to decision points
    for decision, pos_info in zip(decision_points, assistant_positions):
        decision["assistant_token_position"] = pos_info["token_position"]
        # Verify the decision matches
        if pos_info["decision_char"] != decision["decision_key"]:
            logging.warning(f"Decision mismatch: expected {decision['decision_key']}, found {pos_info['decision_char']}")
    
    return {
        "input_ids": input_ids,
        "decision_points": decision_points,
        "box_1_key": box_1_key,
        "box_2_key": box_2_key,
        "full_text": full_text,  # For debugging
        "assistant_token": ASSIST_TOK,  # Store for reference
        "chat_format": True
    }

def get_decision_logprobs_chat_style(sequence_data: Dict[str, Any], model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> List[Dict[str, Any]]:
    """
    Get log probabilities for ALL decisions in ONE forward pass.
    Uses the actual chat template assistant token positions to identify where decisions are made.
    """
    input_ids = sequence_data["input_ids"]
    decision_points = sequence_data["decision_points"]
    box_1_key = sequence_data["box_1_key"]
    box_2_key = sequence_data["box_2_key"]
    
    
   # Get token IDs for box_1 and box_2 keys
    box_1_token_with_space = encode_without_chat_template(tokenizer, f" {box_1_key}").input_ids[0]
    box_2_token_with_space = encode_without_chat_template(tokenizer, f" {box_2_key}").input_ids[0]

    # Handle space and letter split case
    box_1_seq = get_token_sequence(box_1_token_with_space).tolist()
    box_2_seq = get_token_sequence(box_2_token_with_space).tolist()

    # We'll take the *last token* in each sequence as the actual decision letter
    box_1_token = box_1_seq[-1]
    box_2_token = box_2_seq[-1]

    # Single forward pass for the entire sequence
    with torch.inference_mode():
        outputs = model(input_ids=input_ids.to(model.device))
        logits = outputs.logits  
    
    results = []
    
    for decision in decision_points:
        try:
            # Get the assistant token position
            assist_token_pos = decision.get("assistant_token_position")
            if assist_token_pos is None:
                logging.warning(f"No assistant token position found for problem {decision['problem_num']}")
                continue
            
            # The model predicts the next token AFTER the assistant token
            # Get all tokens that make up the assistant marker
            assistant_token_ids = tokenizer(sequence_data["assistant_token"], add_special_tokens=False).input_ids
            num_assistant_toks = len(assistant_token_ids)

            # Compute the position right *after* the full assistant marker
            pred_pos = assist_token_pos + num_assistant_toks

            # --- Skip whitespace tokens if the model splits " A" into separate tokens ---
            for offset in range(0, 2):  # look ahead up to 2 tokens to be safe
                if pred_pos + offset >= input_ids.shape[1]:
                    logging.warning(f"Prediction position {pred_pos} exceeds sequence length {logits.shape[1]}")
                    break
                token_str = tokenizer.decode([input_ids[0, pred_pos + offset]])
                # Skip whitespace-only tokens
                if token_str.strip() == "":
                    continue
                else:
                    pred_pos = pred_pos + offset
                    break

            # Get logprobs at this position
            pred_logits = logits[0, pred_pos, :]
            log_probs = torch.log_softmax(pred_logits, dim=-1)
            
            results.append({
                "problem_num": decision["problem_num"],
                "human_decision": decision["decision_key"],
                "log_prob_box_1": log_probs[box_1_token].item(),
                "log_prob_box_2": log_probs[box_2_token].item(),
                "box_1_key": box_1_key,
                "box_2_key": box_2_key,
            })
            
        except Exception as e:
            logging.error(f"Error processing decision for problem {decision['problem_num']}: {e}")
            continue
    
    return results

def process_DFD_participant_chat_style(participant_data: dict, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, model_key: str, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Process a single LOT participant using chat-style approach.
    """
    text = participant_data["text"]
    participant_id = participant_data["participant"]
    
    try:
        # Parse the data
        parsed_data = parse_lot_data(text)
        
        if verbose:
            logging.info(f"Participant {participant_id}: Found {len(parsed_data['problems'])} decision problems.")
        
        # Build chat sequence with decision points
        sequence_data = build_chat_sequence_with_decisions(parsed_data, tokenizer)
        
        if verbose:
            logging.info(f"Participant {participant_id}: Built {len(sequence_data['decision_points'])} decision points")
        
        # Get predictions for all decisions
        decision_results = get_decision_logprobs_chat_style(sequence_data, model, tokenizer)
        
        if verbose:
            logging.info(f"Participant {participant_id}: Successfully processed {len(decision_results)} decisions")
        
        # Convert to standardized format
        standardized_results = []
        for result in decision_results:
            standardized_results.append({
                "human_decision": result["human_decision"],
                "log_prob_box_1": result["log_prob_box_1"],
                "log_prob_box_2": result["log_prob_box_2"],
                "model": model_key,
                "round": result["problem_num"],
                "participant": participant_id,
                "experiment": "LOT task",
                # Additional LOT-specific info
                "box_1_key": result["box_1_key"],
                "box_2_key": result["box_2_key"],
            })
        
        return standardized_results
        
    except Exception as e:
        logging.error(f"Error processing participant {participant_id}: {e}")
        return []

def run_task(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, model_key: str, 
             test_mode: bool = False, data_file: str = DATA_FILE) -> pd.DataFrame:
    """
    Main task runner function for LOT evaluation.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        model_key: String identifier for the model
        test_mode: Whether to run in test mode (fewer entries)
        data_file: Path to the data file
    
    Returns:
        pandas.DataFrame: Results dataframe with BART decision logprobs
    """
    logging.info(f"Starting LOT task for model: {model_key}")
    
    all_results = []
    
    try:
        # Load participant data
        participant_data = []
        with open(data_file, 'r') as f:
            for line in f:
                participant_data.append(json.loads(line.strip()))
        
        # Handle single participant vs list of participants
        if isinstance(participant_data, dict):
            participant_data = [participant_data]
            
        # Limit entries in test mode
        if test_mode:
            participant_data = participant_data[:1]
            logging.info(f"Test mode: processing only first participant.")
            
        # Process each participant
        for entry_idx, participant in enumerate(participant_data):
            if entry_idx % 500 == 0:
                logging.info(f"Processing LOT participant {entry_idx + 1}/{len(participant_data)}: {participant.get('participant', 'unknown')}")
            
            try:
                results = process_DFD_participant_chat_style(participant, model, tokenizer, model_key, verbose=test_mode)
                all_results.extend(results)
                    
            except Exception as e:
                logging.error(f"Error processing LOT participant {entry_idx}: {e}")
                continue
        
        # Create and return DataFrame
        if all_results:
            df = pd.DataFrame(all_results)
            logging.info(f"LOT task completed. Generated {len(all_results)} rows of results.")
            
            return df
        else:
            logging.warning(f"No results generated for LOT task on {model_key}")
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Error in LOT task for model {model_key}: {e}")
        return pd.DataFrame()

# For standalone testing
if __name__ == "__main__":
    print("This is a task module meant to be imported by model_manager.py")
