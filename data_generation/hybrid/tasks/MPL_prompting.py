#!/usr/bin/env python3
"""
MPL Task Module - Multiple Price List Task evaluation using chat template approach.
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
DATA_FILE = "survey_data/prompts_mpl.jsonl"  

def parse_mpl_data(text: str) -> Dict[str, Any]:
    """
    Parse MPL text and extract all information.
    """
    # Extract pump and stop keys from instructions
    lot_1 = re.search(r'Your task is to choose between lottery ([A-Z])', text)
    lot_2 = re.search(r'and lottery ([A-Z]) in each pair.', text)
    
    if not lot_1 or not lot_2:
        raise ValueError("Could not identify lot keys from instructions")
    
    lot_1_key = lot_1.group(1)
    lot_2_key = lot_2.group(1)
    
    # Extract instructions
    instructions_end = text.find('\n\nProblem 1:')
    instructions = text[:instructions_end].strip() + " For each choice, following the '<<' brackets, strictly respond with one letter." if instructions_end != -1 else ""

    # Split into Problem sections
    Problem_pattern = r'\n\n(Problem \d+:.*?)(?=\n\nProblem \d+:|\Z)'
    Problem_matches = re.findall(Problem_pattern, text, re.DOTALL)
    
    problems = []
    for problem_text in Problem_matches:
        # Extract the problem number from within the text
        problem_num = re.search(r'Problem (\d+):', problem_text)
        problem_num = int(problem_num.group(1))
        
        # Extract all lottery pairs within this problem
        # Each lottery pair consists of two "Lottery T:" or "Lottery R:" lines followed by a choice
        lottery_pair_pattern = r'Lottery ([A-Z]): (.*?)\nLottery ([A-Z]): (.*?)\nYou chose lottery <<([A-Z])>>\.'
        lottery_pairs = re.findall(lottery_pair_pattern, problem_text, re.DOTALL)
        
        # Build decision list for this problem
        decisions = []
        for idx, (lottery_1_name, lottery_1_desc, lottery_2_name, lottery_2_desc, choice) in enumerate(lottery_pairs, 1):
            decisions.append({
                "decision_num": idx,
                "lottery_1_name": lottery_1_name,
                "lottery_1_desc": lottery_1_desc.strip(),
                "lottery_2_name": lottery_2_name,
                "lottery_2_desc": lottery_2_desc.strip(),
                "choice": choice
            })
        
        problems.append({
            "problem_num": problem_num,
            "decisions": decisions
        })

    return {
        "instructions": instructions,
        "lot_1_key": lot_1_key,
        "lot_2_key": lot_2_key,
        "problems": problems
    }

def build_chat_sequence_with_decisions(parsed_data: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Build a single chat sequence containing ALL decisions with proper chat template assistant tokens.
    This allows one forward pass to get logprobs for all decisions.
    """
    USER_TOK, ASSIST_TOK = detect_chat_tokens(tokenizer)
    
    instructions = parsed_data["instructions"]
    lot_1_key = parsed_data["lot_1_key"]
    lot_2_key = parsed_data["lot_2_key"]
    problems = parsed_data["problems"]

    # Build one continuous chat sequence with all decisions
    chat_parts = [f"{USER_TOK} {instructions}"]
    decision_points = []
    
    for problem in problems:
        problem_num = problem["problem_num"]
        decisions = problem["decisions"]

        if problem_num == 1:
            chat_parts.append(f" Problem {problem_num}:")
        else:
            chat_parts.append(f"{USER_TOK} Problem {problem_num}:")
        
        first_decision = True
        
        for decision in decisions:
            decision_num = decision["decision_num"]
            lottery_1_name = decision["lottery_1_name"]
            lottery_1_desc = decision["lottery_1_desc"]
            lottery_2_name = decision["lottery_2_name"]
            lottery_2_desc = decision["lottery_2_desc"]
            choice = decision["choice"]
            
            # Build the sequence for this lottery pair
            if ASSIST_TOK == "<<":
                lottery_text = f"Lottery {lottery_1_name}: {lottery_1_desc}\nLottery {lottery_2_name}: {lottery_2_desc}\nYou chose lottery "
            else:
                lottery_text = f"Lottery {lottery_1_name}: {lottery_1_desc}\nLottery {lottery_2_name}: {lottery_2_desc}\nYou chose lottery <<"
            
            if first_decision:
                # First decision comes right after instructions
                decision_sequence = f" {lottery_text} {ASSIST_TOK} {choice}"
                first_decision = False
            else:
                # Subsequent decisions start with USER_TOK
                decision_sequence = f"{USER_TOK} {lottery_text} {ASSIST_TOK} {choice}"
            
            # Store decision info
            decision_points.append({
                "problem_num": problem_num,
                "decision_num": decision_num,
                "decision_key": choice
            })
            
            # Add to chat sequence
            chat_parts.append(decision_sequence)
    
    # Join everything into one sequence
    full_text = "".join(chat_parts)
    # Now tokenize the full sequence
    tokenized = encode_without_chat_template(tokenizer, full_text)
    input_ids = tokenized.input_ids
    offsets = tokenized.offset_mapping[0].tolist()

    # Find all assistant token occurrences in the text and map to token positions
    assistant_positions = []
    
    # Use regex to find all assistant token positions in the original text
    assist_pattern = re.escape(ASSIST_TOK) + r'\s*([' + re.escape(lot_1_key) + re.escape(lot_2_key) + r'])'
    
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
        "lot_1_key": lot_1_key,
        "lot_2_key": lot_2_key,
        "full_text": full_text,  # For debugging
        "assistant_token": ASSIST_TOK,  # Store for reference
        "chat_format": True
    }


def get_decision_logprobs_chat_style(sequence_data: Dict[str, Any], model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> List[Dict[str, Any]]:
    """
    Get log probabilities for ALL decisions in ONE forward pass.
    Searches directly for the decision letter token in the sequence.
    """
    input_ids = sequence_data["input_ids"]
    decision_points = sequence_data["decision_points"]
    lot_1_key = sequence_data["lot_1_key"]
    lot_2_key = sequence_data["lot_2_key"]
    
    # Get all possible token IDs for the decision letters (with and without space)
    lot_1_tokens_with_space = tokenizer(f" {lot_1_key}", add_special_tokens=False).input_ids
    lot_2_tokens_with_space = tokenizer(f" {lot_2_key}", add_special_tokens=False).input_ids
    lot_1_tokens_no_space = tokenizer(lot_1_key, add_special_tokens=False).input_ids
    lot_2_tokens_no_space = tokenizer(lot_2_key, add_special_tokens=False).input_ids

    # Handle space and letter split case
    lot_1_seq = get_token_sequence(lot_1_tokens_with_space)
    lot_2_seq = get_token_sequence(lot_2_tokens_with_space)
    # We'll take the *last token* in each sequence as the actual decision letter
    lot_1_token = lot_1_seq[-1]
    lot_2_token = lot_2_seq[-1]

    # For logprob extraction, use the first token with space (most common case)
    # lot_1_token = lot_1_tokens_with_space[0]
    # lot_2_token = lot_2_tokens_with_space[0]
    
    # Warn if multi-token
    # if len(lot_1_tokens_with_space) > 1 or len(lot_2_tokens_with_space) > 1:
    #     logging.warning("Key tokenized into multiple tokens — may affect logprobs.")

    # Create set of all possible decision token IDs for matching
    decision_token_ids = set(lot_1_tokens_with_space + lot_2_tokens_with_space + 
                            lot_1_tokens_no_space + lot_2_tokens_no_space)
    
    #logging.info(f"Decision token IDs to search for: {decision_token_ids}")
    #logging.info(f"lot_1_key='{lot_1_key}', lot_2_key='{lot_2_key}'")

    # Single forward pass for the entire sequence
    with torch.inference_mode():
        outputs = model(input_ids=input_ids.to(model.device))
        logits = outputs.logits  # (1, seq_len, vocab)
    
    results = []
    
    for decision in decision_points:
        try:
            # Get the assistant token position as starting point
            assist_token_pos = decision.get("assistant_token_position")
            if assist_token_pos is None:
                logging.warning(f"No assistant token position found for problem {decision['problem_num']}")
                continue
            
            # Search for the decision letter token starting from assistant position
            # Look in a window of next 5 tokens (should be very close)
            search_window = 8
            pred_pos = None
            
            for offset in range(1, search_window + 1):
                check_pos = assist_token_pos + offset
                if check_pos >= input_ids.shape[1]:
                    break
                    
                token_id = input_ids[0, check_pos].item()
                token_str = tokenizer.decode([token_id])
                
                # Check if this token matches one of our decision letters
                if token_id in decision_token_ids:
                    pred_pos = check_pos
                    break
            
            if pred_pos is None:
                logging.warning(f"Could not find decision letter token for problem {decision['problem_num']} near position {assist_token_pos}")
                # Fallback to old method
                assistant_token_ids = tokenizer(sequence_data["assistant_token"], add_special_tokens=False).input_ids
                pred_pos = assist_token_pos + len(assistant_token_ids)
            
            if pred_pos >= logits.shape[1]:
                logging.warning(f"Prediction position {pred_pos} exceeds sequence length {logits.shape[1]}")
                continue
            
            # Get logprobs at the position BEFORE the decision letter
            # (the model predicts the decision letter at this position)
            pred_logits = logits[0, pred_pos - 1, :]
            log_probs = torch.log_softmax(pred_logits, dim=-1)
            
            results.append({
                "problem_num": decision["problem_num"],
                "decision_num": decision["decision_num"],
                "human_decision": decision["decision_key"],
                "log_prob_lot_1": log_probs[lot_1_token].item(),
                "log_prob_lot_2": log_probs[lot_2_token].item(),
                "lot_1_key": lot_1_key,
                "lot_2_key": lot_2_key,
            })
            
        except Exception as e:
            logging.error(f"Error processing decision for problem {decision['problem_num']}: {e}")
            continue
    
    return results



def process_MPL_participant_chat_style(participant_data: dict, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, model_key: str, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Process a single MPL participant using chat-style approach.
    """
    text = participant_data["text"]
    participant_id = participant_data["participant"]
    
    try:
        # Parse the data
        parsed_data = parse_mpl_data(text)
        
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
                "log_prob_lot_1": result["log_prob_lot_1"],
                "log_prob_lot_2": result["log_prob_lot_2"],
                "model": model_key,
                "problem": result["problem_num"],
                "decision": result["decision_num"],
                "participant": participant_id,
                "experiment": "MPL task",
                # Additional MPL-specific info
                "lot_1_key": result["lot_1_key"],
                "lot_2_key": result["lot_2_key"],
            })
        
        return standardized_results
        
    except Exception as e:
        logging.error(f"Error processing participant {participant_id}: {e}")
        return []

def run_task(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, model_key: str, 
             test_mode: bool = False, data_file: str = DATA_FILE) -> pd.DataFrame:
    """
    Main task runner function for MPL evaluation.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        model_key: String identifier for the model
        test_mode: Whether to run in test mode (fewer entries)
        data_file: Path to the data file
    
    Returns:
        pandas.DataFrame: Results dataframe with MPL decision logprobs
    """
    logging.info(f"Starting MPL task for model: {model_key}")
    
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
                logging.info(f"Processing MPL participant {entry_idx + 1}/{len(participant_data)}: {participant.get('participant', 'unknown')}")
            
            try:
                results = process_MPL_participant_chat_style(participant, model, tokenizer, model_key, verbose=test_mode)
                all_results.extend(results)
                    
            except Exception as e:
                logging.error(f"Error processing MPL participant {entry_idx}: {e}")
                continue
        
        # Create and return DataFrame
        if all_results:
            df = pd.DataFrame(all_results)
            logging.info(f"MPL task completed. Generated {len(all_results)} rows of results.")
            
            return df
        else:
            logging.warning(f"No results generated for MPL task on {model_key}")
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Error in MPL task for model {model_key}: {e}")
        return pd.DataFrame()

# For standalone testing
if __name__ == "__main__":
    print("This is a task module meant to be imported by model_manager.py")
