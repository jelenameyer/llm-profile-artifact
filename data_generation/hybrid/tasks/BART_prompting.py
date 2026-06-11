#!/usr/bin/env python3
"""
BART Task Module - Balloon Analogue Risk Task evaluation using chat template approach.
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

DATA_FILE = "survey_data/prompts_bart.jsonl"  


def parse_bart_data(text: str) -> Dict[str, Any]:
    """
    Parse BART text and extract all information including balloon sequences and decision points.
    """
    # Extract pump and stop keys from instructions
    pump_match = re.search(r'pressing ([A-Z]) and you will accumulate', text)
    stop_match = re.search(r'pressing ([A-Z]) and you will collect', text)
    
    if not pump_match or not stop_match:
        raise ValueError("Could not identify pump and stop keys from instructions")
    
    pump_key = pump_match.group(1)
    stop_key = stop_match.group(1)
    
    # Extract instructions
    instructions_end = text.find('\n\nBalloon 1:')
    instructions = text[:instructions_end].strip() + " For each statement, following the '{' bracket, strictly respond with one letter." if instructions_end != -1 else ""
    
    # Split into balloon sections
    balloon_pattern = r'\n\nBalloon (\d+):\n(.*?)(?=\n\nBalloon \d+:|\Z)'
    balloon_matches = re.findall(balloon_pattern, text, re.DOTALL)
    
    balloons = []
    for balloon_num, balloon_text in balloon_matches:
        # Parse key presses
        key_pattern = r'\{([' + re.escape(pump_key) + re.escape(stop_key) + r'])\}'
        key_presses = re.findall(key_pattern, balloon_text)
        
        # Determine outcome
        if "explodes" in balloon_text:
            outcome = "explode"
            final_score = 0
        else:
            score_match = re.search(r'get (\d+) points', balloon_text)
            outcome = "stop"
            final_score = int(score_match.group(1)) if score_match else 0
        
        balloons.append({
            "balloon_num": int(balloon_num),
            "key_presses": key_presses,
            "outcome": outcome,
            "final_score": final_score,
            "pump_count": len([k for k in key_presses if k == pump_key])
        })
    
    return {
        "instructions": instructions,
        "pump_key": pump_key,
        "stop_key": stop_key,
        "balloons": balloons
    }

def build_chat_sequence_with_decisions(parsed_data: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Build a single chat sequence containing ALL decisions with proper chat template assistant tokens.
    This allows one forward pass to get logprobs for all decisions.
    """
    USER_TOK, ASSIST_TOK = detect_chat_tokens(tokenizer)
    
    instructions = parsed_data["instructions"]
    pump_key = parsed_data["pump_key"]
    stop_key = parsed_data["stop_key"]
    balloons = parsed_data["balloons"]
    
    # Build one continuous chat sequence with all decisions
    chat_parts = [f"{USER_TOK} {instructions}"]
    decision_points = []
    
    for balloon in balloons:
        balloon_num = balloon["balloon_num"]
        key_presses = balloon["key_presses"]
        outcome = balloon["outcome"]
        final_score = balloon["final_score"]
        
        # Start this balloon
        balloon_start = f" Balloon {balloon_num}:\nYou press"
        # Build the sequence for this balloon with proper assistant tokens at each decision
        balloon_sequence = [balloon_start]
        
        for i, key_press in enumerate(key_presses):
            # Add the opening brace, then assistant token, then the decision, then closing brace
            if ASSIST_TOK == "<<":
                balloon_sequence.extend([f"{ASSIST_TOK} {key_press}"])
            else:
                balloon_sequence.extend([" {", f"{ASSIST_TOK} {key_press}"])
            
            # Store decision info (we'll find token positions later)
            decision_points.append({
                "balloon_num": balloon_num,
                "decision_num": i + 1,
                "pumps_so_far": key_presses[:i].count(pump_key),
                "choice_made": "pump" if key_press == pump_key else "stop",
                "balloon_outcome": outcome,
                "final_score": final_score,
                "decision_key": key_press
            })
        
        # Add the result of this balloon
        if outcome == "explode":
            balloon_sequence.append(f"{USER_TOK} The balloon was inflated too much and explodes.")
        else:
            balloon_sequence.append(f"{USER_TOK} You stop inflating the balloon and get {final_score} points.")
        
        # Add this balloon to the overall sequence
        chat_parts.extend(balloon_sequence)
    
    # Join everything into one sequence
    full_text = "".join(chat_parts)
    # Now tokenize the full sequence
    tokenized = encode_without_chat_template(tokenizer, full_text)
    input_ids = tokenized.input_ids
    offsets = tokenized.offset_mapping[0].tolist()

    # Find all assistant token occurrences in the text and map to token positions
    assistant_positions = []
    
    # Use regex to find all assistant token positions in the original text
    assist_pattern = re.escape(ASSIST_TOK) + r'\s*([' + re.escape(pump_key) + re.escape(stop_key) + r'])'
    
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
        "pump_key": pump_key,
        "stop_key": stop_key,
        "full_text": full_text,  # For debugging
        "assistant_token": ASSIST_TOK,  # Store for reference
        "chat_format": True
    }

def get_decision_logprobs_chat_style(sequence_data: Dict[str, Any], model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> List[Dict[str, Any]]:
    """
    Get log probabilities for ALL decisions in ONE forward pass.
    Uses the chat template assistant token positions to identify where decisions are made.
    """
    input_ids = sequence_data["input_ids"]
    decision_points = sequence_data["decision_points"]
    pump_key = sequence_data["pump_key"]
    stop_key = sequence_data["stop_key"]
    
    
    # Get token IDs for pump and stop keys
    pump_token_with_space = encode_without_chat_template(tokenizer, f" {pump_key}").input_ids[0]
    stop_token_with_space = encode_without_chat_template(tokenizer, f" {stop_key}").input_ids[0]
    # get warning if the tokenizers split the token that we are looking for
    # if len(pump_token_with_space) > 1 or len(stop_token_with_space) > 1:
    #     logging.warning("Pump/stop key tokenized into multiple tokens — may affect logprobs.")

    # Handle space and letter split case
    pump_seq = get_token_sequence(pump_token_with_space).tolist()
    stop_seq = get_token_sequence(stop_token_with_space).tolist()
    # We'll take the *last token* in each sequence as the actual decision letter
    pump_token = pump_seq[-1]
    stop_token = stop_seq[-1]

    # Single forward pass for the entire sequence
    with torch.inference_mode():
        outputs = model(input_ids=input_ids.to(model.device))
        logits = outputs.logits  # (1, seq_len, vocab)
    
    results = []
    
    for decision in decision_points:
        try:
            # Get the assistant token position
            assist_token_pos = decision.get("assistant_token_position")
            if assist_token_pos is None:
                logging.warning(f"No assistant token position found for balloon {decision['balloon_num']}, decision {decision['decision_num']}")
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
            
            # Get logprobs at this position for pump vs stop
            pred_logits = logits[0, pred_pos, :]
            log_probs = torch.log_softmax(pred_logits, dim=-1)
            
            results.append({
                "balloon_num": decision["balloon_num"],
                "decision_num": decision["decision_num"],
                "pumps_so_far": decision["pumps_so_far"],
                "human_decision": decision["choice_made"],
                "balloon_outcome": decision["balloon_outcome"],
                "final_score": decision["final_score"],
                "log_prob_pump": log_probs[pump_token].item(),
                "log_prob_stop": log_probs[stop_token].item(),
                "pump_key": pump_key,
                "stop_key": stop_key,
            })
            
        except Exception as e:
            logging.error(f"Error processing decision for balloon {decision['balloon_num']}, decision {decision['decision_num']}: {e}")
            continue
    
    return results

def process_bart_participant_chat_style(participant_data: dict, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, model_key: str, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Process a single BART participant using chat-style approach.
    """
    text = participant_data["text"]
    participant_id = participant_data["participant"]
    
    try:
        # Parse the data
        parsed_data = parse_bart_data(text)
        
        if verbose:
            logging.info(f"Participant {participant_id}: Found {len(parsed_data['balloons'])} balloons")
            total_decisions = sum(len(b['key_presses']) for b in parsed_data['balloons'])
            logging.info(f"Participant {participant_id}: Total decisions to process: {total_decisions}")
        
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
                "log_prob_pump": result["log_prob_pump"],
                "log_prob_stop": result["log_prob_stop"],
                "model": model_key,
                "round": result["balloon_num"],
                "decision_num": result["decision_num"],
                "participant": participant_id,
                "experiment": "BART task",
                # Additional BART-specific info
                "pumps_so_far": result["pumps_so_far"],
                "balloon_outcome": result["balloon_outcome"],
                "final_score": result["final_score"],
                "pump_key": result["pump_key"],
                "stop_key": result["stop_key"],
            })
        
        return standardized_results
        
    except Exception as e:
        logging.error(f"Error processing participant {participant_id}: {e}")
        return []

def run_task(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, model_key: str, 
             test_mode: bool = False, data_file: str = DATA_FILE) -> pd.DataFrame:
    """
    Main task runner function for BART evaluation.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        model_key: String identifier for the model
        test_mode: Whether to run in test mode (fewer entries)
        data_file: Path to the data file
    
    Returns:
        pandas.DataFrame: Results dataframe with BART decision logprobs
    """
    logging.info(f"Starting BART task for model: {model_key}")
    
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
            logging.info(f"Test mode: processing only first 2 participants")
            
        # Process each participant
        for entry_idx, participant in enumerate(participant_data):
            if entry_idx % 500 == 0:
                logging.info(f"Processing BART participant {entry_idx + 1}/{len(participant_data)}: {participant.get('participant', 'unknown')}")
            
            try:
                results = process_bart_participant_chat_style(participant, model, tokenizer, model_key, verbose=test_mode)
                all_results.extend(results)
                    
            except Exception as e:
                logging.error(f"Error processing BART participant {entry_idx}: {e}")
                continue
        
        # Create and return DataFrame
        if all_results:
            df = pd.DataFrame(all_results)
            logging.info(f"BART task completed. Generated {len(all_results)} rows of results.")
            return df
        else:
            logging.warning(f"No results generated for BART task on {model_key}")
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Error in BART task for model {model_key}: {e}")
        return pd.DataFrame()

# For standalone testing
if __name__ == "__main__":
    print("This is a task module meant to be imported by model_manager.py")
