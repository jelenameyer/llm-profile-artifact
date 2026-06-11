#!/usr/bin/env python3
"""
CCT Task Module - Columbia Card Task evaluation using chat template approach.
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

# Task-specific configurations
DATA_FILE = "survey_data/prompts_cct.jsonl"

def parse_cct_data(text: str) -> Dict[str, Any]:
    """
    Parse CCT text and extract all information including round sequences and decision points.
    """
    # Extract flip and stop keys from instructions
    flip_match = re.search(r'Press ([A-Z]) to turn a card over, or ([A-Z]) to stop', text)
    
    if not flip_match:
        raise ValueError("Could not identify flip and stop keys from instructions")
    
    flip_key = flip_match.group(1)
    stop_key = flip_match.group(2)
    
    # Extract instructions
    instructions_end = text.find('\n\nRound 1:')
    instructions = text[:instructions_end].strip() + " For each statement, following the '<<' brackets, strictly respond with one letter." if instructions_end != -1 else ""
    
    # Split into round sections
    round_pattern = r'\n\nRound (\d+):\n(.*?)(?=\n\nRound \d+:|\Z)'
    round_matches = re.findall(round_pattern, text, re.DOTALL)
    
    rounds = []
    for round_num, round_text in round_matches:
        # Extract round parameters
        gain_match = re.search(r'You will be awarded (\d+) points for turning over a gain card', round_text)
        loss_match = re.search(r'You will lose (\d+) points for turning over a loss card', round_text)
        loss_cards_match = re.search(r'There are (\d+) loss cards in this round', round_text)
        
        gain_points = int(gain_match.group(1)) if gain_match else 0
        loss_points = int(loss_match.group(1)) if loss_match else 0
        loss_cards = int(loss_cards_match.group(1)) if loss_cards_match else 0
        
        # Parse key presses
        key_pattern = r'<<([' + re.escape(flip_key) + re.escape(stop_key) + r'])>>'
        key_presses = re.findall(key_pattern, round_text)
        
        # Determine outcome - check if round ended with loss card
        if "encountered a loss card" in round_text:
            outcome = "loss"
        else:
            outcome = "stop"
        
        # Extract final score
        final_score_match = re.search(r'Your final score for this round is (-?\d+)', round_text)
        final_score = int(final_score_match.group(1)) if final_score_match else 0
        
        rounds.append({
            "round_num": int(round_num),
            "gain_points": gain_points,
            "loss_points": loss_points,
            "loss_cards": loss_cards,
            "key_presses": key_presses,
            "outcome": outcome,
            "final_score": final_score,
            "flip_count": len([k for k in key_presses if k == flip_key])
        })
    
    return {
        "instructions": instructions,
        "flip_key": flip_key,
        "stop_key": stop_key,
        "rounds": rounds
    }


def build_chat_sequence_with_decisions(parsed_data: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Build a single chat sequence containing ALL decisions with proper chat template assistant tokens.
    This allows one forward pass to get logprobs for all decisions.
    """
    USER_TOK, ASSIST_TOK = detect_chat_tokens(tokenizer)
    
    instructions = parsed_data["instructions"]
    flip_key = parsed_data["flip_key"]
    stop_key = parsed_data["stop_key"]
    rounds = parsed_data["rounds"]
    
    # Build one continuous chat sequence with all decisions
    chat_parts = [f"{USER_TOK} {instructions}"]
    decision_points = []
    
    for round_data in rounds:
        round_num = round_data["round_num"]
        key_presses = round_data["key_presses"]
        outcome = round_data["outcome"]
        final_score = round_data["final_score"]
        gain_points = round_data["gain_points"]
        loss_points = round_data["loss_points"]
        loss_cards = round_data["loss_cards"]
        
        # Start this round with round info
        round_start = (f" Round {round_num}:\n"
                      f"You will be awarded {gain_points} points for turning over a gain card.\n"
                      f"You will lose {loss_points} points for turning over a loss card.\n"
                      f"There are {loss_cards} loss cards in this round.\n"
                      f"You press")
        
        round_sequence = [round_start]
        
        # Track cards flipped and current score for this round
        cards_flipped = 0
        current_score = 0
        
        for i, key_press in enumerate(key_presses):
            # Add the opening markers, then assistant token, then the decision
            if ASSIST_TOK == "<<":
                round_sequence.extend([f"{ASSIST_TOK} {key_press}"])
            else:
                round_sequence.extend([" <<", f"{ASSIST_TOK} {key_press}"])
                        
            # Determine what happened after this key press
            is_last_press = (i == len(key_presses) - 1)
            
            if key_press == flip_key:
                cards_flipped += 1
                # Check if this flip resulted in a loss
                if is_last_press and outcome == "loss":
                    # This was the losing card
                    current_score -= loss_points
                else:
                    # This was a gain card
                    current_score += gain_points
            
            # Store decision info
            decision_points.append({
                "round_num": round_num,
                "decision_num": i + 1,
                "cards_flipped_so_far": cards_flipped - 1 if key_press == flip_key else cards_flipped,
                "choice_made": "flip" if key_press == flip_key else "stop",
                "round_outcome": outcome,
                "final_score": final_score,
                "decision_key": key_press,
                "gain_points": gain_points,
                "loss_points": loss_points,
                "loss_cards": loss_cards
            })
            
            if key_press == flip_key:
                if is_last_press and outcome == "loss":
                    # Hit a loss card
                    round_sequence.append(f"{USER_TOK} and turn over a loss card. Your current score is {current_score}. The round has now ended because you encountered a loss card.\n"
                                        f"Your final score for this round is {final_score}.")
                else:
                    # Hit a gain card
                    round_sequence.append(f"{USER_TOK} and turn over a gain card. Your current score is {current_score}.\n"
                                        f"You press")
            else:  # stop_key
                # Stopped and claimed payout
                round_sequence.append(f"{USER_TOK} and claim your payout.\n"
                                    f"Your final score for this round is {final_score}.")
        
        chat_parts.extend(round_sequence)
    
    # Join everything into one sequence
    full_text = "".join(chat_parts)
    # Tokenize the full sequence
    tokenized = encode_without_chat_template(tokenizer, full_text)
    input_ids = tokenized.input_ids
    offsets = tokenized.offset_mapping[0].tolist()
    
    # Find all assistant token occurrences in the text and map to token positions
    assistant_positions = []
    
    # Use regex to find all assistant token positions in the original text
    assist_pattern = re.escape(ASSIST_TOK) + r'\s*([' + re.escape(flip_key) + re.escape(stop_key) + r'])'
    
    for match in re.finditer(assist_pattern, full_text):
        assist_start = match.start()
        decision_char = match.group(1)
        
        # Find the token position that corresponds to this text position
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
        "flip_key": flip_key,
        "stop_key": stop_key,
        "full_text": full_text,
        "assistant_token": ASSIST_TOK,
        "chat_format": True
    }


def get_decision_logprobs_chat_style(sequence_data: Dict[str, Any], model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> List[Dict[str, Any]]:
    """
    Get log probabilities for ALL decisions in ONE forward pass.
    Uses the actual chat template assistant token positions to identify where decisions are made.
    """
    input_ids = sequence_data["input_ids"]
    decision_points = sequence_data["decision_points"]
    flip_key = sequence_data["flip_key"]
    stop_key = sequence_data["stop_key"]

    # Get token IDs for flip and stop keys
    flip_token = encode_without_chat_template(tokenizer, f" {flip_key}").input_ids[0]
    stop_token = encode_without_chat_template(tokenizer, f" {stop_key}").input_ids[0]
    
    # Handle space and letter split case
    flip_seq = get_token_sequence(flip_token).tolist()
    stop_seq = get_token_sequence(stop_token).tolist()
    # We'll take the *last token* in each sequence as the actual decision letter
    flip_token = flip_seq[-1]
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
                logging.warning(f"No assistant token position found for round {decision['round_num']}, decision {decision['decision_num']}")
                continue
            
            # Get all tokens that make up the assistant marker
            assistant_token_ids = tokenizer(sequence_data["assistant_token"], add_special_tokens=False).input_ids
            num_assistant_toks = len(assistant_token_ids)
            
            # Compute the position right *after* the full assistant marker
            pred_pos = assist_token_pos + num_assistant_toks
            
            if pred_pos >= logits.shape[1]:
                logging.warning(f"Prediction position {pred_pos} exceeds sequence length {logits.shape[1]}")
                continue
            
            # Get logprobs at this position for flip vs stop
            pred_logits = logits[0, pred_pos, :]
            log_probs = torch.log_softmax(pred_logits, dim=-1)
            
            results.append({
                "round_num": decision["round_num"],
                "decision_num": decision["decision_num"],
                "cards_flipped_so_far": decision["cards_flipped_so_far"],
                "human_decision": decision["choice_made"],
                "round_outcome": decision["round_outcome"],
                "final_score": decision["final_score"],
                "log_prob_flip": log_probs[flip_token].item(),
                "log_prob_stop": log_probs[stop_token].item(),
                "flip_key": flip_key,
                "stop_key": stop_key,
                "gain_points": decision["gain_points"],
                "loss_points": decision["loss_points"],
                "loss_cards": decision["loss_cards"]
            })
            
        except Exception as e:
            logging.error(f"Error processing decision for round {decision['round_num']}, decision {decision['decision_num']}: {e}")
            continue
    
    return results

def process_cct_participant_chat_style(participant_data: dict, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, model_key: str, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Process a single CCT participant using chat-style approach.
    """
    text = participant_data["text"]
    participant_id = participant_data["participant"]
    
    try:
        # Parse the data
        parsed_data = parse_cct_data(text)
        
        if verbose:
            logging.info(f"Participant {participant_id}: Found {len(parsed_data['rounds'])} rounds")
            total_decisions = sum(len(r['key_presses']) for r in parsed_data['rounds'])
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
                "log_prob_flip": result["log_prob_flip"],
                "log_prob_stop": result["log_prob_stop"],
                "model": model_key,
                "round": result["round_num"],
                "decision_num": result["decision_num"],
                "participant": participant_id,
                "experiment": "CCT task",
                # Additional CCT-specific info
                "cards_flipped_so_far": result["cards_flipped_so_far"],
                "round_outcome": result["round_outcome"],
                "final_score": result["final_score"],
                "flip_key": result["flip_key"],
                "stop_key": result["stop_key"],
                "gain_points": result["gain_points"],
                "loss_points": result["loss_points"],
                "loss_cards": result["loss_cards"]
            })
        
        return standardized_results
        
    except MemoryError as e:
        logging.warning(f"Skipping participant {participant_id} due to memory constraints: {e}")
        return []
    except Exception as e:
        logging.error(f"Error processing participant {participant_id}: {e}")
        return []

def run_task(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, model_key: str, 
             test_mode: bool = False, data_file: str = DATA_FILE) -> pd.DataFrame:
    """
    Main task runner function for CCT evaluation.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        model_key: String identifier for the model
        test_mode: Whether to run in test mode (fewer entries)
        data_file: Path to the data file
    
    Returns:
        pandas.DataFrame: Results dataframe with CCT decision logprobs, or empty DataFrame if skipped
    """
    logging.info(f"Starting CCT task for model: {model_key}")
    
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
            logging.info(f"Test mode: processing only first participant")
            
        # Process each participant
        for entry_idx, participant in enumerate(participant_data):
            logging.info(f"Processing CCT participant {entry_idx + 1}/{len(participant_data)}: {participant.get('participant', 'unknown')}")
            
            try:
                results = process_cct_participant_chat_style(participant, model, tokenizer, model_key, verbose=test_mode)
                all_results.extend(results)
                    
            except Exception as e:
                logging.error(f"Error processing CCT participant {entry_idx}: {e}")
                continue
        
        # Create and return DataFrame
        if all_results:
            df = pd.DataFrame(all_results)
            logging.info(f"CCT task completed. Generated {len(all_results)} rows of results.")
            
            # Log summary statistics
            if len(all_results) > 0:
                logging.info(f"Choice distribution: {df['human_decision'].value_counts().to_dict()}")
                avg_flips = df.groupby(['participant', 'round'])['cards_flipped_so_far'].max().mean()
                logging.info(f"Average cards flipped per round: {avg_flips:.1f}")
            
            return df
        else:
            logging.warning(f"No results generated for CCT task on {model_key}")
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Error in CCT task for model {model_key}: {e}")
        return pd.DataFrame()

# For standalone testing
if __name__ == "__main__":
    print("This is a task module meant to be imported by model_manager.py")
