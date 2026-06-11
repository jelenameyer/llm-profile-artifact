#!/usr/bin/env python3
"""
AUDIT Task Module - Alcohol Use Disorders Identification Test.
Uses custom formatting due to instructions block pequilarities.
"""

import re
from base_task import BaseSurveyTask, create_task_runner


class AUDITTask(BaseSurveyTask):
    """
    AUDIT questionnaire with mixed scales (1-3 and 1-5).
    I extract logprobs for 1-5 for all questions for consistency,
    even though some questions only use 1-3 in practice.
    """
    
    DATA_FILE = "survey_data/promptsAuditVaried.jsonl"
    TASK_NAME = "AUDIT"
    ANSWER_RANGE = (1, 5)  # Extract logprobs for 1-5 even if some items use 1-3

    
    def format_chat_text(self, text, tokenizer, entry_idx):
        """
        Custom formatting for AUDIT which has:
        1. Instructions block at the top
        2. Questions numbered starting from 0
        3. Format: "0. Question?\n1 = option, 2 = option, ... <<answer>>"
        
        We want to convert each question to:
        <|user|> 0. Question?
        1 = option, 2 = option, ...<<
        <|assistant|> answer
        
        And keep the instructions separate at the top.
        """
        USER_TOK, ASSIST_TOK = self.detect_chat_tokens(tokenizer, entry_idx)
        
        # Split into lines
        lines = text.splitlines()
        rebuilt = []
        
        # First, add instructions as a single user message
        # Find where instructions end (at "Statements:" line)
        instructions_end = 0
        for i, ln in enumerate(lines):
            if ln.strip() == "Statements:":
                instructions_end = i
                break
        
        # Add instructions block as user message
        if instructions_end > 0:
            instruction_lines = lines[:instructions_end + 1]  # Include "Statements:"
            rebuilt.append(f"{USER_TOK} " + "\n".join(instruction_lines))
        i = 0
        # Now process questions (everything after instructions)
        for ln in lines[instructions_end + 1:]:
            # Match question pattern: "N. Question text ... <<answer>>"
            m = re.match(r"(\d+)\.\s*(.*)<<(\d+)>>", ln)
            if m:
                qnum, qtext, ans = m.groups()
                i+=1
                if i == 1:
                    if ASSIST_TOK == "<<":
                        rebuilt.append(f"{qnum}. {qtext}")
                    else:
                        rebuilt.append(f"{qnum}. {qtext} <<")
                else:
                    if ASSIST_TOK == "<<":
                        rebuilt.append(f"{USER_TOK} {qnum}. {qtext}")
                    else:
                        rebuilt.append(f"{USER_TOK} {qnum}. {qtext} <<") 
                rebuilt.append(f"{ASSIST_TOK} {ans.strip()}")
            elif ln.strip():  # Non-empty line that's not a question
                # This shouldn't happen in well-formed AUDIT data
                rebuilt.append(ln)

        
        return "\n".join(rebuilt)


# Create module-style interface for backward compatibility
run_task, get_task_info = create_task_runner(AUDITTask)


if __name__ == "__main__":
    print("This is a task module meant to be imported by model_manager.py")
    print("Task info:", get_task_info())