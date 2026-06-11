#!/usr/bin/env python3
"""
GABS Task Module - Pathological Gambling - 2 to 4 item scale (varies between questions).
Uses custom formatting due to instructions block pequilarities.
"""

import re
from base_task import BaseSurveyTask, create_task_runner


class FTNDTask(BaseSurveyTask):
    
    DATA_FILE = "survey_data/promptsGabsVaried.jsonl"
    TASK_NAME = "GABS"
    ANSWER_RANGE = (1, 4)  # Extract logprobs for 1-5 even if some items use 1-3
    
    def format_chat_text(self, text, tokenizer, entry_idx):
        """
        Custom formatting for GABS which has:
        1. Instructions block at the top and in between items!
        2. Questions not always numbered
        3. Format: "Text ... 0. Question?\n1 = option, 2 = option, ... <<answer>>"
        
        We want to convert each question to:
        <|user|> Text ... 0. Question?
        1 = option, 2 = option, ...<<
        <|assistant|> answer
        
        And keep the first instructions separate at the top.
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
            m = re.match(r"\s*(.*)<<(\d+)>>", ln)
            if m:
                qtext, ans = m.groups()
                i+=1
                if i == 1:
                    if ASSIST_TOK == "<<":
                        rebuilt.append(f"{qtext}")
                    else:
                        rebuilt.append(f"{qtext} <<")
                else:
                    if ASSIST_TOK == "<<":
                        rebuilt.append(f"{USER_TOK} {qtext}")
                    else:
                        rebuilt.append(f"{USER_TOK} {qtext} <<") 
                rebuilt.append(f"{ASSIST_TOK} {ans.strip()}")
            elif ln.strip():  
                rebuilt.append(ln)

        
        return "\n".join(rebuilt)


# Create module-style interface for backward compatibility
run_task, get_task_info = create_task_runner(FTNDTask)


if __name__ == "__main__":
    print("This is a task module meant to be imported by model_manager.py")
    print("Task info:", get_task_info())