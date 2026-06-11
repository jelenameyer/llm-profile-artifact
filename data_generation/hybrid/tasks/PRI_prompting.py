#!/usr/bin/env python3
"""
PRI Task Module - 2-point scale risk-taking questionnaire plus 7-point certainty measures for each risk-taking item.
"""
import re
from base_task import BaseSurveyTask, create_task_runner


class NewTask(BaseSurveyTask):
    DATA_FILE = "survey_data/promptsPriVaried.jsonl" 
    TASK_NAME = "PRI"  # Short name for the task
    ANSWER_RANGE = (1, 7)  # (min, max) answer values

    def format_chat_text(self, text, tokenizer, entry_idx):
        """
        Custom formatting for PRI which has:
        1. Instructions block at the top
        2. Questions numbered starting from 1
        3. Format: "1. Question?\n1 = option, 2 = option, ... <<answer>>, certainty question <<certainty answer>>"
        
        We want to convert each question to:
        <|user|> 1. Question?
        1 = option, 2 = option, ...<<
        <|assistant|> answer
        Certainty <<
        <|assistant|> certainty answer
        
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
            rebuilt.append(f"{USER_TOK}" + "\n".join(instruction_lines))
        i = 0
        # Now process questions (everything after instructions)
        for ln in lines[instructions_end + 1:]:
            # Match question pattern
            m = re.match(r"(\d+)\.\s*(.*)<<(\d+)>>\,\s*(.*)<<(\d+)>>", ln)
            if m:
                qnum, qtext, ans, qcertext, anscert = m.groups()
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
                rebuilt.append(f"{USER_TOK} {qcertext}")
                rebuilt.append(f"{ASSIST_TOK} {anscert}")
            elif ln.strip(): 
                rebuilt.append(ln)

        
        return "\n".join(rebuilt)
    

# Create module-style interface for backward compatibility
run_task, get_task_info = create_task_runner(NewTask)


if __name__ == "__main__":
    print("This is a task module meant to be imported by model_manager.py")
    print("Task info:", get_task_info())