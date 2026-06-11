#!/usr/bin/env python3
"""
DAST Task Module - 2-point scale risk-taking questionnaire. (1= Yes, 2= No (or flipped))
"""

from base_task import BaseSurveyTask, create_task_runner


class DASTTask(BaseSurveyTask):    
    DATA_FILE = "survey_data/promptsDastVaried.jsonl"  
    TASK_NAME = "DAST"  # Short name for the task
    ANSWER_RANGE = (1, 2)  # (min, max) answer values
    

# Create module-style interface for backward compatibility
run_task, get_task_info = create_task_runner(DASTTask)


if __name__ == "__main__":
    print("This is a task module meant to be imported by model_manager.py")
    print("Task info:", get_task_info())