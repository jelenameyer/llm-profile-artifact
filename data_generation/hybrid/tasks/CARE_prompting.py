#!/usr/bin/env python3
"""
CARE Task Module - free integer input scale risk-taking questionnaire. (use scale 0-99, because in summary of human data, that was the highest answer)
"""

from base_task import BaseSurveyTask, create_task_runner


class CARETask(BaseSurveyTask):    
    DATA_FILE = "survey_data/promptsCare.jsonl"  
    TASK_NAME = "CARE"  # Short name for the task
    ANSWER_RANGE = (0, 99)  # (min, max) answer values
    

# Create module-style interface for backward compatibility
run_task, get_task_info = create_task_runner(CARETask)


if __name__ == "__main__":
    print("This is a task module meant to be imported by model_manager.py")
    print("Task info:", get_task_info())