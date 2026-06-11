#!/usr/bin/env python3
"""
DM Task Module - 4-point scale risk-taking questionnaire. (1-4)
"""

from base_task import BaseSurveyTask, create_task_runner


class DMTask(BaseSurveyTask):    
    DATA_FILE = "survey_data/promptsDmVaried.jsonl" 
    TASK_NAME = "DM"  # Short name for the task
    ANSWER_RANGE = (1, 4)  # (min, max) answer values
    

# Create module-style interface for backward compatibility
run_task, get_task_info = create_task_runner(DMTask)


if __name__ == "__main__":
    print("This is a task module meant to be imported by model_manager.py")
    print("Task info:", get_task_info())