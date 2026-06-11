#!/usr/bin/env python3
"""
SOEP Task Module - 10-point scale risk-taking questionnaire. (1-11)
"""

from base_task import BaseSurveyTask, create_task_runner


class SOEPTask(BaseSurveyTask):
    DATA_FILE = "survey_data/promptsSoepVaried.jsonl"  
    TASK_NAME = "SOEP"  # Short name for the task
    ANSWER_RANGE = (1, 11)  # (min, max) answer values
    

# Create module-style interface for backward compatibility
run_task, get_task_info = create_task_runner(SOEPTask)


if __name__ == "__main__":
    print("This is a task module meant to be imported by model_manager.py")
    print("Task info:", get_task_info())