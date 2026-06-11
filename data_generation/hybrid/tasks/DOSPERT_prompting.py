#!/usr/bin/env python3
"""
DOSPERT Task Module - 5-point scale risk-taking questionnaire.
"""

from base_task import BaseSurveyTask, create_task_runner


class DOSPERTTask(BaseSurveyTask):
    
    DATA_FILE = "survey_data/promptsDospertVaried.jsonl"
    TASK_NAME = "DOSPERT"
    ANSWER_RANGE = (1, 5)  # Answers range from 1 to 5


# Create module-style interface for backward compatibility
run_task, get_task_info = create_task_runner(DOSPERTTask)


if __name__ == "__main__":
    print("This is a task module meant to be imported by model_manager.py")
    print("Task info:", get_task_info())