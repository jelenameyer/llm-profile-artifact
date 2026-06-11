#!/usr/bin/env python3
"""
BARRAT Task Module - 4-point scale impulsiveness questionnaire.
"""

from base_task import BaseSurveyTask, create_task_runner


class BARRATTask(BaseSurveyTask):
    
    DATA_FILE = "survey_data/promptsBarratVaried.jsonl"
    TASK_NAME = "BARRAT"
    ANSWER_RANGE = (1, 4)  # Answers range from 1 to 4


# Create module-style interface for backward compatibility
run_task, get_task_info = create_task_runner(BARRATTask)


if __name__ == "__main__":
    print("This is a task module meant to be imported by model_manager.py")
    print("Task info:", get_task_info())