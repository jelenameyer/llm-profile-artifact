#!/usr/bin/env python3
"""
Wrapper Script that loads Calling_LLM_Models.py for each model separately and cleanes all caches in between, for minimal GPU usage.
"""

# --- Packages ---
import subprocess
import time
import logging
import os
from datetime import datetime

# --- Setup global logging (for the launcher itself) ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("logs", "launcher_master.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Models to run sequentially ---
models_to_run = [
    # --- Microsoft Phi ---
    #"Phi-3.5-mini-4B",
    # "Phi-3-mini-4B",
    # "Phi-3-medium-14B",

    #---- Llama Instruct Models ----
    "llama31_8b",
    "llama32_1b",
    "llama32_3b",
    "llama31_70b_it",   
    "llama33_70b_it", 

    # --- Falcon Instruct ---
    "falcon3_1b",
    "falcon3_7b",
    "falcon3_10b",

    # --- Gemma Instruct ------
    "gemma-3-1b",
     "gemma-3-4b",
    "gemma-3-12b",
    "gemma-3-27b",
    "gemma-2-2b",
    "gemma-2-9b",
    "gemma-2-27b",

    # --- Mistral ----
    "Mistral-7b-v0.3",
    "Ministral-8b-2410",
    "mistral-24b-2501",
    
    # --- Qwen ---
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-30B-A3B",
    "Qwen3-32B",
    "Qwen25-1.5B",
    "Qwen25-3B",
    "Qwen25-7B",
    "Qwen25-14B",
    "Qwen25-32B",

    # --- OLMo Instruct ---   
    "olmo2_7b_it",

    # --- BigScience (Bloom) ---
    "bloomz-3b",
    "bloomz-7b1",

    # ----- GPT oss -------
    "gpt-oss-20b",
    
    # ----- Granite ibm ----
    "granite-3.3-2b",
    "granite-3.3-8b",

    # --- Liquid AI --- 
    "LFM2-1.2B",
    "LFM2-2.6B",
    "LFM2-8B",

    # ---- Tilde AI -----
    "TildeOpen-30b",

    # --- HF Models -----
    "SmolLM3-3B",
    "zephyr-7b",

    # ----- Swiss Ai Apertus ----
    "Apertus-8B-2509",
    "Apertus-70B-2509",
]

# --- Task directory ---
task_dir = "tasks"

# --- Command template ---
base_command = ["python", "Calling_LLM_Models.py"]

# --- Sequential runner ---
for model in models_to_run:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_log = os.path.join("logs", f"{model}_{timestamp}.log")

    logging.info(f"=== Starting model: {model} ===")
    print(f"Logging to: {model_log}")

    cmd = base_command + ["--model", model, "--task-dir", task_dir]

    # Open log file for both stdout + stderr
    with open(model_log, "w") as f:
        # Stream output to both file and terminal
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="")  # show live in terminal
            f.write(line)        # write to file

        process.wait()

    if process.returncode == 0:
        msg = f"Model {model} completed successfully."
        logging.info(msg)
        print(msg)
    else:
        msg = f"Model {model} exited with code {process.returncode}. See log: {model_log}"
        logging.error(msg)
        print(msg)

    # cooldown after each model
    time.sleep(60)

logging.info("=== All models completed ===")
print("\nAll models completed.")