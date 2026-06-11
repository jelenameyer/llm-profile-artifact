#!/usr/bin/env python3
"""Direct generation path (open-source models) — entry point.

Runs open-source HF LLMs directly on the instrument items (no per-participant
human context), in both no-context and with-context modes. For each item it
generates a format-constrained answer via the outlines package
(https://github.com/dottxt-ai/outlines) and separately extracts the logits and
probabilities for every answer option.

Reads tasks/jsonl_data/*.jsonl (instrument items).
Writes <outputs-dir>/<model>_<task>_results.csv (answer + per-option logits/probs,
for each context mode).

Usage: python model_manager.py --model all --task-dir tasks
"""

# --- Packages ---
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import argparse
import logging
import os
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional
import importlib.util
import sys
from transformers import PreTrainedModel
import outlines

import random
random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model Configuration Dictionary (Fine-tuned/Instruction Decoder Models Only) ---
MODEL_CONFIGS = {
    #---- Llama Instruct Models ----
    "llama31_8b": {
        "model_key": "Llama-3.1-8B-Instruct",
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "large_model": False,
    },
    "llama32_1b": {
        "model_key": "Llama-3.2-1B-Instruct",
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "large_model": False,
    },
    "llama32_3b": {
        "model_key": "Llama-3.2-3B-Instruct",
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "large_model": False,
    },
    

    # --- Falcon Instruct ---
    "falcon3_1b": {
        "model_key": "Falcon-3-1B-Instruct",
        "model_name": "tiiuae/Falcon3-1B-Instruct",
        "large_model": False,
    },
    "falcon3_7b": {
        "model_key": "Falcon-3-7B-Instruct",
        "model_name": "tiiuae/Falcon3-7B-Instruct",
        "large_model": False,
    },
    "falcon3_10b": {
        "model_key": "Falcon-3-10B-Instruct",
        "model_name": "tiiuae/Falcon3-10B-Instruct",
        "large_model": False,
    },

    # --- Gemma Instruct ------
    "gemma-3-1b": {
        "model_key": "gemma-3-1b-it",
        "model_name": "google/gemma-3-1b-it",
        "large_model": False,
    },
     "gemma-3-4b": {
        "model_key": "gemma-3-4b-it",
        "model_name": "google/gemma-3-4b-it",
        "large_model": False,
    },
    "gemma-3-12b": {
        "model_key": "gemma-3-12b-it",
        "model_name": "google/gemma-3-12b-it",
        "large_model": False,
    },
    "gemma-3-27b": {
        "model_key": "gemma-3-27b-it",
        "model_name": "google/gemma-3-27b-it",
        "large_model": True,
    },
    "gemma-2-2b": {
        "model_key": "gemma-2-2b-it",
        "model_name": "google/gemma-2-2b-it",
        "large_model": False,
    },
    "gemma-2-9b": {
        "model_key": "gemma-2-9b-it",
        "model_name": "google/gemma-2-9b-it",
        "large_model": False,
    },
    "gemma-2-27b": {
        "model_key": "gemma-2-27b-it",
        "model_name": "google/gemma-2-27b-it",
        "large_model": True,
    },

    # --- Mistral ----
    "Mistral-7b-v0.3": {
        "model_key": "Mistral-7B-Instruct-v0.3",
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "large_model": False,
    },
    "Ministral-8b-2410": {
        "model_key": "Ministral-8B-Instruct-2410",
        "model_name": "mistralai/Ministral-8B-Instruct-2410",
        "large_model": False,
    },
    "mistral-24b-2501": {
        "model_key": "Mistral-Small-24B-Instruct-2501",
        "model_name": "mistralai/Mistral-Small-24B-Instruct-2501",
        "large_model": True,
    },
    
    # --- Qwen ---
    "Qwen3-1.7B": {
        "model_key": "Qwen3-1.7B",
        "model_name": "Qwen/Qwen3-1.7B",
        "large_model": False,
    },
    "Qwen3-4B": {
        "model_key": "Qwen3-4B",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "large_model": False,
    },
    "Qwen3-8B": {
        "model_key": "Qwen3-8B",
        "model_name": "Qwen/Qwen3-8B",
        "large_model": False,
    },
    "Qwen3-14B": {
        "model_key": "Qwen3-14B",
        "model_name": "Qwen/Qwen3-14B",
        "large_model": False,
    },
    "Qwen3-30B-A3B": {
        "model_key": "Qwen3-30B-A3B-Instruct-2507",
        "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "large_model": True,
    },
    "Qwen3-32B": {
        "model_key": "Qwen3-32B",
        "model_name": "Qwen/Qwen3-32B",
        "large_model": True,
    },

    "Qwen25-1.5B": {
        "model_key": "Qwen2.5-1.5B-Instruct",
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "large_model": False,
    },
    "Qwen25-3B": {
        "model_key": "Qwen2.5-3B-Instruct",
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "large_model": False,
    },
    "Qwen25-7B": {
        "model_key": "Qwen2.5-7B-Instruct",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "large_model": False,
    },
    "Qwen25-14B": {
        "model_key": "Qwen2.5-14B-Instruct",
        "model_name": "Qwen/Qwen2.5-14B-Instruct",
        "large_model": False,
    },
    "Qwen25-32B": {
        "model_key": "Qwen2.5-32B-Instruct",
        "model_name": "Qwen/Qwen2.5-32B-Instruct",
        "large_model": True,
    },

    # --- OLMo Instruct ---   
    "olmo2_7b_it": {
        "model_key": "OLMo-2-7B-Instruct",
        "model_name": "allenai/OLMo-2-1124-7B-Instruct",
        "large_model": False,
    },

    # --- BigScience (Bloom) ---
    "bloomz-3b": {
        "model_key": "bloomz-3b",
        "model_name": "bigscience/bloomz-3b",
        "large_model": False,
    },
    "bloomz-7b1": {
        "model_key": "bloomz-7b1",
        "model_name": "bigscience/bloomz-7b1",
        "large_model": False,
    },

    # ----- GPT oss -------
    "gpt-oss-20b": {
        "model_key": "gpt-oss-20b",
        "model_name": "openai/gpt-oss-20b",
        "large_model": True,
    },
    
    # ----- Granite ibm ----
    "granite-3.3-2b": {
        "model_key": "granite-3.3-2b-instruct",
        "model_name": "ibm-granite/granite-3.3-2b-instruct",
        "large_model": False,
    },
    "granite-3.3-8b": {
        "model_key": "granite-3.3-8b-instruct",
        "model_name": "ibm-granite/granite-3.3-8b-instruct",
        "large_model": False,
    },

    # --- Microsoft Phi ---
    "Phi-3.5-mini-4B": {
        "model_key": "Phi-3.5-mini-instruct",
        "model_name": "microsoft/Phi-3.5-mini-instruct",
        "large_model": False,
    },
    "Phi-3-mini-4B": {
        "model_key": "Phi-3-mini-128k-instruct",
        "model_name": "microsoft/Phi-3-mini-128k-instruct",
        "large_model": False,
    },
    "Phi-3-medium-14B": {
        "model_key": "Phi-3-medium-128k-instruct",
        "model_name": "microsoft/Phi-3-medium-128k-instruct",
        "large_model": False,
    },

    # --- Liquid AI ---
    "LFM2-1.2B": {
        "model_key": "LFM2-1.2B",
        "model_name": "LiquidAI/LFM2-1.2B",
        "large_model": False,
    },
    "LFM2-2.6B": {
        "model_key": "LFM2-2.6B",
        "model_name": "LiquidAI/LFM2-2.6B",
        "large_model": False,
    },
    "LFM2-8B": {
        "model_key": "LFM2-8B-A1B",
        "model_name": "LiquidAI/LFM2-8B-A1B",
        "large_model": False,
    },

    # --- HF Models -----
    "SmolLM3-3B": {
        "model_key": "SmolLM3-3B",
        "model_name": "HuggingFaceTB/SmolLM3-3B",
        "large_model": False,
    },
    "zephyr-7b": {
        "model_key": "zephyr-7b-beta",
        "model_name": "HuggingFaceH4/zephyr-7b-beta",
        "large_model": False,
    },
    # ----- Swiss Ai Apertus ----
    "Apertus-8B-2509": {
        "model_key": "Apertus-8B-Instruct-2509",
        "model_name": "swiss-ai/Apertus-8B-Instruct-2509",
        "large_model": False,
    },
    
    # ---- Tilde AI -----
    "TildeOpen-30b": {
        "model_key": "TildeOpen-30b",
        "model_name": "TildeAI/TildeOpen-30b",
        "large_model": True,
    },
    
    # ---- 70B Models
    "llama31_70b_it": {
        "model_key": "Llama-3.1-70B-Instruct",
        "model_name": "meta-llama/Llama-3.1-70B-Instruct",
        "large_model": True,
    },   
    "llama33_70b_it": {
        "model_key": "Llama-3.3-70B-Instruct",
        "model_name": "meta-llama/Llama-3.3-70B-Instruct",
        "large_model": True,
    }, 

    "Apertus-70B-2509": {
        "model_key": "Apertus-70B-Instruct-2509",
        "model_name": "swiss-ai/Apertus-70B-Instruct-2509",
        "large_model": True,
    },
}

class ModelManager:
    """Manages model loading and task execution across multiple models."""
    
    def __init__(self, outputs_dir: str = "outputs"):
        self.outputs_dir = outputs_dir
        self.model = None
        self.tokenizer = None
        self.outlines_model = None  
        self.current_model_key = None
        
        os.makedirs(self.outputs_dir, exist_ok=True)


    def load_model(self, model_name: str, model_key: str, model_config: dict) -> bool:
        """Load model and tokenizer with model-specific handling."""
        try:
            logging.info(f"Loading model '{model_name}'...")

            # Detect special model that needs custom handling
            is_gptoss = isinstance(model_name, str) and model_name.startswith("openai/gpt-oss-")
            is_phi3 = isinstance(model_name, str) and model_name.lower().startswith("microsoft/phi-3")
            # Detect large models
            is_large_model = model_config.get("large_model", False)

            # --- STEP 1: Prepare loading kwargs ---
            load_kwargs = {
                "device_map": "auto",
                "trust_remote_code": True,
                "low_cpu_mem_usage": False,
            }

            # GPT-OSS model: preserve MXFP4 quantization
            if is_gptoss:
                load_kwargs["torch_dtype"] = "auto"  # Don't force bf16, keep quantization
                
                # Warn if MXFP4 kernels aren't available
                try:
                    import importlib.util
                    if importlib.util.find_spec("kernels") is None:
                        logging.warning(
                            "OpenAI MXFP4 'kernels' package not found. Model may dequantize to bf16 and OOM. "
                            "Install with: pip install -U triton==3.4 kernels"
                        )
                except Exception:
                    pass
            else:
                # All other models: use bfloat16
                load_kwargs["torch_dtype"] = torch.bfloat16
                load_kwargs["attn_implementation"] = "eager"

            # Cap per-GPU VRAM to ~95% to avoid OOM during loading
            if is_large_model and torch.cuda.is_available() and torch.cuda.device_count() > 0:
                max_memory = {}
                for i in range(torch.cuda.device_count()):
                    total = torch.cuda.get_device_properties(i).total_memory
                    cap = int(total * 0.95)
                    max_memory[i] = cap
                load_kwargs["max_memory"] = max_memory
                pretty_caps = {k: f"{v/2**30:.1f}GiB" for k, v in max_memory.items()}
                logging.info(f"Large model detected: Using per-GPU memory caps: {pretty_caps}")

            # --- STEP 2: Load Model ---
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs
            )
            self.model.eval()

            # Normalize generation defaults toward greedy decoding.
            # Some checkpoints ship sampling params (e.g., top_p) in generation_config,
            # which can trigger warnings and unintended stochastic behavior.
            gen_cfg = getattr(self.model, "generation_config", None)
            if gen_cfg is not None:
                gen_cfg.do_sample = False
                for attr in ("top_p", "top_k", "temperature", "typical_p"):
                    if hasattr(gen_cfg, attr):
                        setattr(gen_cfg, attr, None)
                # Phi-3 remote modeling has recurring cache API incompatibilities.
                # Keep generation cache disabled for stability.
                if is_phi3 and hasattr(gen_cfg, "use_cache"):
                    gen_cfg.use_cache = False
            if is_phi3 and hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False

            # --- STEP 3: Load Tokenizer ---
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer.padding_side = 'right'
            
            # if pad and eos tokens are missing, define them separately, otherwise warnings when really prompting the models
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                self.model.resize_token_embeddings(len(self.tokenizer))

            if self.tokenizer.eos_token is None:
                self.tokenizer.add_special_tokens({"eos_token": "<|eos|>"})
                self.model.resize_token_embeddings(len(self.tokenizer))

            # --- Step 4: Create wrapper using OUTLINES package ---

            if is_phi3:
                logging.info("Phi-3 detected: skipping Outlines wrapper due known cache incompatibility")
                self.outlines_model = None
            else:
                try:
                    self.outlines_model = outlines.models.Transformers(
                        self.model,
                        self.tokenizer
                    )
                    logging.info("Outlines wrapper created successfully")
                except Exception as e:
                    logging.warning(f"Could not create outlines wrapper: {e}")
                    logging.warning("Falling back to unconstrained text generation")
                    self.outlines_model = None

            self.current_model_key = model_key
            logging.info(f"Model and tokenizer loaded successfully.")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load model '{model_name}'. Error: {e}")
            return False

    
    def unload_model(self):
        """Clean up model from memory."""
        from base_functions import clear_generator_cache
        clear_generator_cache()
        if self.outlines_model is not None:
            del self.outlines_model
            self.outlines_model = None
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer  
            self.tokenizer = None
        self.current_model_key = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logging.info(torch.cuda.memory_summary())
        logging.info("Model unloaded and memory cleared.")
    
    def load_task_module(self, task_path: str):
        """Dynamically load a task module."""
        spec = importlib.util.spec_from_file_location("task_module", task_path)
        task_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task_module)
        return task_module
    
    def discover_tasks(self, task_dir: str) -> List[str]:
        """Discover all .py files in the tasks directory."""
        task_dir_path = Path(task_dir)
        if not task_dir_path.exists():
            logging.error(f"Task directory does not exist: {task_dir}")
            return []
        
        task_files = list(task_dir_path.glob("*.py"))
        # Filter out __init__.py and other special files
        task_files = [str(f) for f in task_files if not f.name.startswith("__")]
        
        logging.info(f"Discovered {len(task_files)} task files in {task_dir}")
        for task_file in task_files:
            logging.info(f"  - {Path(task_file).name}")
        
        return task_files
    
    def run_task(self, task_module, task_name: str) -> Optional[pd.DataFrame]:
        """Run a single task on the current model."""
        if self.model is None or self.tokenizer is None:
            logging.error("No model loaded. Cannot run task.")
            return None
        
        try:
            logging.info(f"Running task '{task_name}' on model '{self.current_model_key}'")
            
            # Run the task - pass model, tokenizer, and model_key to the task
            result_df = task_module.run_task(
                model=self.model,
                tokenizer=self.tokenizer,
                outlines_model=self.outlines_model,
                model_key=self.current_model_key,
            )
            
            if result_df is not None and not result_df.empty:
                # Save results
                output_filename = f"{self.current_model_key}_{task_name}_results.csv"
                output_path = os.path.join(self.outputs_dir, output_filename)
                result_df.to_csv(output_path, index=False)
                logging.info(f"Task '{task_name}' completed. Results saved to {output_path}")
                # Force GPU cleanup after each task
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    torch.cuda.synchronize()
                return result_df
            else:
                logging.warning(f"!!Task '{task_name}' returned empty results.")
                return None
                
        except Exception as e:
            logging.error(f"Error running task '{task_name}' on model '{self.current_model_key}': {e}")
            return None
    
    def run_all_tasks_on_all_models(self, models_to_run: List[str], task_paths: List[str]):
        """Main orchestrator: run all tasks on all models."""
        
        # Load all task modules
        task_modules = {}
        for task_path in task_paths:
            task_name = os.path.splitext(os.path.basename(task_path))[0]
            try:
                task_modules[task_name] = self.load_task_module(task_path)
                logging.info(f"Loaded task module: {task_name}")
            except Exception as e:
                logging.error(f"Failed to load task module '{task_path}': {e}")
                continue
        
        if not task_modules:
            logging.error("No task modules loaded. Exiting.")
            return
        
        # Process each model
        for model_key in models_to_run:
            if model_key not in MODEL_CONFIGS:
                logging.error(f"Invalid model key: {model_key}")
                continue
                
            model_config = MODEL_CONFIGS[model_key]
            logging.info(f"\n=== Processing model: {model_config['model_key']} ===")
            
            # Load the model
            if not self.load_model(model_config["model_name"], model_config["model_key"], model_config=model_config):
                logging.error(f"Failed to load model {model_key}, skipping...")
                continue
                
            # Run all tasks on this model
            for task_name, task_module in task_modules.items():
                try:
                    self.run_task(task_module, task_name)
                except Exception as e:
                    logging.error(f"Error running task {task_name} on model {model_key}: {e}")
                    continue
            
            # Unload model before moving to next one
            self.unload_model()
        
        logging.info("\n=== All models and tasks processed ===")

def main():
    parser = argparse.ArgumentParser(description="Run multiple tasks across multiple models.")
    parser.add_argument("--models", type=str, nargs="+", 
                        help="Space-separated list of specific models to run")
    parser.add_argument("--model", type=str, choices=list(MODEL_CONFIGS.keys()) + ["all"],
                        default="all", help="Single model to run (or 'all' for all models)")

    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--tasks", type=str, nargs="+",
                        help="Space-separated list of task script paths")
    task_group.add_argument("--task-dir", type=str,
                        help="Directory containing task scripts (will run all .py files)")
    parser.add_argument("--outputs-dir", type=str, default="outputs",
                        help="Directory to save output files")

    args = parser.parse_args()
    
    # Determine which models to run
    if args.models:
        models_to_run = args.models
    elif args.model == "all":
        models_to_run = list(MODEL_CONFIGS.keys())
    else:
        models_to_run = [args.model]
    
    # Validate model choices
    invalid_models = [m for m in models_to_run if m not in MODEL_CONFIGS]
    if invalid_models:
        logging.error(f"Invalid model(s): {invalid_models}")
        logging.error(f"Available models: {list(MODEL_CONFIGS.keys())}")
        return
    
    # Get task paths
    if args.task_dir:
        manager = ModelManager(outputs_dir=args.outputs_dir)
        task_paths = manager.discover_tasks(args.task_dir)
        if not task_paths:
            logging.error("No tasks found or task directory doesn't exist.")
            return
    else:
        task_paths = args.tasks
        # Validate task files exist
        for task_path in task_paths:
            if not os.path.exists(task_path):
                logging.error(f"Task file does not exist: {task_path}")
                return
    
    logging.info(f"Running evaluation on models: {models_to_run}")
    logging.info(f"Tasks to run: {[os.path.basename(t) for t in task_paths]}")
    logging.info(f"Output directory: {args.outputs_dir}")
    
    # Initialize model manager and run everything
    manager = ModelManager(outputs_dir=args.outputs_dir)
    manager.run_all_tasks_on_all_models(models_to_run, task_paths)

if __name__ == "__main__":
    main()
