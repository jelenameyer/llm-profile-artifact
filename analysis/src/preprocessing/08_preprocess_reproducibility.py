#!/usr/bin/env python3
"""Preprocess the replicate administrations of the large open-weight models into
one paired file: every item answered twice under identical requests (pass 1 and
rep2), both reasoning conditions, IPIP-NEO-300 and the 12 risk instruments.

Reads source/reproducibility/outputs_api_<run>{,_rep2}/ (per-model *_results.csv)
+ the IPIP-NEO item key.
Writes data/reproducibility/rep_pairs.csv with one row per (model, condition,
task, item): answer_1, answer_2, and for IPIP items the trait and keying.
Answers stay as recorded (no re-reversal); binary-choice tasks are coded
s1 = 1, s2 = 2. Missing = no extractable answer in that pass.
"""

# ----------- packages  ------------------------------------
import glob
import os
import re

import pandas as pd
from utils import create_traits_map, create_reverse_map

SRC = "../../source/reproducibility"
OUT_DIR = "../../data/reproducibility"
os.makedirs(OUT_DIR, exist_ok=True)

# pass-1 output dir per model; the replicate is the same name + "_rep2"
RUNS = {
    "vllm__zai-org__GLM-5.3-Flash": "outputs_api_large",
    "vllm__nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": "outputs_api_nemotron",
    "vllm__MiniMaxAI__MiniMax-M3": "outputs_api_minimax",
    "vllm__moonshotai__Kimi-K2.6": "outputs_api_kimi",
}

# ---------- Reading data ----------------------------------

def load_pass(folder):
    frames = []
    for f in glob.glob(os.path.join(SRC, folder, "*_results.csv")):
        cond, task = re.search(r"__(n?r)_([a-z]+)_results\.csv$", f).groups()
        d = pd.read_csv(f, low_memory=False)
        # unique item key: MPL repeats gamble_id across its rows, source_row_id is unique
        key = next(k for k in ("source_row_id", "gamble_id", "item_id") if k in d)
        if "model_answer_option" in d:   # binary-choice tasks (lot, dfd, mpl)
            answer = d["model_answer_option"].map({"s1": 1, "s2": 2})
        else:
            answer = pd.to_numeric(d["model_answer"], errors="coerce")
        frames.append(pd.DataFrame({
            "condition": cond, "task": task,
            "item": d[key].astype(str), "answer": answer,
        }))
    return pd.concat(frames, ignore_index=True)

# --------- process data -----------------------------------

pairs = []
for model, folder in RUNS.items():
    p1 = load_pass(folder)
    p2 = load_pass(folder + "_rep2")
    merged = p1.merge(p2, on=["condition", "task", "item"], how="outer",
                      suffixes=("_1", "_2"), validate="one_to_one")
    merged.insert(0, "model", model)
    pairs.append(merged)
pairs = pd.concat(pairs, ignore_index=True)

# IPIP-NEO trait and keying from the original item key
meta = pd.read_excel("../../source/ipipneo300_data/human_data/IPIP-NEO-ItemKey.xls",
                     sheet_name="IPIP-NEO-ItemKey")
traits = {str(k): v for k, v in create_traits_map(meta).items()}
reverse = {str(k): v for k, v in create_reverse_map(meta).items()}
is_ipip = pairs["task"] == "ipipneo"
pairs.loc[is_ipip, "traits"] = pairs.loc[is_ipip, "item"].map(traits)
pairs.loc[is_ipip, "reverse_coded"] = pairs.loc[is_ipip, "item"].map(reverse)

# ---------- save data -------------------------------
pairs.to_csv(os.path.join(OUT_DIR, "rep_pairs.csv"), index=False)
print(pairs.shape)
print(pairs.groupby(["model", "condition"]).size().unstack())
