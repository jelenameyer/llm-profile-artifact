#!/usr/bin/env python3
"""Preprocess the IPIP-NEO-300 open-weight LLM responses into long per-item format
(raw + recoded; flipped + no-flip variants).

Reads data/raw/ipipneo300_data/llm_data/ (per-model *_ipipneo*_results.csv) + the item key.
Writes the six data/intermediate/ipipneo300_data/llm_data/ipipneo_*.csv variants.
"""

# ----------- packages  ------------------------------------
import pandas as pd
import numpy as np
from utils import create_traits_map, create_facet_map, create_reverse_map, load_dataframes, add_argmax_column

# ---------- Reading data ----------------------------------

# load data
IPIP_data = load_dataframes(task_name="ipipneo")
IPIP_data["experiment"] = "IPIP-NEO-300"

# read metadata neo-ipip
meta_data_ipip = pd.read_excel("../../data/raw/ipipneo300_data/human_data/IPIP-NEO-ItemKey.xls", sheet_name=None)
meta_data_ipip = meta_data_ipip['IPIP-NEO-ItemKey']

# --------- process data -----------------------------------

# add argmax logit and probability coloumn
IPIP_data = add_argmax_column(IPIP_data)

# add columns with traits, facets, and reversed items keys (all from original material)
trait_map_ipip_neo_300 = create_traits_map(meta_data_ipip)
facet_map_ipip_neo_300 = create_facet_map(meta_data_ipip)
reversed_items_ipip_neo_300 = create_reverse_map(meta_data_ipip)

# add traits and facets and reversed map to llm ipip neo 300 data
IPIP_data["traits"] = IPIP_data["item_id"].map(trait_map_ipip_neo_300)
IPIP_data["category"] = IPIP_data["item_id"].map(facet_map_ipip_neo_300)
IPIP_data["reverse_coded"] = IPIP_data["item_id"].map(reversed_items_ipip_neo_300)


# seperate the datasets in non-flipped and flipped version (plus version that is re-flipped)
raw_ipip = IPIP_data[IPIP_data["flipped"] == False]
raw_ipip_flipped = IPIP_data[IPIP_data["flipped"] == True]
ipip_flipped_back = raw_ipip_flipped.copy()
ipip_flipped_back["model_answer"] = 6 - ipip_flipped_back["model_answer"]
ipip_flipped_back["logit_score"] = 6 - ipip_flipped_back["logit_score"]
ipip_flipped_back["prob_score"] = 6 - ipip_flipped_back["prob_score"]



# ------ for normal data ------------
# Apply mapping row-wise based on item number
ipip_reversed = raw_ipip.copy()

# flip back answers that where reverse coded
mask = (ipip_reversed["reverse_coded"] == True)
ipip_reversed.loc[mask, "model_answer"] = 6 - ipip_reversed.loc[mask, "model_answer"]
ipip_reversed.loc[mask, "logit_score"] = 6 - ipip_reversed.loc[mask, "logit_score"]
ipip_reversed.loc[mask, "prob_score"] = 6 - ipip_reversed.loc[mask, "prob_score"]


# ------ for re-flipped data ------------
# Apply mapping row-wise based on item number
ipip_flipped_back_reversed_back = ipip_flipped_back.copy()

# flip back answers that where reverse coded
mask = (ipip_flipped_back_reversed_back["reverse_coded"] == True)
ipip_flipped_back_reversed_back.loc[mask, "model_answer"] = 6 - ipip_flipped_back_reversed_back.loc[mask, "model_answer"]
ipip_flipped_back_reversed_back.loc[mask, "logit_score"] = 6 - ipip_flipped_back_reversed_back.loc[mask, "logit_score"]
ipip_flipped_back_reversed_back.loc[mask, "prob_score"] = 6 - ipip_flipped_back_reversed_back.loc[mask, "prob_score"]

# organize outcoming dfs

# all reflipped and re-reversed together
all_data_reflipped_and_rereversed = pd.concat([ipip_reversed, ipip_flipped_back_reversed_back], ignore_index=True)
# all non-reflipped and non-re-reversed together
all_data_raw = IPIP_data.copy()

# only non-flipped data (once re-reversed, once raw) 
no_flip_data_rereversed = ipip_reversed
no_flip_data_raw = raw_ipip

# only flipped data (once re-reversed, once raw) 
flip_data_rereversed = ipip_flipped_back_reversed_back
flip_data_raw = raw_ipip_flipped


# ---------- save data -------------------------------
all_data_reflipped_and_rereversed.to_csv('../../data/intermediate/ipipneo300_data/llm_data/ipipneo_all_data_reflipped_and_rereversed.csv', index=False)
all_data_raw.to_csv('../../data/intermediate/ipipneo300_data/llm_data/ipipneo_all_data_raw.csv', index=False)

no_flip_data_rereversed.to_csv('../../data/intermediate/ipipneo300_data/llm_data/ipipneo_no_flip_data_rereversed.csv', index=False)
no_flip_data_raw.to_csv('../../data/intermediate/ipipneo300_data/llm_data/ipipneo_no_flip_data_raw.csv', index=False)

flip_data_rereversed.to_csv('../../data/intermediate/ipipneo300_data/llm_data/ipipneo_flip_data_rereversed.csv', index=False)
flip_data_raw.to_csv('../../data/intermediate/ipipneo300_data/llm_data/ipipneo_flip_data_raw.csv', index=False)
