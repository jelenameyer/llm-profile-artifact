#!/usr/bin/env python3
"""Preprocess the IPIP-NEO-300 proprietary-API LLM responses into long per-item
format (raw + recoded). API models carry no logits, so those columns are set null.

Reads data/raw/ipipneo300_data/outputs_api/ (per-model *_ipipneo*_results.csv) + the item key.
Writes data/intermediate/ipipneo300_data/api_data/ipipneo_api_data_{rereversed,raw}.csv.
"""

# ----------- packages  ------------------------------------
import pandas as pd
import numpy as np
from utils import create_traits_map, create_facet_map, create_reverse_map, load_dataframes

# ---------- Reading data ----------------------------------

# load data
IPIP_data = load_dataframes(task_name="ipipneo", path = "../../data/raw/ipipneo300_data/outputs_api")
IPIP_data["experiment"] = "IPIP-NEO-300"

# read metadata neo-ipip
meta_data_ipip = pd.read_excel("../../data/raw/ipipneo300_data/human_data/IPIP-NEO-ItemKey.xls", sheet_name=None)
meta_data_ipip = meta_data_ipip['IPIP-NEO-ItemKey']

# --------- process data -----------------------------------


# add columns with traits, facets, and reversed items keys (all from original material)
trait_map_ipip_neo_300 = create_traits_map(meta_data_ipip)
facet_map_ipip_neo_300 = create_facet_map(meta_data_ipip)
reversed_items_ipip_neo_300 = create_reverse_map(meta_data_ipip)

# add traits and facets and reversed map to llm ipip neo 300 data
IPIP_data["traits"] = IPIP_data["item_id"].map(trait_map_ipip_neo_300)
IPIP_data["category"] = IPIP_data["item_id"].map(facet_map_ipip_neo_300)
IPIP_data["reverse_coded"] = IPIP_data["item_id"].map(reversed_items_ipip_neo_300)

# add logit and probs column for later matching (plus a flag column, that non existend in these models)
IPIP_data["logit_score"] = None
IPIP_data["prob_score"] = None
IPIP_data["has_logits"] = False
IPIP_data["context_mode"] = "no_context" # add no context condition for all, to be comparable


# ------ for normal data ------------
# Apply mapping row-wise based on item number
ipip_reversed = IPIP_data.copy()

# flip back answers that where reverse coded
mask = (ipip_reversed["reverse_coded"] == True)
ipip_reversed.loc[mask, "model_answer"] = 6 - ipip_reversed.loc[mask, "model_answer"]


# ---------- save data -------------------------------
ipip_reversed.to_csv('../../data/intermediate/ipipneo300_data/api_data/ipipneo_api_data_rereversed.csv', index=False)
IPIP_data.to_csv('../../data/intermediate/ipipneo300_data/api_data/ipipneo_api_data_raw.csv', index=False)
