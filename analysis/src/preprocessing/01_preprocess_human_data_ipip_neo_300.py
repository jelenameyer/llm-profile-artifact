#!/usr/bin/env python3
"""Preprocess the IPIP-NEO-300 human reference data into long per-item format.

Reads source/ipipneo300_data/human_data/ (ipip20993.dat + IPIP-NEO-ItemKey.xls).
Writes data/ipipneo300_data/human_data/ipipneo_human{,_raw}.csv.
"""

# ----------- packages  ------------------------------------
import pandas as pd
import numpy as np
from utils import create_traits_map, create_facet_map, create_reverse_map

# ---------- Reading data ----------------------------------
# read neo-ipip data
neo_ipip_data = pd.read_csv("../../source/ipipneo300_data/human_data/ipip20993.dat", sep="\t")

# read metadata neo-ipip
meta_data_ipip = pd.read_excel("../../source/ipipneo300_data/human_data/IPIP-NEO-ItemKey.xls", sheet_name=None)
meta_data_ipip = meta_data_ipip['IPIP-NEO-ItemKey']



# --------- process data ------------------------------------

# convert data to long format to have score of person per item

# create person id
neo_ipip_data = neo_ipip_data.reset_index().rename(columns={"index": "person_id"})

# select only columns like I1, I2, ..., I300
i_cols = neo_ipip_data.filter(regex=r"^I\d+$").columns

# reshape to long format
neo_ipip_data_long = neo_ipip_data.melt(
    id_vars="person_id",
    value_vars=i_cols,
    var_name="item",
    value_name="response"
)

# convert item labels I1 -> 1, I2 -> 2, ...
neo_ipip_data_long["item"] = neo_ipip_data_long["item"].str.extract(r"(\d+)").astype(int)


# add columns with traits, facets, and reversed items keys (all from original material)
trait_map_ipip_neo_300 = create_traits_map(meta_data_ipip)
facet_map_ipip_neo_300 = create_facet_map(meta_data_ipip)
reversed_items_ipip_neo_300 = create_reverse_map(meta_data_ipip)

# add traits and facets and reverse map to long ipip neo 300 data
neo_ipip_data_long["traits"] = neo_ipip_data_long["item"].map(trait_map_ipip_neo_300)
neo_ipip_data_long["category"] = neo_ipip_data_long["item"].map(facet_map_ipip_neo_300)
neo_ipip_data_long["reverse_coded"] = neo_ipip_data_long["item"].map(reversed_items_ipip_neo_300)

# reproduce RAW data (see original docu, they have already reversed data, so we need to bring in raw form again.)

# make copy of data 
neo_ipip_data_long_raw = neo_ipip_data_long.copy() 

# flip back answers that where reverse coded - to RAW form
mask = (neo_ipip_data_long_raw["reverse_coded"] == True)
neo_ipip_data_long_raw.loc[mask, "response"] = 6 - neo_ipip_data_long_raw.loc[mask, "response"]


# ---------- save data -------------------------------
# save IPIP-NEO-300 raw and processed forms of data in long format
neo_ipip_data_long_raw.to_csv('../../data/ipipneo300_data/human_data/ipipneo_human_raw.csv', index=False)
neo_ipip_data_long.to_csv('../../data/ipipneo300_data/human_data/ipipneo_human.csv', index=False)

print("All done!")