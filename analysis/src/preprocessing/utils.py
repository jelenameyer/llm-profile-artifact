# packages --------------------------------------------------------------------
import pandas as pd
import numpy as np
import glob
import os

# Loading all data files of one task ------------------------------------------------------------

def load_dataframes(task_name, path = "../../data/raw/ipipneo300_data/llm_data"):

    # Initialize empty list to store DataFrames
    dataframe = []

    for file in glob.glob(os.path.join(path, f"*_{task_name}*_results.csv")):
        # Read the CSV
        df = pd.read_csv(file)
        
        # Append to list
        dataframe.append(df)
        
    # Concatenate all DataFrames into one big DataFrame
    merged_data = pd.concat(dataframe, ignore_index=True)

    print(f"Merged DataFrame shape: {merged_data.shape}")
    print(f"Total models: {merged_data['model'].nunique()}")

    return(merged_data)



# filter out probability LLM assigned to real item answer  ------------------------------------------
def filter_pred_prob(data, human_col = "human_number"):
    data["prob_pred"] = data.apply(
        lambda row: row[f"prob_{row[human_col]}"], axis=1
    )
    return(data)



# -----------------------------------Helper for post-porcessing------------------------------------------------

# function to extract traits per item
def create_traits_map(df):
    trait_map = (
        df.assign(trait=df["Key"].str[0])
          .set_index("Full#")["trait"]
          .to_dict()
    )
    return trait_map


# function to extract facets per item
def create_facet_map(df):
    facet_map = (
        df.assign(facet=df["Facet"])
          .set_index("Full#")["facet"]
          .to_dict()
    )
    return facet_map

# function to extract frward-reverse item mapping
def create_reverse_map(df):
    reverse_map = (
        df.assign(reversed=df["Sign"].str.startswith("-"))
          .set_index("Full#")["reversed"]
          .to_dict()
    )
    return reverse_map


# calculate score for prob and logits

def parse_score(colname):
    suffix = colname.split("_", 1)[1]   # everything after "prob_"
    try:
        return int(suffix)
    except ValueError:
        return suffix
    
def add_argmax_column(df):
    log_cols = [col for col in df.columns if col.startswith("logit_")]
    prob_cols = [col for col in df.columns if col.startswith("prob_")]

    df["logit_score"] = (
        df[log_cols]
        .idxmax(axis=1)
        .apply(parse_score)
    )

    df["prob_score"] = (
        df[prob_cols]
        .idxmax(axis=1)
        .apply(parse_score)
    )

    return df




