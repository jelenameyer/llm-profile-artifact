"""SI table: forward-reverse correlations per Big Five trait across all prompting
conditions (context x text/logit x flipped). Writes
results/tables/forward_reverse_corr_table_supp.tex."""
import pandas as pd
from utils import corrs_table_from_conditions, to_latex_table, format_ci_cell

# load all dfs necessary
raw_ipipneo_data = pd.read_csv("../../../data/ipipneo300_data/llm_data/ipipneo_no_flip_data_raw.csv", low_memory=False)
raw_ipipneo_data_flip = pd.read_csv("../../../data/ipipneo300_data/llm_data/ipipneo_flip_data_raw.csv", low_memory=False)
raw_ipipneo_data_humans = pd.read_csv("../../../data/ipipneo300_data/human_data/ipipneo_human_raw.csv", low_memory=False)


api_data = pd.read_csv("../../../data/ipipneo300_data/api_data/ipipneo_api_data_raw.csv", low_memory=False)
api_nr = api_data[api_data["model"].str.contains("_nr", regex=False)]


# for humans convert to numerical response column
raw_ipipneo_data_humans["response"] = pd.to_numeric(
    raw_ipipneo_data_humans["response"].replace({"0": None, 0: None, " ": None}),
    errors="coerce"
)

# divide dfs by conditions and add api data to no context no flip condition only
raw_ipipneo_data_no_context = raw_ipipneo_data[raw_ipipneo_data["context_mode"] == "no_context"]
raw_ipipneo_data_no_context = pd.concat([raw_ipipneo_data_no_context, api_nr], ignore_index=True)
raw_ipipneo_data_with_context = raw_ipipneo_data[raw_ipipneo_data["context_mode"] == "with_context"]

raw_ipipneo_flip_data_no_context = raw_ipipneo_data_flip[raw_ipipneo_data_flip["context_mode"] == "no_context"]
raw_ipipneo_flip_data_with_context = raw_ipipneo_data_flip[raw_ipipneo_data_flip["context_mode"] == "with_context"]



# each df is already filtered to one condition
conditions = [
    {
        "name": "No context, textgen",
        "df": raw_ipipneo_data_no_context,
        "entity_var": "model",
        "dependent_var": "model_answer",
    },
    {
        "name": "With context, textgen",
        "df": raw_ipipneo_data_with_context,
        "entity_var": "model",
        "dependent_var": "model_answer",
    },
    {
        "name": "No context, logits",
        "df": raw_ipipneo_data_no_context,
        "entity_var": "model",
        "dependent_var": "logit_score",
    },
    {
        "name": "With context, logits",
        "df": raw_ipipneo_data_with_context,
        "entity_var": "model",
        "dependent_var": "logit_score",
    },
    {
        "name": "Flipped, no context, textgen",
        "df": raw_ipipneo_flip_data_no_context,
        "entity_var": "model",
        "dependent_var": "model_answer",
    },
    {
        "name": "Flipped, with context, textgen",
        "df": raw_ipipneo_flip_data_with_context,
        "entity_var": "model",
        "dependent_var": "model_answer",
    },
    {
        "name": "Flipped, no context, logits",
        "df": raw_ipipneo_flip_data_no_context,
        "entity_var": "model",
        "dependent_var": "logit_score",
    },
    {
        "name": "Flipped, with context, logits",
        "df": raw_ipipneo_flip_data_with_context,
        "entity_var": "model",
        "dependent_var": "logit_score",
    },
    {
        "name": "Human",
        "df": raw_ipipneo_data_humans,
        "entity_var": "person_id",
        "dependent_var": "response",
    },
]

table_df = corrs_table_from_conditions(
    conditions=conditions,
    condition_col="prompting condition",
    compare_to="Human",
    add_diff_rows=False,
)

# format all CI cells into two-line mini tables
trait_cols = [c for c in table_df.columns if c != "prompting condition"]
for idx in table_df.index:
    for col in trait_cols:
        table_df.at[idx, col] = format_ci_cell(table_df.at[idx, col], decimals=2)

latex = to_latex_table(
    table_df,
    float_format="%.2f",
    caption="Forward-reverse correlations by trait and condition",
    label="tab:forward_reverse_traits",
    escape=False,
)

output_path = "../../../results/tables/forward_reverse_corr_table_supp.tex"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(latex)
