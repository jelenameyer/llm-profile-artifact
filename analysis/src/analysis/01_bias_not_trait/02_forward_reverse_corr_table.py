"""Main-text Table 1: forward-reverse item-mean correlations per Big Five trait,
LLMs vs humans (IPIP-NEO-300). Writes results/tables/forward_reverse_corr_table.tex."""
import pandas as pd
import re
from utils import corrs_table_from_conditions, format_ci_cell

# load all dfs necessary
raw_ipipneo_data = pd.read_csv("../../../data/intermediate/ipipneo300_data/llm_data/ipipneo_no_flip_data_raw.csv", low_memory=False)
raw_ipipneo_data_humans = pd.read_csv("../../../data/intermediate/ipipneo300_data/human_data/ipipneo_human_raw.csv", low_memory=False)
# api data
api_data = pd.read_csv("../../../data/intermediate/ipipneo300_data/api_data/ipipneo_api_data_raw.csv", low_memory=False)
api_nr = api_data[api_data["model"].str.contains("_nr", regex=False)] # only use no-reasoning condition here
raw_ipipneo_data = pd.concat([raw_ipipneo_data, api_nr], ignore_index=True) # add api data to other raw llm data

# for humans convert to numerical response column
raw_ipipneo_data_humans["response"] = pd.to_numeric(
    raw_ipipneo_data_humans["response"].replace({"0": None, 0: None, " ": None}),
    errors="coerce"
)

# divide dfs by conditions 
raw_ipipneo_data_no_context = raw_ipipneo_data[raw_ipipneo_data["context_mode"] == "no_context"]

# filter LLMs with any variance in answers (exclude constant-response models)
model_var = (
    raw_ipipneo_data_no_context
    .groupby("model")["model_answer"]
    .var()
    .fillna(0)
)
models_with_var = model_var[model_var > 0].index
raw_ipipneo_data_no_context_var = raw_ipipneo_data_no_context[
    raw_ipipneo_data_no_context["model"].isin(models_with_var)
]
n_models_var = raw_ipipneo_data_no_context_var["model"].nunique()

# each df is already filtered to one condition
conditions = [
    {
        "name": "LLMs",
        "df": raw_ipipneo_data_no_context,
        "entity_var": "model",
        "dependent_var": "model_answer",
    },
    {
        "name": f"LLMs (var>0; N={n_models_var})",
        "df": raw_ipipneo_data_no_context_var,
        "entity_var": "model",
        "dependent_var": "model_answer",
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
    condition_col="population",
    compare_to="Human",
    add_diff_rows=False,
)

# reorder rows: LLMs, LLMs (var>0), Human
desired_order = ["LLMs", f"LLMs (var>0; N={n_models_var})", "Human"]
order_map = {name: i for i, name in enumerate(desired_order)}
table_df["_order"] = table_df["population"].map(order_map).fillna(999).astype(int)
table_df = table_df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


# format LLMs/Human rows to have 2-line cells with CI and no leading zero in CI
trait_cols = [c for c in table_df.columns if c != "population"]
for idx, row in table_df.iterrows():
    if row["population"] in {"LLMs", f"LLMs (var>0; N={n_models_var})", "Human"}:
        for col in trait_cols:
            table_df.at[idx, col] = format_ci_cell(row[col], decimals=2)

latex = table_df.to_latex(
    index=False,
    escape=False,
    caption="Forward-reverse correlations by trait for LLMs versus humans",
    label="tab:forward_reverse_traits_small",
    column_format="lccccc",
)

# customize table wrapper and spacing
latex = latex.replace("\\begin{table}\n", "\\begin{table}[t!]\n")
latex = latex.replace("\\centering\n", "")
latex = latex.replace(
    "\\label{tab:forward_reverse_traits_small}\n",
    "\\label{tab:forward_reverse_traits_small}\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n",
)
latex = re.sub(r"(LLMs\\s*&.*?\\\\)\n\\s*\n(Human\\s*&)", r"\\1\n\\2", latex, count=1)

output_path = "../../../results/tables/forward_reverse_corr_table.tex"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(latex)
