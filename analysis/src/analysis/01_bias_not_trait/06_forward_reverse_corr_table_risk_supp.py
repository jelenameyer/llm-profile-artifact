"""SI table: forward-reverse correlations of the reverse-keyed risk instruments
(BARRATT/BIS, SSSV, LOT, DFD) across all prompting conditions. Writes
results/tables/forward_reverse_corr_table_risk_supp.tex.

Needs the original Frey et al. (2017) lottery data (lotteries.csv), which is not
redistributed here; the script skips with instructions if it is absent."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from utils import corrs_table_from_conditions, to_latex_table, format_ci_cell

# The LOT human rows are rebuilt from the original per-trial Frey lottery data, which
# is not shipped with this repo (see README, "Human data"). Skip if it is missing.
LOT_SRC = Path("../../../data/raw/risk_data/orig_human_data/lotteries.csv")
if not LOT_SRC.exists():
    print(
        f"[skip] needs original Frey et al. (2017) data not redistributed here: {LOT_SRC.name}.\n"
        "       Download from https://osf.io/rce7g/ and place it at\n"
        f"       analysis/data/raw/risk_data/orig_human_data/{LOT_SRC.name}, then re-run.",
        file=sys.stderr,
    )
    sys.exit(77)

# load all dfs necessary
raw_risk_data = pd.read_csv("../../../data/intermediate/risk_data/LLM_data_proc_prompts_direct/LLM_no_flip_data_raw.csv", low_memory=False)
raw_risk_data_flip = pd.read_csv("../../../data/intermediate/risk_data/LLM_data_proc_prompts_direct/LLM_flip_data_raw.csv", low_memory=False)
raw_risk_data_humans = pd.read_csv("../../../data/intermediate/risk_data/human_data_proc/raw_items_per_person.csv", low_memory=False)

# proprietary (API) models: added to the no-context, textgen condition only (they
# support only that variant and expose no token-level logits), mirroring the
# IPIP-NEO table in 05_forward_reverse_corr_table_supp.py.
api_data = pd.read_csv("../../../data/intermediate/risk_data/api_data/LLM_api_no_flip_data_raw.csv", low_memory=False)
api_nr = api_data[api_data["model"].str.contains("_nr", regex=False)]
# raw LOT trials: used to rebuild the LOT human rows with per-trial keying (see the
# LOT override block below).
raw_lotteries_humans = pd.read_csv(LOT_SRC, low_memory=False)

# for humans convert to numerical response column
# Note: do NOT map 0 -> None — for binary risk tasks (LOT, DFD, MPL, BART, CCT)
# 0 is a valid response (e.g. "did not choose risky"). The four Likert scales
# (BARRAT, DOSPERT, SSSV, SOEP) contain no zeros, so " " is the only sentinel needed.
raw_risk_data_humans["score"] = pd.to_numeric(
    raw_risk_data_humans["score"].replace({" ": None}),
    errors="coerce"
)

# normalize experiment names to match LLM codes
EXPERIMENT_MAP = {
    "BARRAT scale": "BARRATT",
    "BARRATT scale": "BARRATT",
    "SSSV scale": "SSSV",
    "LOT task": "LOT",
    "DFD task": "DFD",
}
raw_risk_data_humans["experiment"] = raw_risk_data_humans["experiment"].map(EXPERIMENT_MAP).fillna(raw_risk_data_humans["experiment"])

def _normalize_item_id(x):
    if pd.isna(x):
        return None
    s = str(x)
    if s.endswith(".0"):
        s = s[:-2]
    return s

# build reverse-coded map from LLM data to apply to human data
llm_item_map = (
    raw_risk_data[["experiment", "item_id", "reverse_coded"]]
    .dropna(subset=["reverse_coded"])
    .assign(item_id=lambda d: d["item_id"].map(_normalize_item_id))
    .drop_duplicates()
)
reverse_map = {
    (row["experiment"], row["item_id"]): row["reverse_coded"]
    for _, row in llm_item_map.iterrows()
}
raw_risk_data_humans["item_id"] = raw_risk_data_humans["item"].map(_normalize_item_id)
raw_risk_data_humans["reverse_coded"] = raw_risk_data_humans.apply(
    lambda r: reverse_map.get((r["experiment"], r["item_id"])),
    axis=1,
)

# Override LOT-human rows with per-trial position coding (must match the LOT panel in
# 04_profile_instability/01_profiles_per_keying.py). LOT presentation order was randomised per trial
# (Presentation_XZ), so the fixed reverse_coded[Dec_ID] applied above is wrong for
# humans and the stored `score` is R (chose risky), not position. Rebuild: score =
# chose first-presented option (mirrors the LLM); reverse_coded = risky gamble shown
# second. Risky gamble per Frey table S8: X if Dec_ID >= 16, else Z.
lot_src = raw_lotteries_humans.dropna(
    subset=["Decision_X", "Presentation_XZ", "Dec_ID", "partid"]
).copy()
lot_dx = lot_src["Decision_X"].astype(int)
lot_pxz = lot_src["Presentation_XZ"].astype(int)
lot_risky_is_X = lot_src["Dec_ID"].astype(int) >= 16
lot_humans_rebuilt = pd.DataFrame({
    "experiment": "LOT",
    "partid": lot_src["partid"].to_numpy(),
    "item": lot_src["Dec_ID"].astype(int).to_numpy(),
    "score": np.where(lot_pxz == 1, lot_dx, 1 - lot_dx).astype(float),
    "category": np.nan,
    "item_id": lot_src["Dec_ID"].astype(int).astype(str).to_numpy(),
    "reverse_coded": np.where(lot_risky_is_X, lot_pxz == 0, lot_pxz == 1),
})
raw_risk_data_humans = pd.concat(
    [raw_risk_data_humans[raw_risk_data_humans["experiment"] != "LOT"], lot_humans_rebuilt],
    ignore_index=True,
)

def _risk_trait(df):
    return df["category"].where(df["category"].notna() & (df["category"] != ""), df["experiment"])

# keep only scales/subscales that include reverse-coded items
allowed_traits = set(
    _risk_trait(raw_risk_data[raw_risk_data["reverse_coded"] == True]).dropna().unique()
)

# explicit, readable column order for the risk traits
trait_order = [t for t in ["BISn", "BISm", "BISa", "SSdis", "SSbor", "SSexp", "SStas", "LOT", "DFD"] if t in allowed_traits]
trait_order += [t for t in sorted(allowed_traits) if t not in trait_order]
# divide dfs by conditions and add api data to the no-context (textgen) condition only
raw_risk_data_no_context = raw_risk_data[raw_risk_data["context_mode"] == "no_context"]
raw_risk_data_no_context = pd.concat([raw_risk_data_no_context, api_nr], ignore_index=True)
raw_risk_data_with_context = raw_risk_data[raw_risk_data["context_mode"] == "with_context"]

raw_risk_flip_data_no_context = raw_risk_data_flip[raw_risk_data_flip["context_mode"] == "no_context"]
raw_risk_flip_data_with_context = raw_risk_data_flip[raw_risk_data_flip["context_mode"] == "with_context"]


def _filter_and_tag(df):
    df = df.copy()
    df["risk_trait"] = _risk_trait(df)
    df = df[df["risk_trait"].isin(allowed_traits)]
    df = df[df["reverse_coded"].isin([True, False])]
    return df

raw_risk_data_no_context = _filter_and_tag(raw_risk_data_no_context)
raw_risk_data_with_context = _filter_and_tag(raw_risk_data_with_context)
raw_risk_flip_data_no_context = _filter_and_tag(raw_risk_flip_data_no_context)
raw_risk_flip_data_with_context = _filter_and_tag(raw_risk_flip_data_with_context)
raw_risk_data_humans = _filter_and_tag(raw_risk_data_humans)


# each df is already filtered to one condition
conditions = [
    {
        "name": "No context, textgen",
        "df": raw_risk_data_no_context,
        "entity_var": "model",
        "dependent_var": "model_answer",
    },
    {
        "name": "With context, textgen",
        "df": raw_risk_data_with_context,
        "entity_var": "model",
        "dependent_var": "model_answer",
    },
    {
        "name": "No context, logits",
        "df": raw_risk_data_no_context,
        "entity_var": "model",
        "dependent_var": "logit_score",
    },
    {
        "name": "With context, logits",
        "df": raw_risk_data_with_context,
        "entity_var": "model",
        "dependent_var": "logit_score",
    },
    {
        "name": "Flipped, no context, textgen",
        "df": raw_risk_flip_data_no_context,
        "entity_var": "model",
        "dependent_var": "model_answer",
    },
    {
        "name": "Flipped, with context, textgen",
        "df": raw_risk_flip_data_with_context,
        "entity_var": "model",
        "dependent_var": "model_answer",
    },
    {
        "name": "Flipped, no context, logits",
        "df": raw_risk_flip_data_no_context,
        "entity_var": "model",
        "dependent_var": "logit_score",
    },
    {
        "name": "Flipped, with context, logits",
        "df": raw_risk_flip_data_with_context,
        "entity_var": "model",
        "dependent_var": "logit_score",
    },
    {
        "name": "Human",
        "df": raw_risk_data_humans,
        "entity_var": "partid",
        "dependent_var": "score",
    },
]

table_df = corrs_table_from_conditions(
    conditions=conditions,
    condition_col="prompting condition",
    compare_to="Human",
    add_diff_rows=False,
    category="risk_trait",
    trait_order=trait_order
)

# simplify diff row labels to avoid repeating condition names
table_df["prompting condition"] = table_df["prompting condition"].str.replace(
    r"^\$\\Delta z'\$\s*\(.*\)$", lambda _: r"$\Delta z'$", regex=True
)
table_df["prompting condition"] = table_df["prompting condition"].str.replace(
    r"^z-statistic\s*\(.*\)$", "z-statistic", regex=True
)

# format all CI cells into two-line mini tables
trait_cols = [c for c in table_df.columns if c != "prompting condition"]
for idx in table_df.index:
    for col in trait_cols:
        table_df.at[idx, col] = format_ci_cell(table_df.at[idx, col], decimals=2)

latex = to_latex_table(
    table_df,
    float_format="%.2f",
    caption="Forward-reverse correlations of risk tasks by condition",
    label="tab:forward_reverse_risk",
    escape=False,
)

# insert a \midrule after each z-statistic row
lines = latex.splitlines()
try:
    start = lines.index(r"\midrule") + 1
    end = lines.index(r"\bottomrule")
    body = lines[start:end]
    new_body = []
    for line in body:
        new_body.append(line)
        if line.lstrip().startswith("z-statistic") and line.strip().endswith(r"\\"):
            new_body.append(r"\midrule")
    latex = "\n".join(lines[:start] + new_body + lines[end:])
except ValueError:
    pass

output_path = "../../../results/tables/forward_reverse_corr_table_risk_supp.tex"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(latex)
