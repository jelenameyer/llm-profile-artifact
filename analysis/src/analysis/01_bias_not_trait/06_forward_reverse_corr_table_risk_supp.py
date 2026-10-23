"""SI table: forward-reverse correlations of the reverse-keyed risk instruments
(BARRATT/BIS, SSSV, LOT, DFD) across all prompting conditions. Writes
results/tables/forward_reverse_corr_table_risk_supp.tex.

Needs two original Frey et al. (2017) files (lotteries.csv, dfd_perprob.csv), which
are not redistributed here; the script skips with instructions if either is absent."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from utils import corrs_table_from_conditions, to_latex_table, format_ci_cell

# The LOT and DFD human rows are rebuilt from the original Frey per-trial data, which is
# not re-hosted here but fetched from the public Frey OSF project by download_data.py.
LOT_SRC = Path("../../../source/risk_data/orig_human_data/lotteries.csv")
DFD_SRC = Path("../../../source/risk_data/orig_human_data/dfd_perprob.csv")
_missing = [p.name for p in (LOT_SRC, DFD_SRC) if not p.exists()]
if _missing:
    print(
        f"[skip] missing {', '.join(_missing)} (original Frey et al. 2017 data, normally fetched by\n"
        "       download_data.py). Run `python download_data.py` (or python run_all.py),\n"
        "       then re-run. Skipping for now.",
        file=sys.stderr,
    )
    sys.exit(77)

# load all dfs necessary
raw_risk_data = pd.read_csv("../../../data/risk_data/LLM_data_proc_prompts_direct/LLM_no_flip_data_raw.csv", low_memory=False)
raw_risk_data_flip = pd.read_csv("../../../data/risk_data/LLM_data_proc_prompts_direct/LLM_flip_data_raw.csv", low_memory=False)
raw_risk_data_humans = pd.read_csv("../../../data/risk_data/human_data_proc/raw_items_per_person.csv", low_memory=False)

# proprietary (API) models: added to the no-context, textgen condition only (they
# support only that variant and expose no token-level logits), mirroring the
# IPIP-NEO table in 05_forward_reverse_corr_table_supp.py.
api_data = pd.read_csv("../../../data/risk_data/api_data/LLM_api_no_flip_data_raw.csv", low_memory=False)
api_nr = api_data[api_data["model"].str.contains("_nr", regex=False)]
# raw LOT trials and DFD choices: used to rebuild the LOT and DFD human rows with
# position-coded responses (see the override blocks below).
raw_lotteries_humans = pd.read_csv(LOT_SRC, low_memory=False)
raw_dfd_humans = pd.read_csv(DFD_SRC, low_memory=False)

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
# Override DFD-human rows the same way. The stored DFD `score` is R (chose risky),
# whereas the LLM answer is the chosen POSITION; on reverse-keyed items (risky option
# listed second) the two are opposite, which would flip the sign of the human
# forward-reverse r. DFD order is fixed per gamble, so the LLM keying applies as is;
# rebuild score = chose option A (position 1), as in the DFD panel of
# 04_profile_instability/01_profiles_per_keying.py.
dfd_src = raw_dfd_humans.dropna(subset=["decision", "gamble_lab", "partid"]).copy()
dfd_humans_rebuilt = pd.DataFrame({
    "experiment": "DFD",
    "partid": dfd_src["partid"].to_numpy(),
    "item": dfd_src["gamble_lab"].to_numpy(),
    "score": (dfd_src["decision"].astype(str).str.upper() == "A").astype(float).to_numpy(),
    "category": np.nan,
    "item_id": dfd_src["gamble_lab"].map(_normalize_item_id).to_numpy(),
})
dfd_humans_rebuilt["reverse_coded"] = [
    reverse_map.get(("DFD", i)) for i in dfd_humans_rebuilt["item_id"]
]
raw_risk_data_humans = pd.concat(
    [raw_risk_data_humans[~raw_risk_data_humans["experiment"].isin(["LOT", "DFD"])],
     lot_humans_rebuilt, dfd_humans_rebuilt],
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
    category="risk_trait",
    trait_order=trait_order
)

# format all CI cells into two-line mini tables
trait_cols = [c for c in table_df.columns if c != "prompting condition"]
for idx in table_df.index:
    for col in trait_cols:
        table_df.at[idx, col] = format_ci_cell(table_df.at[idx, col], decimals=2)

latex = to_latex_table(
    table_df,
    float_format="%.2f",
    caption=("Forward--reverse item-mean correlation $\\rho(\\bar R_f, \\bar R_r)$ on the risk battery by "
             "(sub-)scale and prompting condition, restricted to (sub-)scales with at least one "
             "reverse-keyed item. Cells: point estimate above, Fisher-$z$ 95\\% CI below. "
             "Human baseline: Frey et al. (2017) sample, $n = 1{,}507$."),
    label="tab:forward_reverse_risk",
    escape=False,
)

# ten CI columns: shrink the font and column padding so the table fits the text width
latex = latex.replace("\\begin{tabular}", "\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular}", 1)

output_path = "../../../results/tables/forward_reverse_corr_table_risk_supp.tex"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(latex)
