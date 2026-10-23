"""Generalisable orthogonality table: cross-instrument correlation between p_r and
mean inter-item correlation per prompting condition, with a small/large size split.
Backs main-text Fig 4 and SI Table tab:robustness_prompting_risk_orthognality.

Reads risk_data + ipipneo300_data (LLM/api/human, all conditions).
Writes results/tables/orthogonality_corr_by_condition.{csv,tex}.
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import sys

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(1, str(BASE_DIR.parent))   # ../panel.py

from panel import MODEL_SIZE, bare_model_name, low_variability_models

from utils import (
    compute_mean_interitem_corr,
    normalize_experiment_names,
    apply_domain_fixes as _apply_domain_fixes,
    compute_pct_reversed_map,
    add_pct_reversed,
    pick_item_col,
    pick_individual_col,
    cluster_bootstrap_ci,
    wild_cluster_p,
)

DATA_DIR = (BASE_DIR / "../../../data/risk_data").resolve()
IPIP_DIR = (BASE_DIR / "../../../data/ipipneo300_data").resolve()
OUTPUT_DIR = (BASE_DIR / "../../../results/tables").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "orthogonality_corr_by_condition.csv"
OUTPUT_TEX = OUTPUT_DIR / "orthogonality_corr_by_condition.tex"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def corr_ci_fisher(r, n, alpha=0.05):
    if not np.isfinite(r) or n < 4:
        return (np.nan, np.nan)
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    lo = z - zcrit * se
    hi = z + zcrit * se
    return (np.tanh(lo), np.tanh(hi))


def format_corr_ci(r, ci, decimals=2):
    if not np.isfinite(r):
        return "NaN"
    lo, hi = ci
    if np.isfinite(lo) and np.isfinite(hi):
        return f"{r:+.{decimals}f} [{lo:+.{decimals}f}, {hi:+.{decimals}f}]"
    return f"{r:+.{decimals}f}"


def _pearson_for_subset(data, score_col, item_col, individual_col, pct_map):
    if data.empty:
        return np.nan, 0
    mean_corr = compute_mean_interitem_corr(
        data, score=score_col, item_name=item_col, individual=individual_col,
    )
    mean_corr = mean_corr.rename(columns={"category": "domain"})
    mean_corr = _apply_domain_fixes(mean_corr, is_llm=(individual_col == "model"))
    mean_corr = add_pct_reversed(mean_corr, pct_map)
    df_corr = mean_corr[["pct_reversed_keyed_items", "mean_interitem_corr"]].dropna()
    n = len(df_corr)
    if n < 2:
        return np.nan, n
    x = df_corr["pct_reversed_keyed_items"].to_numpy()
    y = df_corr["mean_interitem_corr"].to_numpy()
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, n
    r, _ = stats.pearsonr(x, y)
    return r, n


def subscale_table(df, score_col, pct_map, context_mode=None):
    """Per-subscale (experiment, domain) p_r and mean inter-item correlation: the
    points behind one cross-instrument correlation."""
    data = df.copy()
    if context_mode is not None and "context_mode" in data.columns:
        data = data[data["context_mode"] == context_mode]
    individual_col = pick_individual_col(data)
    mean_corr = compute_mean_interitem_corr(
        data,
        score=score_col,
        item_name=pick_item_col(data),
        individual=individual_col,
    )
    mean_corr = mean_corr.rename(columns={"category": "domain"})
    mean_corr = _apply_domain_fixes(mean_corr, is_llm=(individual_col == "model"))
    return add_pct_reversed(mean_corr, pct_map)


def compute_condition_summary(df, score_col, pct_map, context_mode=None, label=None,
                              small_models=None, large_models=None):
    data = df.copy()
    if context_mode is not None and "context_mode" in data.columns:
        data = data[data["context_mode"] == context_mode]

    item_col = pick_item_col(data)
    individual_col = pick_individual_col(data)

    mean_corr = subscale_table(data, score_col, pct_map)

    df_corr = mean_corr[["experiment", "pct_reversed_keyed_items", "mean_interitem_corr"]].dropna()
    n = len(df_corr)

    x = df_corr["pct_reversed_keyed_items"].to_numpy()
    y = df_corr["mean_interitem_corr"].to_numpy()

    if n >= 2 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = np.nan

    pearson_ci = corr_ci_fisher(pearson_r, n)
    spearman_ci = corr_ci_fisher(spearman_r, n)

    groups = df_corr["experiment"].to_numpy()
    pearson_ci_cluster, n_boot_degenerate = cluster_bootstrap_ci(x, y, groups)
    p_wild_cluster = wild_cluster_p(x, y, groups)

    if individual_col == "model" and small_models is not None and large_models is not None:
        pearson_r_small, n_small = _pearson_for_subset(
            data[data["model"].isin(small_models)],
            score_col, item_col, individual_col, pct_map,
        )
        pearson_r_large, n_large = _pearson_for_subset(
            data[data["model"].isin(large_models)],
            score_col, item_col, individual_col, pct_map,
        )
    else:
        pearson_r_small = np.nan
        pearson_r_large = np.nan
        n_small = 0
        n_large = 0

    return {
        "condition": label,
        "context": context_mode if context_mode is not None else "with_context",
        "score": score_col,
        "n_subscales": n,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "pearson_ci": pearson_ci,
        "n_instruments": len(np.unique(groups)),
        "pearson_ci_cluster": pearson_ci_cluster,
        "n_boot_degenerate": n_boot_degenerate,
        "p_wild_cluster": p_wild_cluster,
        "spearman_rho": spearman_r,
        "spearman_p": spearman_p,
        "spearman_ci": spearman_ci,
        "pearson_r_small": pearson_r_small,
        "pearson_r_large": pearson_r_large,
        "n_subscales_small": n_small,
        "n_subscales_large": n_large,
        **{f"{name}_{end}": float(ci[i]) for name, ci in [
            ("pearson_ci", pearson_ci), ("pearson_ci_cluster", pearson_ci_cluster),
            ("spearman_ci", spearman_ci)] for i, end in enumerate(("lo", "hi"))},
    }


# CSVs carry the CI bounds as separate _lo/_hi columns, not as tuples.
CI_TUPLES = ["pearson_ci", "pearson_ci_cluster", "spearman_ci"]


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

risk_data_LLM_direct_prompting = pd.read_csv(
    DATA_DIR / "LLM_data_proc_prompts_direct/LLM_no_flip_data_rereversed.csv",
    low_memory=False,
)

risk_data_flip_LLM_direct_prompting = pd.read_csv(
    DATA_DIR / "LLM_data_proc_prompts_direct/LLM_flip_data_rereversed.csv",
    low_memory=False,
)

risk_data_LLM_weighed_prompting = pd.read_csv(
    DATA_DIR / "LLM_data_proc_prompts_weighed/LLM_weighed_no_flip_data_rereversed.csv",
    low_memory=False,
)

risk_data_flip_LLM_weighed_prompting = pd.read_csv(
    DATA_DIR / "LLM_data_proc_prompts_weighed/LLM_weighed_flip_data_rereversed.csv",
    low_memory=False,
)

risk_data_humans = pd.read_csv(
    DATA_DIR / "human_data_proc/items_per_person.csv",
    low_memory=False,
)

risk_data_humans["experiment"] = risk_data_humans["experiment"].replace(
    {"DM scale": "Dm scale"}
)

# IPIP-NEO-300 (LLM + Humans)
ipip_llm = pd.read_csv(
    IPIP_DIR / "llm_data/ipipneo_all_data_reflipped_and_rereversed.csv",
    low_memory=False,
)
ipip_llm = ipip_llm.copy()
ipip_llm["category"] = ipip_llm["traits"]

ipip_hum = pd.read_csv(
    IPIP_DIR / "human_data/ipipneo_human.csv",
    low_memory=False,
)
ipip_hum["response"] = pd.to_numeric(
    ipip_hum["response"].replace({"0": None, 0: None, " ": None})
)
ipip_hum = ipip_hum.dropna(subset=["response"]).copy()
ipip_hum["experiment"] = "IPIP-NEO-300"
ipip_hum["category"] = ipip_hum["traits"]
ipip_hum = ipip_hum.rename(columns={"response": "score", "person_id": "partid"})

# API-run models, proprietary and large open-weight (risk + IPIP), no-reasoning (__nr) / no_context /
# no-flip — the only slice the API runs collected. These are added ONLY to the
# headline condition further below (direct / no flip / no context / textgen);
# no API data exists for the logit, with_context, flip or weighed conditions.
risk_api_direct = pd.read_csv(
    DATA_DIR / "api_data/LLM_api_no_flip_data_rereversed.csv", low_memory=False
)
risk_api_direct = risk_api_direct[
    risk_api_direct["model"].str.endswith("__nr")
    & (risk_api_direct["context_mode"] == "no_context")
    & (risk_api_direct["flipped"] == False)
].copy()

ipip_api = pd.read_csv(
    IPIP_DIR / "api_data/ipipneo_api_data_rereversed.csv", low_memory=False
)
ipip_api = ipip_api[
    ipip_api["model"].str.endswith("__nr")
    & (ipip_api["context_mode"] == "no_context")
    & (ipip_api["flipped"] == False)
].copy()
ipip_api["category"] = ipip_api["traits"]


# -----------------------------------------------------------------------------
# Make names equal
# -----------------------------------------------------------------------------

risk_data_LLM_direct_prompting = normalize_experiment_names(risk_data_LLM_direct_prompting)
risk_data_flip_LLM_direct_prompting = normalize_experiment_names(risk_data_flip_LLM_direct_prompting)
risk_data_LLM_weighed_prompting = normalize_experiment_names(risk_data_LLM_weighed_prompting)
risk_data_flip_LLM_weighed_prompting = normalize_experiment_names(risk_data_flip_LLM_weighed_prompting)
risk_data_humans = normalize_experiment_names(risk_data_humans)
ipip_llm = normalize_experiment_names(ipip_llm)
ipip_hum = normalize_experiment_names(ipip_hum)
risk_api_direct = normalize_experiment_names(risk_api_direct)
ipip_api = normalize_experiment_names(ipip_api)


# -----------------------------------------------------------------------------
# Compute pct reversed maps
# -----------------------------------------------------------------------------

pct_map_direct = compute_pct_reversed_map(
    pd.concat([risk_data_LLM_direct_prompting, ipip_llm], ignore_index=True),
    item_col="item_id",
)

pct_map_direct = _apply_domain_fixes(pct_map_direct, is_llm=True)

pct_map_flip_direct = compute_pct_reversed_map(
    pd.concat([risk_data_flip_LLM_direct_prompting, ipip_llm], ignore_index=True),
    item_col="item_id",
)
pct_map_flip_direct = _apply_domain_fixes(pct_map_flip_direct, is_llm=True)

pct_map_weighed = compute_pct_reversed_map(
    risk_data_LLM_weighed_prompting, item_col="item"
)
pct_map_weighed = _apply_domain_fixes(pct_map_weighed, is_llm=True)

pct_map_weighed_flip = compute_pct_reversed_map(
    risk_data_flip_LLM_weighed_prompting, item_col="item"
)


pct_map_weighed_flip = _apply_domain_fixes(pct_map_weighed_flip, is_llm=True)

humans_with_ipip = pd.concat([risk_data_humans, ipip_hum], ignore_index=True)
# Restrict humans to the instruments the directly-prompted LLMs were actually
# administered (the reference no-flip/no-context set), so the human correlation is
# computed over the same (sub-)scales as the LLM headline (as in Fig 4A); BART,
# CCT and DFE exist only in the weighted condition.
ref_llm_experiments = set(
    pd.concat([risk_data_LLM_direct_prompting, ipip_llm], ignore_index=True)["experiment"]
    .dropna()
    .unique()
)
humans_with_ipip = humans_with_ipip[
    humans_with_ipip["experiment"].isin(ref_llm_experiments)
].copy()
# Use LLM-based reversed-keying maps for humans (risk data lacks reverse_coded).
# pct_map_direct already includes IPIP traits via ipip_llm.
pct_map_humans = _apply_domain_fixes(pct_map_direct, is_llm=False).copy()
# LOT presentation order was randomised per trial for humans, so the LLM-instrument
# p_r (0.40 after folding) doesn't reflect what humans experienced (0.50 by design).
# Override the LOT entry on the humans-only map.
pct_map_humans.loc[
    pct_map_humans["experiment"] == "LOT", "pct_reversed_keyed_items"
] = 0.5

# -----------------------------------------------------------------------------
# Median split of LLMs by parameter count (B)
# -----------------------------------------------------------------------------

_llm_dfs = [
    risk_data_LLM_direct_prompting,
    risk_data_flip_LLM_direct_prompting,
    risk_data_LLM_weighed_prompting,
    risk_data_flip_LLM_weighed_prompting,
    ipip_llm,
]
_all_llm_models = set()
for _d in _llm_dfs:
    if "model" in _d.columns:
        _all_llm_models |= set(_d["model"].dropna().unique())

_missing_size = _all_llm_models - set(MODEL_SIZE)
if _missing_size:
    raise ValueError(f"Models missing from MODEL_SIZE: {sorted(_missing_size)}")

_sizes = [MODEL_SIZE[m] for m in _all_llm_models]
SIZE_MEDIAN_B = float(np.median(_sizes))
SMALL_MODELS = {m for m in _all_llm_models if MODEL_SIZE[m] <= SIZE_MEDIAN_B}
LARGE_MODELS = {m for m in _all_llm_models if MODEL_SIZE[m] > SIZE_MEDIAN_B}
print(f"Median model size: {SIZE_MEDIAN_B} B  |  small n={len(SMALL_MODELS)}  large n={len(LARGE_MODELS)}")

# -----------------------------------------------------------------------------
# Conditions
# -----------------------------------------------------------------------------

conditions = []

# Direct prompting (no flip / flip) with context and no context, model answer + logits
for ctx in ["no_context", "with_context"]:
    for score_col in ["model_answer", "logit_score"]:
        ipip_ctx = ipip_llm[ipip_llm["context_mode"] == ctx]
        direct_noflip_parts = [
            risk_data_LLM_direct_prompting,
            ipip_ctx[ipip_ctx["flipped"] == False],
        ]
        # Headline condition only: add the 14 API models (10 proprietary + 4 large
        # open-weight) to direct / no flip / no context / textgen — the single
        # condition the API runs cover. The size split uses locally run models only,
        # so API models enter the all-models estimate but not the S/L subsets.
        if ctx == "no_context" and score_col == "model_answer":
            direct_noflip_parts += [risk_api_direct, ipip_api]
        conditions.append(
            (
                "LLM direct (no flip)",
                pd.concat(direct_noflip_parts, ignore_index=True),
                ctx,
                score_col,
                pct_map_direct,
            )
        )
        conditions.append(
            (
                "LLM direct (flip)",
                pd.concat(
                    [
                        risk_data_flip_LLM_direct_prompting,
                        ipip_ctx[ipip_ctx["flipped"] == True],
                    ],
                    ignore_index=True,
                ),
                ctx,
                score_col,
                pct_map_flip_direct,
            )
        )

# Weighed prompting (single condition each)
# Use score column (score_expected is mostly NaN in current files)
conditions.append(
    (
        "LLM weighted (no flip)",
        risk_data_LLM_weighed_prompting,
        "with_context",
        "score",
        pct_map_weighed,
    )
)
conditions.append(
    (
        "LLM weighted (flip)",
        risk_data_flip_LLM_weighed_prompting,
        "with_context",
        "score",
        pct_map_weighed_flip,
    )
)

# Humans
conditions.append(("Humans", humans_with_ipip, "with_context", "score", pct_map_humans))

# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------

rows = []
for label, df, ctx, score_col, pct_map in conditions:
    if score_col not in df.columns:
        raise ValueError(f"Score column '{score_col}' not found for condition: {label}")
    rows.append(
        compute_condition_summary(
            df=df,
            score_col=score_col,
            pct_map=pct_map,
            context_mode=ctx,
            label=label,
            small_models=SMALL_MODELS,
            large_models=LARGE_MODELS,
        )
    )

summary_df = pd.DataFrame(rows)

# Robustness: the headline condition again without constant / low-variability
# models (rule in ../panel.py), written to a separate file that
# 04_profile_instability/04_low_variability_robustness_supp.py reads.
_headline_label, _headline_df, _headline_ctx, _headline_score, _headline_pct = conditions[0]
assert (_headline_label, _headline_ctx, _headline_score) == (
    "LLM direct (no flip)", "no_context", "model_answer")
_exclusion_sets, _ = low_variability_models()
_lowvar_rows = []
for _set_name, _excluded in _exclusion_sets.items():
    _keep = ~_headline_df["model"].map(bare_model_name).isin(_excluded)
    _row = compute_condition_summary(
        df=_headline_df[_keep], score_col=_headline_score, pct_map=_headline_pct,
        context_mode=_headline_ctx, label=f"LLM direct (no flip), excl. {_set_name}",
        small_models=SMALL_MODELS, large_models=LARGE_MODELS,
    )
    _row.update(excluded=_set_name, n_models=int(_headline_df.loc[_keep, "model"].nunique()))
    _lowvar_rows.append(_row)
pd.DataFrame(_lowvar_rows).drop(columns=CI_TUPLES).to_csv(
    OUTPUT_DIR / "orthogonality_low_variability.csv", index=False)

# Per-subscale points of the headline LLM condition and the humans, for the
# parent-instrument ICC (04_profile_instability/03_clustering_icc_supp.py).
_sub_tables = []
for label, df, ctx, score_col, pct_map in conditions:
    if (label, ctx, score_col) in {("LLM direct (no flip)", "no_context", "model_answer"),
                                   ("Humans", "with_context", "score")}:
        t = subscale_table(df, score_col, pct_map, context_mode=ctx)
        _sub_tables.append(t.assign(population="LLMs" if label != "Humans" else "Humans"))
pd.concat(_sub_tables, ignore_index=True).to_csv(
    OUTPUT_DIR / "orthogonality_subscales_reference.csv", index=False)

# Per-subscale points of every condition, for the behavioural-task robustness
# check (07_behavioural_tasks_supp.py), plus the humans on the behavioural tasks
# that only the weighted LLM condition administered (BART, CCT, DFE), with their
# p_r from the weighted map.
_humans_extra = normalize_experiment_names(
    pd.concat([risk_data_humans, ipip_hum], ignore_index=True))
_humans_extra = _humans_extra[_humans_extra["experiment"].isin(["BART", "CCT", "DFE"])]
_pct_humans_extra = _apply_domain_fixes(pct_map_weighed, is_llm=False)
pd.concat([
    subscale_table(df, score_col, pct_map, context_mode=ctx).assign(
        condition=label, context=ctx, score=score_col)
    for label, df, ctx, score_col, pct_map in conditions
] + [
    subscale_table(_humans_extra, "score", _pct_humans_extra, context_mode="with_context").assign(
        condition="Humans (BART, CCT, DFE)", context="with_context", score="score")
], ignore_index=True).to_csv(OUTPUT_DIR / "orthogonality_subscales_by_condition.csv", index=False)

summary_df["pearson (95% CI)"] = summary_df.apply(
    lambda r: format_corr_ci(r["pearson_r"], r["pearson_ci"]), axis=1
)
summary_df["spearman (95% CI)"] = summary_df.apply(
    lambda r: format_corr_ci(r["spearman_rho"], r["spearman_ci"]), axis=1
)
summary_df["pearson cluster CI"] = summary_df["pearson_ci_cluster"].map(
    lambda ci: f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"
)
summary_df["wild cluster p"] = summary_df["p_wild_cluster"].map(
    lambda p: "<.001" if p < .001 else f"{p:.3f}"
)

latex_df = summary_df[[
    "condition",
    "context",
    "score",
    "n_subscales",
    "n_instruments",
    "pearson (95% CI)",
    "pearson cluster CI",
    "wild cluster p",
    "spearman (95% CI)",
]].rename(
    columns={
        "condition": "Condition",
        "context": "Context",
        "score": "Score",
        "n_subscales": "N scales",
        "n_instruments": "N instr.",
        "pearson (95% CI)": "Pearson r (95% CI)",
        "pearson cluster CI": "Clustered 95% CI",
        "wild cluster p": "Wild p",
        "spearman (95% CI)": "Spearman rho (95% CI)",
    }
)

summary_df.drop(columns=CI_TUPLES).to_csv(OUTPUT_CSV, index=False)
# escape=True keeps the underscore-bearing data cells (no_context, model_answer,
# logit_score) safe; the math header is written as plain "rho" here and restored
# to $\rho$ afterwards so escaping does not mangle it.
latex = latex_df.to_latex(index=False, escape=True)
latex = latex.replace("Spearman rho", "Spearman $\\rho$")
latex = latex.replace("<.001", "$<$.001")  # "<" must be in math mode in LaTeX
latex = "\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n" + latex  # fit SI text width

with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
    f.write(latex)

print(f"Wrote: {OUTPUT_CSV}")
print(f"Wrote: {OUTPUT_TEX}")
