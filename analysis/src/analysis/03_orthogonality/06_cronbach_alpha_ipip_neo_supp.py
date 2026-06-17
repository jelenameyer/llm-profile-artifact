"""SI Tables tab:ipip_neo_trait_reversed / tab:ipip_neo_trait_raw: trait-level
IPIP-NEO-300 Cronbach alpha + mean inter-item correlation, LLMs (56, incl. 10
proprietary API) vs humans, no-context textgen. Mirrors the risk table
(05_cronbach_alpha_risk_supp.py).

Reads ipipneo300_data (llm/api/human, raw + rereversed).
Writes results/tables/:
  - recoded (rereversed):  cronbach_alpha_ipip_neo_recoded.tex
  - raw:                   cronbach_alpha_ipip_neo_raw.tex
"""
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    compute_cronbach_alpha,
    compute_mean_interitem_corr_with_ci,
    format_cronbach_latex,
)

UNIT = "traits"  # group at the trait level (O/C/E/A/N), not the facet level

HERE = Path(__file__).resolve().parent
DATA = HERE / "../../../data/ipipneo300_data"
OUT = HERE / "../../../results/tables"


def compute_pct_reversed_map(llm_data):
    if "reverse_coded" not in llm_data.columns:
        raise ValueError("reverse_coded column missing in llm data")
    tmp = llm_data[["experiment", UNIT, "item_id", "reverse_coded"]].drop_duplicates()
    tmp = tmp.rename(columns={UNIT: "domain"})
    tmp["domain"] = tmp["domain"].fillna("total")
    # True proportion of reverse-keyed items. Report it directly (do NOT fold
    # values > 0.5 to 1 - p_r): the column is labelled "pct rev. items" and must
    # match the keying table (tab:pct_rev_ipipneo300), where O = 0.53 and A = 0.60.
    pct = (
        tmp.groupby(["experiment", "domain"])["reverse_coded"]
        .mean()
        .reset_index(name="pct_reversed_keyed_items")
    )
    return pct


def add_pct_reversed(df, pct_map):
    out = df.copy()
    pct = pct_map.copy()

    map_domain = dict(zip(zip(pct["experiment"], pct["domain"]), pct["pct_reversed_keyed_items"]))
    map_experiment = dict(zip(pct["experiment"], pct["pct_reversed_keyed_items"]))

    out["pct_reversed_keyed_items"] = list(zip(out["experiment"], out["domain"]))
    out["pct_reversed_keyed_items"] = out["pct_reversed_keyed_items"].map(map_domain)
    out["pct_reversed_keyed_items"] = out["pct_reversed_keyed_items"].combine_first(
        out["experiment"].map(map_experiment)
    ).fillna(0)
    return out


def build_overview_table(llm_data, human_data, experiment_name):
    llm_data = llm_data.copy()
    human_data = human_data.copy()

    llm_data["experiment"] = experiment_name
    human_data["experiment"] = experiment_name

    alpha_llm = compute_cronbach_alpha(
        llm_data,
        score="model_answer",
        item_name="item_id",
        individual="model",
        domain_col=UNIT,
    )
    alpha_human = compute_cronbach_alpha(
        human_data,
        score="response",
        item_name="item",
        individual="person_id",
        domain_col=UNIT,
    )

    alpha_llm[["alpha_ci_low_llm", "alpha_ci_high_llm"]] = pd.DataFrame(
        alpha_llm["alpha_CI"].tolist(), index=alpha_llm.index
    )
    alpha_human[["alpha_ci_low_humans", "alpha_ci_high_humans"]] = pd.DataFrame(
        alpha_human["alpha_CI"].tolist(), index=alpha_human.index
    )

    mean_llm = compute_mean_interitem_corr_with_ci(
        llm_data,
        score="model_answer",
        item_name="item_id",
        individual="model",
        domain_col=UNIT,
    )
    mean_human = compute_mean_interitem_corr_with_ci(
        human_data,
        score="response",
        item_name="item",
        individual="person_id",
        domain_col=UNIT,
    )

    table = (
        alpha_llm.rename(columns={"alpha": "cronbach_alpha_llm", "k": "k_llm"})
        .merge(
            alpha_human.rename(columns={"alpha": "cronbach_alpha_humans", "k": "k_humans"}),
            on=["experiment", "domain"],
            how="outer",
        )
        .merge(
            mean_llm.rename(columns={
                "mean_interitem_corr": "avg_interitem_corr_llm",
                "k": "k_llm_mean",
                "mean_interitem_corr_ci_low": "avg_interitem_corr_llm_ci_low",
                "mean_interitem_corr_ci_high": "avg_interitem_corr_llm_ci_high",
            }),
            on=["experiment", "domain"],
            how="outer",
        )
        .merge(
            mean_human.rename(columns={
                "mean_interitem_corr": "avg_interitem_corr_humans",
                "k": "k_humans_mean",
                "mean_interitem_corr_ci_low": "avg_interitem_corr_humans_ci_low",
                "mean_interitem_corr_ci_high": "avg_interitem_corr_humans_ci_high",
            }),
            on=["experiment", "domain"],
            how="outer",
        )
    )

    table["k"] = (
        table["k_llm"]
        .combine_first(table["k_humans"])
        .combine_first(table["k_llm_mean"])
        .combine_first(table["k_humans_mean"])
    )

    pct_map = compute_pct_reversed_map(llm_data)
    table = add_pct_reversed(table, pct_map)

    table["domain"] = table["domain"].replace("total", None)

    out = table[[
        "experiment",
        "domain",
        "k",
        "pct_reversed_keyed_items",
        "avg_interitem_corr_llm",
        "avg_interitem_corr_llm_ci_low",
        "avg_interitem_corr_llm_ci_high",
        "cronbach_alpha_llm",
        "alpha_ci_low_llm",
        "alpha_ci_high_llm",
        "avg_interitem_corr_humans",
        "avg_interitem_corr_humans_ci_low",
        "avg_interitem_corr_humans_ci_high",
        "cronbach_alpha_humans",
        "alpha_ci_low_humans",
        "alpha_ci_high_humans",
    ]].copy()

    return out


# ── pooled LLM panel (OSS + 10 proprietary API) ──────────────────────────────

def load_pooled_llm(oss_path, api_path):
    """OSS no-context + 10 non-reasoning proprietary API models (no-context)."""
    oss = pd.read_csv(oss_path, low_memory=False)
    oss = oss[oss["context_mode"] == "no_context"]
    api = pd.read_csv(api_path, low_memory=False)
    api = api[
        api["model"].str.contains("_nr", regex=False)
        & (api["context_mode"] == "no_context")
    ]
    llm = pd.concat([oss, api], ignore_index=True)
    llm["model_answer"] = pd.to_numeric(llm["model_answer"], errors="coerce")
    n = llm["model"].nunique()
    print(f"  pooled LLM panel: {n} models "
          f"({oss['model'].nunique()} OSS + {api['model'].nunique()} API)")
    return llm


def build(oss_csv, api_csv, human_csv, out_name):
    llm = load_pooled_llm(DATA / oss_csv, DATA / api_csv)
    human = pd.read_csv(DATA / human_csv, low_memory=False)
    if "response" in human.columns:
        human["response"] = pd.to_numeric(human["response"], errors="coerce")

    df = build_overview_table(llm, human, experiment_name="IPIP-NEO-300")
    df = df.sort_values(by=["cronbach_alpha_llm"], ascending=False,
                        na_position="last").reset_index(drop=True)
    latex = format_cronbach_latex(df, fmt_k=True).to_latex(index=False, escape=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / out_name).write_text(latex)
    print(f"  wrote {out_name}")


if __name__ == "__main__":
    print("recoded (rereversed):")
    build(
        "llm_data/ipipneo_no_flip_data_rereversed.csv",
        "api_data/ipipneo_api_data_rereversed.csv",
        "human_data/ipipneo_human.csv",
        "cronbach_alpha_ipip_neo_recoded.tex",
    )
    print("raw:")
    build(
        "llm_data/ipipneo_no_flip_data_raw.csv",
        "api_data/ipipneo_api_data_raw.csv",
        "human_data/ipipneo_human_raw.csv",
        "cronbach_alpha_ipip_neo_raw.tex",
    )
