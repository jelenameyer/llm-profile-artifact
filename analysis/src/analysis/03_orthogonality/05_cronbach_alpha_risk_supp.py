"""SI Table tab:alpha_table_risk: Cronbach alpha + mean inter-item correlation per
risk instrument, LLMs (56, incl. 10 proprietary API) vs humans, no-context textgen.

Reads risk_data (LLM/api no-flip rereversed, human items_per_person).
Writes results/tables/cronbach_alpha_risk.tex.
"""
import pandas as pd
from pathlib import Path

from utils import (
    compute_cronbach_alpha,
    compute_mean_interitem_corr_with_ci,
    apply_domain_fixes,
    compute_pct_reversed_map,
    add_pct_reversed,
    format_cronbach_latex,
)


def build_overview_table(llm_data, human_data):
    domain_exclude = {"SOEP scale", "SOEP"}
    llm_experiments = set(llm_data["experiment"].dropna().unique())

    alpha_llm = compute_cronbach_alpha(llm_data, score="model_answer", item_name="item_id",
                                       individual="model", domain_exclude=domain_exclude)
    alpha_human = compute_cronbach_alpha(human_data, score="score", item_name="item",
                                         individual="partid", domain_exclude=domain_exclude)
    alpha_llm[["alpha_ci_low_llm", "alpha_ci_high_llm"]] = pd.DataFrame(
        alpha_llm["alpha_CI"].tolist(), index=alpha_llm.index)
    alpha_human[["alpha_ci_low_humans", "alpha_ci_high_humans"]] = pd.DataFrame(
        alpha_human["alpha_CI"].tolist(), index=alpha_human.index)

    mean_llm = compute_mean_interitem_corr_with_ci(llm_data, score="model_answer",
                                                   item_name="item_id", individual="model",
                                                   domain_exclude=domain_exclude)
    mean_human = compute_mean_interitem_corr_with_ci(human_data, score="score",
                                                     item_name="item", individual="partid",
                                                     domain_exclude=domain_exclude)

    alpha_llm = apply_domain_fixes(alpha_llm, is_llm=True)
    mean_llm = apply_domain_fixes(mean_llm, is_llm=True)
    alpha_human = apply_domain_fixes(alpha_human, is_llm=False)
    mean_human = apply_domain_fixes(mean_human, is_llm=False)

    alpha_llm = alpha_llm[alpha_llm["experiment"].isin(llm_experiments)]
    alpha_human = alpha_human[alpha_human["experiment"].isin(llm_experiments)]
    mean_llm = mean_llm[mean_llm["experiment"].isin(llm_experiments)]
    mean_human = mean_human[mean_human["experiment"].isin(llm_experiments)]

    table = (
        alpha_llm.rename(columns={"alpha": "cronbach_alpha_llm", "k": "k_llm"})
        .merge(alpha_human.rename(columns={"alpha": "cronbach_alpha_humans", "k": "k_humans"}),
               on=["experiment", "domain"], how="outer")
        .merge(mean_llm.rename(columns={
            "mean_interitem_corr": "avg_interitem_corr_llm", "k": "k_llm_mean",
            "mean_interitem_corr_ci_low": "avg_interitem_corr_llm_ci_low",
            "mean_interitem_corr_ci_high": "avg_interitem_corr_llm_ci_high"}),
               on=["experiment", "domain"], how="outer")
        .merge(mean_human.rename(columns={
            "mean_interitem_corr": "avg_interitem_corr_humans", "k": "k_humans_mean",
            "mean_interitem_corr_ci_low": "avg_interitem_corr_humans_ci_low",
            "mean_interitem_corr_ci_high": "avg_interitem_corr_humans_ci_high"}),
               on=["experiment", "domain"], how="outer")
    )
    table["k"] = (table["k_llm"].combine_first(table["k_humans"])
                  .combine_first(table["k_llm_mean"]).combine_first(table["k_humans_mean"]))

    table = add_pct_reversed(table, compute_pct_reversed_map(llm_data))
    table["domain"] = table["domain"].replace("total", None)

    return table[[
        "experiment", "domain", "k", "pct_reversed_keyed_items",
        "avg_interitem_corr_llm", "avg_interitem_corr_llm_ci_low", "avg_interitem_corr_llm_ci_high",
        "cronbach_alpha_llm", "alpha_ci_low_llm", "alpha_ci_high_llm",
        "avg_interitem_corr_humans", "avg_interitem_corr_humans_ci_low", "avg_interitem_corr_humans_ci_high",
        "cronbach_alpha_humans", "alpha_ci_low_humans", "alpha_ci_high_humans",
    ]].copy()


def main():
    llm = pd.read_csv(
        "../../../data/risk_data/LLM_data_proc_prompts_direct/LLM_no_flip_data_rereversed.csv",
        low_memory=False)
    llm = llm[llm["context_mode"] == "no_context"]
    # Add the 10 proprietary API models (no-reasoning / no_context / no-flip slice).
    api = pd.read_csv(
        "../../../data/risk_data/api_data/LLM_api_no_flip_data_rereversed.csv",
        low_memory=False)
    api = api[api["model"].str.endswith("__nr")
              & (api["context_mode"] == "no_context") & (api["flipped"] == False)]
    llm = pd.concat([llm, api], ignore_index=True)
    print(f"LLM risk roster: {llm['model'].nunique()} models "
          f"({api['model'].nunique()} proprietary API added)")

    human = pd.read_csv(
        "../../../data/risk_data/human_data_proc/items_per_person.csv",
        low_memory=False)
    human["experiment"] = (human["experiment"].str.replace(r"\s.+$", "", regex=True)
                                              .str.replace("BARRAT", "BARRATT", regex=False))

    df = build_overview_table(llm, human)
    df = df.sort_values("cronbach_alpha_llm", ascending=False, na_position="last").reset_index(drop=True)

    out_dir = Path("../../../results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cronbach_alpha_risk.tex"
    out_path.write_text(format_cronbach_latex(df).to_latex(index=False, escape=False))
    print(f"Saved LaTeX table to: {out_path}")


if __name__ == "__main__":
    main()
