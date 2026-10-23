"""SI: principal analyses rerun without constant and low-variability models.

Four nested samples (rule in ../panel.py): all models; without constant models
(SD = 0); without models whose within-model SD falls below the 1st percentile of the
human within-person SD; and below the 5th percentile (sensitivity check). Per
sample, reference condition (no context, no flip, non-reasoning for API models):

  - Table 1: forward-reverse correlation rho per Big Five trait, Fisher-z CI, and
    pi_b = (1 + rho) / 2 with the transformed CI;
  - Fig 3 bottom row: item-noise corrected |b_hat| (correction re-estimated within
    each sample) vs log10(total parameters), open-weight models, and open-weight vs
    proprietary (medians, Mann-Whitney p);
  - Fig 4: cross-instrument correlation of p_r with mean inter-item correlation,
    with the parent-instrument cluster-bootstrap CI and wild-cluster p (computed in
    03_orthogonality/02_orthogonality_corr_by_condition.py).

Reads ipipneo300_data (llm/api, raw) and results/tables/orthogonality_*.csv.
Writes results/tables/low_variability_robustness.{csv,tex} and
low_variability_models.csv (every model's SD, human percentile, set membership).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "02_capability_and_bias"))
from panel import MODEL_SIZE, bare_model_name, low_variability_models  # noqa: E402
from utils import attenuation_correct, per_respondent_contrasts  # noqa: E402

DATA_DIR = HERE.parents[2] / "data/ipipneo300_data"
TABLE_DIR = HERE.parents[2] / "results/tables"
TRAITS = ["O", "C", "E", "A", "N"]


def load_ipip_reference() -> pd.DataFrame:
    """Same frame as Table 1 / Fig 3: local models no-context + API non-reasoning."""
    llm = pd.read_csv(DATA_DIR / "llm_data/ipipneo_no_flip_data_raw.csv", low_memory=False)
    api = pd.read_csv(DATA_DIR / "api_data/ipipneo_api_data_raw.csv", low_memory=False)
    df = pd.concat([llm, api[api["model"].str.endswith("__nr")]], ignore_index=True)
    df = df[df["context_mode"] == "no_context"].copy()
    df["model_answer"] = pd.to_numeric(df["model_answer"], errors="coerce")
    return df


def fisher_ci(r, n, alpha=0.05):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    zc = stats.norm.ppf(1 - alpha / 2)
    return np.tanh(z - zc * se), np.tanh(z + zc * se)


def analyse(df: pd.DataFrame) -> list[dict]:
    contrasts = per_respondent_contrasts(df, "model", "model_answer", "item_id")
    attenuation_correct(contrasts)
    b_corr = contrasts.pivot(index="model", columns="trait", values="b_corr")
    rows = []
    for t in TRAITS:
        g = (df[df["traits"] == t]
             .groupby(["model", "reverse_coded"])["model_answer"].mean().unstack())
        Rf, Rr = g[False], g[True]
        n = len(g)
        rho = np.corrcoef(Rf, Rr)[0, 1]
        lo, hi = fisher_ci(rho, n)

        abs_b = b_corr[t].loc[g.index].abs()
        is_open = abs_b.index.isin(list(MODEL_SIZE))
        sizes = np.log10(abs_b.index[is_open].map(MODEL_SIZE).astype(float))
        reg = stats.linregress(sizes, abs_b[is_open])
        mw_p = stats.mannwhitneyu(abs_b[is_open], abs_b[~is_open]).pvalue

        rows.append({
            "trait": t, "n_models": n,
            "rho": rho, "rho_lo": lo, "rho_hi": hi,
            "pi_b": (1 + rho) / 2, "pi_b_lo": (1 + lo) / 2, "pi_b_hi": (1 + hi) / 2,
            "n_open": int(is_open.sum()), "n_proprietary": int((~is_open).sum()),
            "size_r": reg.rvalue, "size_p": reg.pvalue, "size_slope": reg.slope,
            "median_abs_b_open": abs_b[is_open].median(),
            "median_abs_b_proprietary": abs_b[~is_open].median(),
            "mw_p_open_vs_prop": mw_p,
        })
    return rows


def main() -> None:
    sets, table = low_variability_models()
    table.to_csv(TABLE_DIR / "low_variability_models.csv", index_label="model")
    print(f"Human within-person SD (N = {table.attrs['n_humans']:,}): "
          f"p1 = {table.attrs['human_p1']:.3f}, p5 = {table.attrs['human_p5']:.3f}")

    samples = {"all": set(), **sets}
    df = load_ipip_reference()
    rows = []
    for name, excluded in samples.items():
        keep = ~df["model"].map(bare_model_name).isin(excluded)
        for r in analyse(df[keep]):
            rows.append({"sample": name, "n_excluded": len(excluded), **r})
    res = pd.DataFrame(rows)

    # Cross-instrument correlation: full sample from the main table, exclusions from
    # the robustness rows of the same script.
    orth_all = pd.read_csv(TABLE_DIR / "orthogonality_corr_by_condition.csv").iloc[[0]]
    orth_all = orth_all.assign(excluded="all")
    orth_ex = pd.read_csv(TABLE_DIR / "orthogonality_low_variability.csv")
    orth = pd.concat([orth_all, orth_ex], ignore_index=True).set_index("excluded")
    for col in ["pearson_r", "pearson_ci_cluster_lo", "pearson_ci_cluster_hi",
                "p_wild_cluster", "n_subscales"]:
        res[f"orth_{col}"] = res["sample"].map(orth[col])
    res.to_csv(TABLE_DIR / "low_variability_robustness.csv", index=False)

    # Compact LaTeX: one row per sample, trait ranges.
    def rng(s, d=2):
        return f"{s.min():.{d}f} to {s.max():.{d}f}"
    labels = {"all": "All models", "constant": "Excl. constant (SD = 0)",
              "below_p1": "Excl. SD $<$ human P1", "below_p5": "Excl. SD $<$ human P5"}
    compact = []
    for name, g in res.groupby("sample", sort=False):
        o = g.iloc[0]
        lo, hi = o["orth_pearson_ci_cluster_lo"], o["orth_pearson_ci_cluster_hi"]
        compact.append({
            "Sample": labels[name], "N": int(o["n_models"]),
            r"$\pi_b$ (range over traits)": rng(g["pi_b"]),
            r"95\% CI bounds": f"{g['pi_b_lo'].min():.2f}--{g['pi_b_hi'].max():.2f}",
            r"size $r$": rng(g["size_r"]),
            r"size $p$": rng(g["size_p"], 3),
            r"open vs prop. $p$": rng(g["mw_p_open_vs_prop"], 3),
            r"cross-instr. $r$ [clustered CI]": f"{o['orth_pearson_r']:.2f} [{lo:.2f}, {hi:.2f}]",
        })
    with open(TABLE_DIR / "low_variability_robustness.tex", "w", encoding="utf-8") as f:
        f.write(pd.DataFrame(compact).to_latex(
            index=False, escape=False, column_format="lrcccccc",
            caption="Principal analyses without constant and low-variability models. "
                    "Ranges are over the five Big Five traits.",
            label="tab:low_variability_robustness"))

    pd.set_option("display.width", 250)
    print(pd.DataFrame(compact).to_string(index=False))
    for name in ["below_p1", "below_p5"]:
        newly = sorted(sets[name] - (sets["constant"] if name == "below_p1" else sets["below_p1"]))
        print(f"\n{name}: {len(sets[name])} excluded; added at this step: {', '.join(newly)}")


if __name__ == "__main__":
    main()
