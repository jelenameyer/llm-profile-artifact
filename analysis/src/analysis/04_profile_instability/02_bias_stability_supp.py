"""SI: is response bias a stable per-responder property, or a
per-task quirk?

Runs the stability machinery over every (sub)scale that has both forward AND
reverse items, so a response bias is estimable:

  Big Five (IPIP-NEO-300):  O C E A N
  BARRATT (impulsivity):    BISa BISm BISn
  SSSV (sensation seeking): SSbor SSdis SSexp SStas
  DFD (decisions-from-description) and LOT (lotteries) behavioral tasks.

  -> 5 personality + 9 risk = 14 subscales.

(The other risk surveys — CARE, DOSPERT, SOEP, DAST, DM, GABS, MPL, PRI — have
zero reverse-coded items, so no response bias contrast exists for them and they
cannot enter this analysis.)

For each responder we take the SAME normalised per-subscale bias used by the
profile figure (reused directly from 08 so the two figures agree by
construction):

    b_norm[subscale] = ((R̄_f + R̄_r)/2 - midpoint) / half_scale_range

normalised so the 14 different Likert ranges are comparable. Treating the
subscales as "items" measuring one latent per-responder response bias we report,
per scope:

  - pairwise Pearson + Spearman correlations between subscales
  - Cronbach's alpha (subscales as items)
  - ICC(2,1) — two-way random, single rater, absolute agreement

Scopes (signed b̂ and |b̂| each):
  LLM_all     — all 14 subscales, every model present in both rosters    (the headline)
  LLM_novar{t}— same 14 subscales, but DROPPING any model that gave a
                constant answer (zero within-subscale variance of the raw
                response) on ≥ t subscales, for t in NOVAR_THRESHOLDS (1, 2, 3).
                Such a degenerate cell has R̄_f = R̄_r, so the model's profile
                sits on the y=x line and inflates the inter-subscale
                correlations; these scopes show the stability with the constant
                responders removed at three tolerances (t=1 = strict "var>0 on
                every subscale", t=2/3 = allow one/two constant subscales).
  LLM_big5    — Big-Five block of the full model set
  LLM_risk    — risk block of the full model set
  Human_big5  — 5 traits, IPIP human sample (20,993)        (reference; not plotted)
  Human_risk  — 9 risk subscales, risk human sample (1,507) (reference; not plotted)

NOTE on humans: the Big-Five sample and the risk sample are DISJOINT people, so
a single human cross-family 14x14 cannot be formed; humans are reported within
each family only (these feed the summary table and per-scope CSVs but not the
figures, which show four LLM panels).

Outputs:
  results/tables/bias_stability_summary.csv
  results/figures/bias_stability_heatmaps.pdf          (drop constant on ≥1)
  results/figures/bias_stability_heatmaps_excl2.pdf    (drop constant on ≥2)
  results/figures/bias_stability_heatmaps_excl3.pdf    (drop constant on ≥3)
"""
import importlib.util
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[4]

# ============================================================
# Reuse the profile figure's loaders + per-subscale bias definition
# ------------------------------------------------------------
# Import the sibling profile figure (01_profiles_per_keying.py) so this analysis
# runs on the EXACT same bias as that figure and the two cannot drift apart. Its
# module name starts with a digit, so it must be loaded via importlib, not a
# plain import.
# ============================================================

_M08_PATH = Path(__file__).resolve().parent / "01_profiles_per_keying.py"
_spec = importlib.util.spec_from_file_location("_m08_profiles", _M08_PATH)
m08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m08)

SUBSCALES = m08.SUBSCALES                      # (experiment, category, label, smin, smax) × 14
_subscale_slice = m08._subscale_slice

SUBSCALE_LABELS = [row[2] for row in SUBSCALES]
BIG5_LABELS = [row[2] for row in SUBSCALES if row[0] == "IPIP-NEO-300"]
RISK_LABELS = [row[2] for row in SUBSCALES if row[0] != "IPIP-NEO-300"]
N_BIG5 = len(BIG5_LABELS)

# Compact tick codes for the heatmaps (Big-Five letter; else subscale category,
# else the task name for the unsplit behavioral tasks).
SHORT = {}
for _exp, _cat, _label, _smin, _smax in SUBSCALES:
    if _exp == "IPIP-NEO-300":
        SHORT[_label] = _label[0]      # Openness -> O, ...
    elif _cat:
        SHORT[_label] = _cat           # BISa, SSbor, ...
    else:
        SHORT[_label] = _exp           # DFD, LOT

OUT_FIG_DIR = ROOT / "analysis/results/figures"
OUT_TABLE_DIR = ROOT / "analysis/results/tables"
for d in [OUT_FIG_DIR, OUT_TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FIGURE_BASENAME = "bias_stability_heatmaps"
# Bottom-row exclusion thresholds: one figure per t, dropping models that
# answered constantly on ≥ t subscales. t=1 (strict "var>0 everywhere") keeps
# the original filename; t≥2 get an "_excl{t}" suffix.
NOVAR_THRESHOLDS = (1, 2, 3)

# ============================================================
# Per-responder, per-subscale normalised bias  b_norm
# (identical formula on both sides; matches 08.compute_mu_a_risk)
# ============================================================


def llm_subscale_bias(llm_df: pd.DataFrame) -> pd.DataFrame:
    """Wide [model x subscale] table of normalised response bias.

    b_norm = ((R̄_f + R̄_r)/2 - midpoint) / half_range, with R̄_f / R̄_r the
    model's mean raw answer over forward / reverse items of the subscale.
    """
    rows = []
    for exp, cat, label, smin, smax in SUBSCALES:
        sub = _subscale_slice(llm_df, exp, cat)
        midpoint = (smin + smax) / 2
        half_range = (smax - smin) / 2
        g = sub.groupby(["model", "reverse_coded"])["model_answer"].mean().unstack()
        for model in g.index:
            Rf = g.loc[model].get(False, np.nan)
            Rr = g.loc[model].get(True, np.nan)
            rows.append({
                "model": model,
                "subscale": label,
                "b_norm": ((Rf + Rr) / 2 - midpoint) / half_range,
            })
    long = pd.DataFrame(rows)
    return long.pivot(index="model", columns="subscale", values="b_norm")


def human_subscale_bias(human_items: pd.DataFrame) -> pd.DataFrame:
    """Long [person_id, subscale, b_norm] per-person bias, same formula as the
    LLM side but per respondent. Kept long because the Big-Five and risk
    samples are disjoint people — they are pivoted to wide separately."""
    rows = []
    for exp, cat, label, smin, smax in SUBSCALES:
        sub = _subscale_slice(human_items, exp, cat)
        if sub.empty:
            continue
        midpoint = (smin + smax) / 2
        half_range = (smax - smin) / 2
        Rf = sub[~sub["reverse_coded"]].groupby("person_id")["response"].mean()
        Rr = sub[sub["reverse_coded"]].groupby("person_id")["response"].mean()
        idx = Rf.index.intersection(Rr.index)
        b = ((Rf.loc[idx] + Rr.loc[idx]) / 2 - midpoint) / half_range
        rows.append(pd.DataFrame({"person_id": idx, "subscale": label, "b_norm": b.values}))
    return pd.concat(rows, ignore_index=True)


def to_wide(long_df: pd.DataFrame, group_col: str, cols: list[str]) -> pd.DataFrame:
    """Pivot to [responder × subscale] in the canonical column order, keeping
    only responders with a value on every requested subscale."""
    wide = long_df.pivot(index=group_col, columns="subscale", values="b_norm")
    return wide.reindex(columns=cols).dropna()


def constant_scale_counts(llm_df: pd.DataFrame) -> pd.Series:
    """Per model, the NUMBER of subscales on which it gave a constant raw answer
    (zero within-subscale variance of model_answer, ≥2 items).

    For such a subscale R̄_f = R̄_r (the model answered the same value to every
    item, forward and reverse alike), so its normalised bias is pinned to the
    scale extreme there. These degenerate cells fall on the y=x line and inflate
    the inter-subscale correlations. Thresholding this count drives the LLM_novar
    scopes: a model is dropped from LLM_novar{t} when it is constant on ≥ t
    subscales (t=1 reproduces the strict "var>0 everywhere" filter). Models that
    were never constant do not appear in the returned Series."""
    counts: dict[str, int] = {}
    for exp, cat, _label, _smin, _smax in SUBSCALES:
        grp = _subscale_slice(llm_df, exp, cat).groupby("model")["model_answer"]
        constant = (grp.nunique() <= 1) & (grp.count() >= 2)
        for m in constant.index[constant]:
            counts[m] = counts.get(m, 0) + 1
    return pd.Series(counts, dtype=int)


# ============================================================
# Stability statistics
# ============================================================


def cronbach_alpha(X: np.ndarray) -> float:
    """Standardised Cronbach's alpha across columns ('items')."""
    X = np.asarray(X, float)
    k = X.shape[1]
    item_vars = X.var(axis=0, ddof=1).sum()
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return np.nan
    return (k / (k - 1)) * (1.0 - item_vars / total_var)


def icc_2_1(X: np.ndarray) -> float:
    """ICC(2,1) — two-way random, single rater, absolute agreement.

    Rows = subjects (responders), columns = raters (subscales).
    """
    X = np.asarray(X, float)
    n, k = X.shape
    grand = X.mean()
    row_m = X.mean(axis=1)
    col_m = X.mean(axis=0)
    SS_total = ((X - grand) ** 2).sum()
    SS_row = k * ((row_m - grand) ** 2).sum()
    SS_col = n * ((col_m - grand) ** 2).sum()
    SS_err = SS_total - SS_row - SS_col
    MS_row = SS_row / (n - 1)
    MS_col = SS_col / (k - 1)
    MS_err = SS_err / ((n - 1) * (k - 1))
    denom = MS_row + (k - 1) * MS_err + k * (MS_col - MS_err) / n
    if denom <= 0:
        return np.nan
    return (MS_row - MS_err) / denom


def mean_offdiag(C: pd.DataFrame) -> float:
    A = C.values.copy()
    np.fill_diagonal(A, np.nan)
    return float(np.nanmean(A))


# ============================================================
# Compute
# ============================================================


def main() -> None:
    print("Loading combined LLM data (risk + IPIP-NEO-300) via figure-08 loader...")
    llm_df = m08.load_combined_reference()
    llm_wide_full = llm_subscale_bias(llm_df)
    # Headline LLM matrix: the models with a bias on all 14 subscales. The
    # within-family sub-blocks reuse this SAME model set so any difference is
    # the subscales, not the sample.
    llm_all = llm_wide_full.reindex(columns=SUBSCALE_LABELS).dropna()
    print(f"  LLMs with all 14 subscales: {len(llm_all)}")

    # How many subscales each model answered constantly on (zero within-subscale
    # variance). The bottom row of each figure drops models constant on ≥ t of
    # them, for the thresholds in NOVAR_THRESHOLDS.
    const_counts = constant_scale_counts(llm_df)
    print(f"  models constant on ≥1 subscale: {(const_counts >= 1).sum()}")

    print("Loading human per-person item data via figure-08 loader...")
    human_long = human_subscale_bias(m08.load_human_items())
    human_big5 = to_wide(human_long, "person_id", BIG5_LABELS)
    human_risk = to_wide(human_long, "person_id", RISK_LABELS)
    print(f"  Humans with all 5 Big-Five traits: {len(human_big5)}")
    print(f"  Humans with all 9 risk subscales : {len(human_risk)}")

    # scope name -> signed wide matrix; |b̂| versions derived below.
    base = {
        "LLM_all":    llm_all,
        "LLM_big5":   llm_all[BIG5_LABELS],
        "LLM_risk":   llm_all[RISK_LABELS],
        "Human_big5": human_big5,
        "Human_risk": human_risk,
    }
    # One var-filtered scope family per threshold (LLM_novar{t}). t=1 is the
    # strict "var>0 on every subscale" filter; higher t tolerates a few constant
    # subscales before dropping the model.
    for t in NOVAR_THRESHOLDS:
        dropped = [m for m in const_counts.index[const_counts >= t] if m in llm_all.index]
        novar = llm_all.drop(index=dropped)
        base[f"LLM_novar{t}"]      = novar
        base[f"LLM_novar{t}_big5"] = novar[BIG5_LABELS]
        base[f"LLM_novar{t}_risk"] = novar[RISK_LABELS]
        print(f"  novar (drop constant on ≥{t}): dropped {len(dropped)}, kept {len(novar)}")

    summary = []
    corr_pearson, corr_spearman = {}, {}
    for scope, X_signed in base.items():
        for kind, X in (("signed", X_signed), ("abs", X_signed.abs())):
            name = f"{scope}_{kind}"
            Cp = X.corr(method="pearson")
            Cs = X.corr(method="spearman")
            corr_pearson[name] = Cp
            corr_spearman[name] = Cs
            summary.append({
                "condition":       name,
                "scope":           scope,
                "kind":            kind,
                "n":               len(X),
                "n_subscales":     X.shape[1],
                "mean_r_pearson":  mean_offdiag(Cp),
                "mean_r_spearman": mean_offdiag(Cs),
                "cronbach_alpha":  cronbach_alpha(X.values),
                "icc_2_1":         icc_2_1(X.values),
            })

    summary_df = pd.DataFrame(summary)
    print("\n=== Response bias stability summary ===")
    print(summary_df.drop(columns=["scope", "kind"]).round(3).to_string(index=False))

    summary_df.to_csv(OUT_TABLE_DIR / "bias_stability_summary.csv", index=False)

    # ============================================================
    # Figures — one per NOVAR_THRESHOLDS entry, each a 2×2 of LLM panels
    # (14×14, block separator at the personality↔risk divide):
    #   Top row    : ALL models                       — signed b̂ and |b̂|
    #   Bottom row : models constant on < t subscales — signed b̂ and |b̂|
    # The top row is identical across the three figures; only the bottom-row
    # exclusion threshold changes. The human within-family panels are still
    # computed (they feed the summary table + per-scope CSVs) but not drawn.
    # ============================================================
    for t in NOVAR_THRESHOLDS:
        build_figure(corr_pearson, summary_df, t)


def draw(ax, C: pd.DataFrame, title: str, block_split: int | None = None) -> "plt.cm.ScalarMappable":
    """One 14x14 (or smaller) correlation heatmap with cell annotations and an
    optional black line splitting the personality block from the risk block."""
    labels = [SHORT[c] for c in C.columns]
    A = C.values
    im = ax.imshow(A, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    fs = 7 if len(labels) > 9 else 9
    ax.set_xticklabels(labels, fontsize=fs, rotation=90)
    ax.set_yticklabels(labels, fontsize=fs)
    cell_fs = 5 if len(labels) > 9 else 8
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                    fontsize=cell_fs,
                    color="black" if abs(A[i, j]) < 0.6 else "white")
    if block_split is not None:
        ax.axhline(block_split - 0.5, color="black", lw=1.2)
        ax.axvline(block_split - 0.5, color="black", lw=1.2)
    ax.set_title(title, fontsize=9)
    return im


def build_figure(corr_pearson: dict, summary_df: pd.DataFrame, threshold: int) -> None:
    """Render + save the 2x2 LLM heatmap figure whose bottom row drops models
    that answered constantly on ≥ `threshold` subscales.

    t=1 keeps the original filename (bias_stability_heatmaps.pdf); t≥2
    get an "_excl{t}" suffix."""
    def s(name, field):
        return summary_df.loc[summary_df["condition"] == name, field].iloc[0]

    nv = f"LLM_novar{threshold}"
    plural = "s" if threshold > 1 else ""
    n_all = int(s("LLM_all_signed", "n"))
    n_dropped = n_all - int(s(f"{nv}_signed", "n"))

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))

    im = draw(
        axes[0, 0], corr_pearson["LLM_all_signed"],
        rf"LLMs — Pearson $r(\hat b)$, all 14 subscales (n={n_all})"
        "\n"
        rf"all: $\bar r$={s('LLM_all_signed','mean_r_pearson'):.2f}, "
        rf"$\alpha$={s('LLM_all_signed','cronbach_alpha'):.2f}, "
        rf"ICC={s('LLM_all_signed','icc_2_1'):.2f}   |   "
        rf"Big5 $\alpha$={s('LLM_big5_signed','cronbach_alpha'):.2f}, "
        rf"risk $\alpha$={s('LLM_risk_signed','cronbach_alpha'):.2f}",
        block_split=N_BIG5,
    )
    draw(
        axes[0, 1], corr_pearson["LLM_all_abs"],
        rf"LLMs — Pearson $r(|\hat b|)$, all 14 subscales (n={n_all})"
        "\n"
        rf"all: $\bar r$={s('LLM_all_abs','mean_r_pearson'):.2f}, "
        rf"$\alpha$={s('LLM_all_abs','cronbach_alpha'):.2f}, "
        rf"ICC={s('LLM_all_abs','icc_2_1'):.2f}   |   "
        rf"Big5 $\alpha$={s('LLM_big5_abs','cronbach_alpha'):.2f}, "
        rf"risk $\alpha$={s('LLM_risk_abs','cronbach_alpha'):.2f}",
        block_split=N_BIG5,
    )
    draw(
        axes[1, 0], corr_pearson[f"{nv}_signed"],
        rf"LLMs (drop constant on $\geq${threshold} subscale{plural}) — "
        rf"Pearson $r(\hat b)$ (n={int(s(f'{nv}_signed','n'))})"
        "\n"
        rf"all: $\bar r$={s(f'{nv}_signed','mean_r_pearson'):.2f}, "
        rf"$\alpha$={s(f'{nv}_signed','cronbach_alpha'):.2f}, "
        rf"ICC={s(f'{nv}_signed','icc_2_1'):.2f}   |   "
        rf"Big5 $\alpha$={s(f'{nv}_big5_signed','cronbach_alpha'):.2f}, "
        rf"risk $\alpha$={s(f'{nv}_risk_signed','cronbach_alpha'):.2f}",
        block_split=N_BIG5,
    )
    draw(
        axes[1, 1], corr_pearson[f"{nv}_abs"],
        rf"LLMs (drop constant on $\geq${threshold} subscale{plural}) — "
        rf"Pearson $r(|\hat b|)$ (n={int(s(f'{nv}_abs','n'))})"
        "\n"
        rf"all: $\bar r$={s(f'{nv}_abs','mean_r_pearson'):.2f}, "
        rf"$\alpha$={s(f'{nv}_abs','cronbach_alpha'):.2f}, "
        rf"ICC={s(f'{nv}_abs','icc_2_1'):.2f}   |   "
        rf"Big5 $\alpha$={s(f'{nv}_big5_abs','cronbach_alpha'):.2f}, "
        rf"risk $\alpha$={s(f'{nv}_risk_abs','cronbach_alpha'):.2f}",
        block_split=N_BIG5,
    )

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7,
                 label="Pearson r between subscales")
    fig.suptitle(
        "Is response bias a stable per-model property across instruments?\n"
        rf"Inter-subscale correlations of $\hat b$ — top: all {n_all} LLMs, "
        rf"bottom: drop the {n_dropped} models constant on $\geq${threshold} subscale{plural}"
        "\n(personality vs. risk; black lines split the two families)",
        fontsize=13, fontweight="bold", y=0.995,
    )

    out_name = f"{FIGURE_BASENAME}{'' if threshold == 1 else f'_excl{threshold}'}.pdf"
    fig.savefig(OUT_FIG_DIR / out_name, bbox_inches="tight", dpi=200)
    print(f"Saved: {OUT_FIG_DIR / out_name}")
    plt.close(fig)


if __name__ == "__main__":
    main()
