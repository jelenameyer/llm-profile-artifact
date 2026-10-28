"""SI: is response bias a stable per-responder property, or a
per-task quirk? And how does the trait component compare?

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

normalised so the 14 different Likert ranges are comparable. The trait
component gets the same treatment, on the same -1..+1 range:

    theta_norm[subscale] = ((R̄_f - R̄_r)/2) / half_scale_range

i.e. the keyed trait score centred at the scale midpoint (a constant responder
has theta_norm = 0 there). For LOT and DFD (position-coded: 1 = first option,
reverse = risky option second) theta_norm = 2 * share of risky choices - 1.
Only the signed theta is analysed; |theta| is the extremity of a trait score,
not a trait.

Treating the subscales as "items" measuring one latent per-responder response
bias we report, per scope:

  - pairwise Pearson + Spearman correlations between subscales
  - mean absolute Pearson correlation
  - Cronbach's alpha (subscales as items)
  - ICC(2,1) — two-way random, single rater, absolute agreement

The trait components are distinct constructs, so these one-latent summaries do
not apply to them: the table and the figure subtitles report the bias only, and
the figure shows the trait correlation matrices. The trait summaries are still
written to the CSV.

Scopes (signed b̂, |b̂| and signed θ̂ each; |b̂| is kept in the CSV only):
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
  Human_big5  — 5 traits, IPIP human sample (20,993)
  Human_risk  — 9 risk subscales, risk human sample (1,507)

NOTE on humans: the Big-Five sample and the risk sample are DISJOINT people, so
a single human cross-family 14x14 cannot be formed; humans are reported within
each family only. The stability figures show LLM panels only (bias left, trait
right); the comparison figure adds humans below the LLMs, Big Five x risk empty.

Facet level: within the IPIP-NEO-300, every one of the 30 facets has items of
both keyings, so the trait component is also computed per facet (same formula)
for the LLMs and the human sample; the figures show both 30x30 facet matrices
(panels E, F), and facet_trait_correlations.csv holds one row per facet pair.

Outputs:
  results/tables/bias_stability_summary.csv
  results/tables/facet_trait_correlations.csv
  results/tables/bias_stability.tex
  results/figures/bias_stability_heatmaps.pdf          (drop constant on ≥1)
  results/figures/bias_stability_heatmaps_excl2.pdf    (drop constant on ≥2)
  results/figures/bias_stability_heatmaps_excl3.pdf    (drop constant on ≥3)
  results/figures/stability_heatmaps_llms_vs_humans.pdf (all LLMs vs humans)
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

# IPIP-NEO-300 facets for the facet-level trait panels: domains in the paper's
# order, facets in the IPIP-NEO order within each domain. Every facet has 10
# items with both keyings (2-8 reverse-keyed).
FACETS = {
    "O": ["Imagination", "Artistic Interests", "Emotionality", "Adventurousness", "Intellect", "Liberalism"],
    "C": ["Self-Efficacy", "Orderliness", "Dutifulness", "Achievement-Striving", "Self-Discipline", "Cautiousness"],
    "E": ["Friendliness", "Gregariousness", "Assertiveness", "Activity Level", "Excitement-Seeking", "Cheerfulness"],
    "A": ["Trust", "Morality", "Altruism", "Cooperation", "Modesty", "Sympathy"],
    "N": ["Anxiety", "Anger", "Depression", "Self-Consciousness", "Immoderation", "Vulnerability"],
}
FACET_ORDER = [f for fs in FACETS.values() for f in fs]
FACET_TICK = {f: f"{d}{i + 1} {f}" for d, fs in FACETS.items() for i, f in enumerate(fs)}

FIGURE_BASENAME = "bias_stability_heatmaps"
# Bottom-row exclusion thresholds: one figure per t, dropping models that
# answered constantly on ≥ t subscales. t=1 (strict "var>0 everywhere") keeps
# the original filename; t≥2 get an "_excl{t}" suffix.
NOVAR_THRESHOLDS = (1, 2, 3)

# ============================================================
# Per-responder, per-subscale normalised bias b_norm and trait theta_norm
# (identical formulas on both sides; b_norm matches 08.compute_mu_a_risk)
# ============================================================


def normalised_contrast(Rf, Rr, smin: float, smax: float, component: str):
    """b_norm = ((R̄_f + R̄_r)/2 - midpoint) / half_range  (component="bias")
    theta_norm = ((R̄_f - R̄_r)/2) / half_range           (component="trait")"""
    half_range = (smax - smin) / 2
    if component == "bias":
        return ((Rf + Rr) / 2 - (smin + smax) / 2) / half_range
    return ((Rf - Rr) / 2) / half_range


def llm_subscale(llm_df: pd.DataFrame, component: str) -> pd.DataFrame:
    """Wide [model x subscale] table of the normalised bias or trait, with
    R̄_f / R̄_r the model's mean raw answer over forward / reverse items."""
    rows = []
    for exp, cat, label, smin, smax in SUBSCALES:
        sub = _subscale_slice(llm_df, exp, cat)
        g = sub.groupby(["model", "reverse_coded"])["model_answer"].mean().unstack()
        for model in g.index:
            Rf = g.loc[model].get(False, np.nan)
            Rr = g.loc[model].get(True, np.nan)
            rows.append({
                "model": model,
                "subscale": label,
                "value": normalised_contrast(Rf, Rr, smin, smax, component),
            })
    long = pd.DataFrame(rows)
    return long.pivot(index="model", columns="subscale", values="value")


def human_subscale(human_items: pd.DataFrame, component: str) -> pd.DataFrame:
    """Long [person_id, subscale, value] per-person bias or trait, same formula
    as the LLM side but per respondent. Kept long because the Big-Five and risk
    samples are disjoint people — they are pivoted to wide separately."""
    rows = []
    for exp, cat, label, smin, smax in SUBSCALES:
        sub = _subscale_slice(human_items, exp, cat)
        if sub.empty:
            continue
        Rf = sub[~sub["reverse_coded"]].groupby("person_id")["response"].mean()
        Rr = sub[sub["reverse_coded"]].groupby("person_id")["response"].mean()
        idx = Rf.index.intersection(Rr.index)
        v = normalised_contrast(Rf.loc[idx], Rr.loc[idx], smin, smax, component)
        rows.append(pd.DataFrame({"person_id": idx, "subscale": label, "value": v.values}))
    return pd.concat(rows, ignore_index=True)


def to_wide(long_df: pd.DataFrame, group_col: str, cols: list[str]) -> pd.DataFrame:
    """Pivot to [responder × subscale] in the canonical column order, keeping
    only responders with a value on every requested subscale."""
    wide = long_df.pivot(index=group_col, columns="subscale", values="value")
    return wide.reindex(columns=cols).dropna()


def facet_trait(df: pd.DataFrame, unit: str, x: str) -> pd.DataFrame:
    """Wide [unit x facet] theta_norm on the 30 IPIP-NEO-300 facets (1-5 scale);
    units missing a keying on any facet are dropped."""
    g = df.groupby([unit, "facet", "reverse_coded"])[x].mean().unstack("reverse_coded")
    theta = normalised_contrast(g[False], g[True], 1, 5, "trait").unstack("facet")
    return theta.reindex(columns=FACET_ORDER).dropna()


def load_facet_trait(llm_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Facet-level theta_norm for the LLMs (same model set as llm_df) and the
    human IPIP sample. The facet of each item comes from the human item key."""
    human = pd.read_csv(m08.HUMAN_IPIP_CSV, low_memory=False,
                        usecols=["person_id", "item", "category", "response", "reverse_coded"])
    human["response"] = pd.to_numeric(human["response"], errors="coerce")
    human = human[human["response"].between(1, 5)].rename(columns={"category": "facet"})
    human["reverse_coded"] = human["reverse_coded"].astype(bool)

    facet_of_item = human.drop_duplicates("item").set_index("item")["facet"]
    llm = llm_df[llm_df["experiment"] == "IPIP-NEO-300"].copy()
    llm["facet"] = pd.to_numeric(llm["item_id"], errors="coerce").map(facet_of_item)
    assert llm["facet"].notna().all(), "IPIP item without a facet"
    return facet_trait(llm, "model", "model_answer"), facet_trait(human, "person_id", "response")


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
    # Headline LLM matrix: the models with a bias on all 14 subscales. The
    # within-family sub-blocks reuse this SAME model set so any difference is
    # the subscales, not the sample. Bias and trait come from the same
    # forward/reverse means, so they cover the same models.
    llm_all = llm_subscale(llm_df, "bias").reindex(columns=SUBSCALE_LABELS).dropna()
    llm_trait = llm_subscale(llm_df, "trait").loc[llm_all.index, SUBSCALE_LABELS]
    print(f"  LLMs with all 14 subscales: {len(llm_all)}")

    # How many subscales each model answered constantly on (zero within-subscale
    # variance). The bottom row of each figure drops models constant on ≥ t of
    # them, for the thresholds in NOVAR_THRESHOLDS.
    const_counts = constant_scale_counts(llm_df)
    print(f"  models constant on ≥1 subscale: {(const_counts >= 1).sum()}")

    print("Loading human per-person item data via figure-08 loader...")
    human_items = m08.load_human_items()
    human_long = {c: human_subscale(human_items, c) for c in ("bias", "trait")}
    human_wide = {(c, fam): to_wide(human_long[c], "person_id", cols)
                  for c in ("bias", "trait")
                  for fam, cols in (("big5", BIG5_LABELS), ("risk", RISK_LABELS))}
    print(f"  Humans with all 5 Big-Five traits: {len(human_wide[('bias', 'big5')])}")
    print(f"  Humans with all 9 risk subscales : {len(human_wide[('bias', 'risk')])}")

    # (component, scope name) -> signed wide matrix; |b̂| versions derived below.
    base = {}
    for component, llm in (("bias", llm_all), ("trait", llm_trait)):
        base[(component, "LLM_all")] = llm
        base[(component, "LLM_big5")] = llm[BIG5_LABELS]
        base[(component, "LLM_risk")] = llm[RISK_LABELS]
        base[(component, "Human_big5")] = human_wide[(component, "big5")]
        base[(component, "Human_risk")] = human_wide[(component, "risk")]
        # One var-filtered scope family per threshold (LLM_novar{t}). t=1 is the
        # strict "var>0 on every subscale" filter; higher t tolerates a few
        # constant subscales before dropping the model.
        for t in NOVAR_THRESHOLDS:
            dropped = [m for m in const_counts.index[const_counts >= t] if m in llm.index]
            novar = llm.drop(index=dropped)
            base[(component, f"LLM_novar{t}")]      = novar
            base[(component, f"LLM_novar{t}_big5")] = novar[BIG5_LABELS]
            base[(component, f"LLM_novar{t}_risk")] = novar[RISK_LABELS]
            if component == "bias":
                print(f"  novar (drop constant on ≥{t}): dropped {len(dropped)}, kept {len(novar)}")

    summary = []
    corr_pearson, corr_spearman = {}, {}
    for (component, scope), X_signed in base.items():
        kinds = (("signed", X_signed), ("abs", X_signed.abs())) if component == "bias" \
            else (("trait", X_signed),)
        for kind, X in kinds:
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
                "mean_abs_r_pearson": mean_offdiag(Cp.abs()),
                "mean_r_spearman": mean_offdiag(Cs),
                "cronbach_alpha":  cronbach_alpha(X.values),
                "icc_2_1":         icc_2_1(X.values),
            })

    summary_df = pd.DataFrame(summary)
    print("\n=== Response bias and trait stability summary ===")
    print(summary_df.drop(columns=["scope", "kind"]).round(3).to_string(index=False))

    summary_df.to_csv(OUT_TABLE_DIR / "bias_stability_summary.csv", index=False)

    # LaTeX (bare tabular; the SI supplies the table environment and caption)
    sm = summary_df.set_index(["scope", "kind"])
    tex_rows = [("LLMs, all (sub-)scales", "LLM_all"), ("LLMs, Big Five only", "LLM_big5"),
                ("LLMs, risk only", "LLM_risk"),
                (r"LLMs, all, drop const.\ $\ge 3$", "LLM_novar3"), None,
                ("Humans, Big Five", "Human_big5"), ("Humans, risk", "Human_risk")]
    # Bias only: the trait components are distinct constructs, so treating them
    # as items of one latent (alpha, ICC) does not apply; they are shown in the
    # figure and kept in the CSV.
    stat_cols = ("mean_r_pearson", "cronbach_alpha", "icc_2_1")
    lines = [r"\begin{tabular}{lrr ccc}", r"\toprule",
             r"sample & $n$ & $k$ & $\bar r$ & $\alpha$ & ICC \\", r"\midrule"]
    for row in tex_rows:
        if row is None:
            lines.append(r"\midrule"); continue
        label, scope = row
        b = sm.loc[(scope, "signed")]
        cells = [f"{b[c]:.2f}" for c in stat_cols]
        lines.append(f"{label} & {int(b['n']):,} & {int(b['n_subscales'])} & "
                     + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(OUT_TABLE_DIR / "bias_stability.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Facet level (IPIP-NEO-300 only): trait correlations between the 30 facets in
    # LLMs (same model set) and humans, and how the two matrices agree over the
    # 435 facet pairs, signed and absolute.
    print("Facet-level trait component...")
    facet_llm, facet_hum = load_facet_trait(llm_df[llm_df["model"].isin(llm_all.index)])
    facet_corr = {"LLM": facet_llm.corr(), "Human": facet_hum.corr()}
    print(f"  LLMs: {len(facet_llm)}   humans with all 30 facets: {len(facet_hum)}")
    iu = np.triu_indices(len(FACET_ORDER), k=1)
    pairs = pd.DataFrame({
        "facet_1": [FACET_ORDER[i] for i in iu[0]],
        "facet_2": [FACET_ORDER[j] for j in iu[1]],
        "same_domain": [i // 6 == j // 6 for i, j in zip(*iu)],
        "r_llm": facet_corr["LLM"].values[iu],
        "r_human": facet_corr["Human"].values[iu],
    })
    pairs.to_csv(OUT_TABLE_DIR / "facet_trait_correlations.csv", index=False)
    for label, d in (("all", pairs), ("within domain", pairs[pairs["same_domain"]]),
                     ("between domains", pairs[~pairs["same_domain"]])):
        print(f"  {label:16s} n={len(d):3d}  mean|r| LLM={d['r_llm'].abs().mean():.2f} "
              f"human={d['r_human'].abs().mean():.2f}  "
              f"r(LLM, human) signed={np.corrcoef(d['r_llm'], d['r_human'])[0, 1]:.2f} "
              f"absolute={np.corrcoef(d['r_llm'].abs(), d['r_human'].abs())[0, 1]:.2f}")

    # ============================================================
    # Figures — one per NOVAR_THRESHOLDS entry, each 3×2:
    #   A, B : ALL models                       — signed b̂ and θ̂ (14×14)
    #   C, D : models constant on < t subscales — signed b̂ and θ̂ (14×14)
    #   E, F : facet-level θ̂, LLMs and humans (30×30, lower triangle)
    # Only C and D change across the three figures. The human 14-subscale values
    # appear only in the separate LLM-vs-human figure; |b̂| is in the CSV only.
    # ============================================================
    for t in NOVAR_THRESHOLDS:
        build_figure(corr_pearson, summary_df, t, facet_corr, (len(facet_llm), len(facet_hum)))
    build_human_comparison_figure(corr_pearson, summary_df)


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
            if np.isnan(A[i, j]):  # empty cell (human cross-family block)
                continue
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                    fontsize=cell_fs,
                    color="black" if abs(A[i, j]) < 0.6 else "white")
    if block_split is not None:
        ax.axhline(block_split - 0.5, color="black", lw=1.2)
        ax.axvline(block_split - 0.5, color="black", lw=1.2)
    ax.set_title(title, fontsize=9)
    return im


def draw_facets(ax, C: pd.DataFrame, title: str) -> None:
    """30x30 facet heatmap, lower triangle only, with cell annotations and
    domain blocks."""
    A = C.values.copy()
    n = len(A)
    A[np.triu_indices(n)] = np.nan
    ax.imshow(A, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
    for i in range(n):
        for j in range(i):
            ax.text(j, i, f"{A[i, j]:.2f}".replace("0.", "."), ha="center", va="center",
                    fontsize=3.6, color="white" if abs(A[i, j]) >= 0.6 else "black")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([FACET_TICK[f] for f in C.columns], fontsize=5.5, rotation=90)
    ax.set_yticklabels([FACET_TICK[f] for f in C.index], fontsize=5.5)
    for b in range(6, n, 6):  # domain blocks, below the diagonal only
        ax.plot([-0.5, b - 0.5], [b - 0.5, b - 0.5], color="black", lw=1)
        ax.plot([b - 0.5, b - 0.5], [b - 0.5, n - 0.5], color="black", lw=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_title(title, fontsize=9)


def build_figure(corr_pearson: dict, summary_df: pd.DataFrame, threshold: int,
                 facet_corr: dict, facet_n: tuple[int, int]) -> None:
    """Render + save the 3x2 heatmap figure: LLM bias and trait on the 14
    (sub-)scales (top), the same after dropping models that answered constantly
    on ≥ `threshold` subscales (middle), facet-level trait in LLMs and humans
    (bottom).

    t=1 keeps the original filename (bias_stability_heatmaps.pdf); t≥2
    get an "_excl{t}" suffix."""
    def s(name, field):
        return summary_df.loc[summary_df["condition"] == name, field].iloc[0]

    def bias_stats(scope_all, prefix):
        """Bias panel subtitle (the table's statistics); scope_all is the
        14-subscale scope, prefix + _big5 / _risk the within-family scopes of
        the same model set. Trait panels get none: alpha and ICC assume one
        latent, and the traits are distinct constructs."""
        return (rf"all: $\bar r$={s(f'{scope_all}_signed', 'mean_r_pearson'):.2f}, "
                rf"$\alpha$={s(f'{scope_all}_signed', 'cronbach_alpha'):.2f}, "
                rf"ICC={s(f'{scope_all}_signed', 'icc_2_1'):.2f}   |   "
                rf"Big5 $\alpha$={s(f'{prefix}_big5_signed', 'cronbach_alpha'):.2f}, "
                rf"risk $\alpha$={s(f'{prefix}_risk_signed', 'cronbach_alpha'):.2f}")

    nv = f"LLM_novar{threshold}"
    plural = "s" if threshold > 1 else ""
    n_all = int(s("LLM_all_signed", "n"))
    n_kept = int(s(f"{nv}_signed", "n"))

    fig, axes = plt.subplots(3, 2, figsize=(13, 20.5), gridspec_kw={"height_ratios": [1, 1, 1.35]})
    for row, (scope_all, prefix, heading) in enumerate((
        ("LLM_all", "LLM", f"LLMs (n={n_all})"),
        (nv, nv, rf"LLMs, drop constant on $\geq${threshold} subscale{plural} (n={n_kept})"),
    )):
        im = draw(axes[row, 0], corr_pearson[f"{scope_all}_signed"],
                  rf"{heading} — bias $\hat b$" "\n" + bias_stats(scope_all, prefix),
                  block_split=N_BIG5)
        draw(axes[row, 1], corr_pearson[f"{scope_all}_trait"],
             rf"{heading} — trait $\hat\theta$", block_split=N_BIG5)

    for col, (who, n) in enumerate((("LLMs", facet_n[0]), ("Humans", facet_n[1]))):
        C = facet_corr["LLM" if col == 0 else "Human"]
        offdiag = np.abs(C.values[np.tril_indices(len(C), k=-1)])
        draw_facets(axes[2, col], C, rf"{who} (n={n:,}) — trait $\hat\theta$, IPIP-NEO-300 facets"
                    "\n" rf"mean |r|={offdiag.mean():.2f}")

    for ax, letter in zip(axes.ravel(), "ABCDEF"):
        ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontsize=14, fontweight="bold")

    # No suptitle: the SI caption describes the figure.
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5,
                 label="Pearson r between (sub-)scales or facets")

    out_name = f"{FIGURE_BASENAME}{'' if threshold == 1 else f'_excl{threshold}'}.pdf"
    fig.savefig(OUT_FIG_DIR / out_name, bbox_inches="tight", dpi=200)
    print(f"Saved: {OUT_FIG_DIR / out_name}")
    plt.close(fig)


def human_block_matrix(corr_pearson: dict, kind: str) -> pd.DataFrame:
    """14x14 human matrix from the two within-family matrices; the Big Five x
    risk cells stay empty because the two human samples are disjoint people."""
    C = pd.DataFrame(np.nan, index=SUBSCALE_LABELS, columns=SUBSCALE_LABELS)
    for fam, cols in (("big5", BIG5_LABELS), ("risk", RISK_LABELS)):
        C.loc[cols, cols] = corr_pearson[f"Human_{fam}_{kind}"].loc[cols, cols].values
    return C


def build_human_comparison_figure(corr_pearson: dict, summary_df: pd.DataFrame) -> None:
    """2x2 like the stability figures: LLMs (top, all models) and humans (bottom),
    bias left, trait right."""
    def s(name, field):
        return summary_df.loc[summary_df["condition"] == name, field].iloc[0]

    def alphas(prefix):
        return (rf"Big5 $\alpha$={s(f'{prefix}_big5_signed', 'cronbach_alpha'):.2f}, "
                rf"risk $\alpha$={s(f'{prefix}_risk_signed', 'cronbach_alpha'):.2f}")

    n_llm = int(s("LLM_all_signed", "n"))
    human = (f"Humans (Big Five n={int(s('Human_big5_signed', 'n')):,}; "
             f"risk n={int(s('Human_risk_signed', 'n')):,})")

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    im = draw(axes[0, 0], corr_pearson["LLM_all_signed"],
              rf"LLMs (n={n_llm}) — bias $\hat b$" "\n" + alphas("LLM"), block_split=N_BIG5)
    draw(axes[0, 1], corr_pearson["LLM_all_trait"],
         rf"LLMs (n={n_llm}) — trait $\hat\theta$", block_split=N_BIG5)
    draw(axes[1, 0], human_block_matrix(corr_pearson, "signed"),
         rf"{human} — bias $\hat b$" "\n" + alphas("Human"), block_split=N_BIG5)
    draw(axes[1, 1], human_block_matrix(corr_pearson, "trait"),
         rf"{human} — trait $\hat\theta$", block_split=N_BIG5)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7,
                 label="Pearson r between subscales")
    out = OUT_FIG_DIR / "stability_heatmaps_llms_vs_humans.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
