"""SI: robustness of the behavioural-task results to DFD and LOT.

DFD's keying coincides with outcome domain: the four items with the risky option
first are all gain problems, the four with it second all loss problems. Under the
reflection effect (risk aversion for gains, risk seeking for losses) a respondent
with purely content-driven choices therefore picks position 2 on both and looks
position-biased, so DFD cannot separate response bias from content. In LOT, gain
and loss gambles appear in both keyings (mixed gambles only with the risky option
second); its option order was randomised per trial for humans, which the human
position coding below accounts for.

Reports, for the reference condition:
  A. Fig 4A: cross-instrument correlation of p_r with mean inter-item correlation,
     LLMs and humans, with all instruments and without DFD, LOT, or both
     (instrument-clustered bootstrap CI and wild-cluster p, as in script 02);
  B. behavioural tasks: mean inter-item correlation (LLMs vs humans), and the
     keying shift b on LOT and DFD per model and per person (0-1 response scale,
     positive = favours the first-listed option), corrected for item noise with
     the same measurement-error model for both populations; plus the human
     risky-choice share by DFD keying (the reflection effect);
  C. Fig 5: per-model mean |normalised b| (item-noise corrected, as annotated in
     Fig 5) over the 14 (sub-)scales vs the 13 without DFD (rank agreement and
     group summaries);
  D. all six behavioural tasks in the weighted condition (BART, CCT, DFE only
     exist there), LLMs vs humans, and the behavioural-only gradient with and
     without the gain/loss-confounded DFD and DFE;
  E. within model: LOT keying shift vs mean self-report bias (sign agreement
     against chance, and magnitude).

Reads results/tables/orthogonality_subscales_{reference,by_condition}.csv
(script 02) and the data loaders of 04_profile_instability/01_profiles_per_keying.py.
Writes results/tables/behavioural_tasks_{gradient,keying_shift,fig5_without_dfd,
weighted,weighted_gradient,lot_vs_selfreport}.csv and behavioural_tasks_robustness.tex.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from utils import cluster_bootstrap_ci, wild_cluster_p

HERE = Path(__file__).resolve().parent
TABLE_DIR = HERE.parents[2] / "results/tables"

# Fig 5 script: its loaders (incl. position-coded human LOT/DFD) and |mu_b| summary.
# Importing it exits with the run_all "skipped" code if the Frey source files are absent.
_spec = importlib.util.spec_from_file_location(
    "profiles", HERE.parent / "04_profile_instability/01_profiles_per_keying.py")
profiles = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(profiles)

BEHAVIOURAL = ["MPL", "LOT", "DFD"]

# ============================================================
# A. Fig 4A gradient without DFD / LOT
# ============================================================

sub = pd.read_csv(TABLE_DIR / "orthogonality_subscales_reference.csv")
sub = sub.dropna(subset=["pct_reversed_keyed_items", "mean_interitem_corr"])

grad = []
for pop in ["LLMs", "Humans"]:
    for label, drop in [("all", []), ("without DFD", ["DFD"]),
                        ("without LOT", ["LOT"]), ("without DFD and LOT", ["DFD", "LOT"])]:
        d = sub[(sub["population"] == pop) & ~sub["experiment"].isin(drop)]
        x = d["pct_reversed_keyed_items"].to_numpy()
        y = d["mean_interitem_corr"].to_numpy()
        g = d["experiment"].to_numpy()
        lo, hi = cluster_bootstrap_ci(x, y, g)[0]
        grad.append({"population": pop, "sample": label, "n_subscales": len(d),
                     "n_instruments": len(np.unique(g)), "r": stats.pearsonr(x, y)[0],
                     "ci_cluster_lo": lo, "ci_cluster_hi": hi,
                     "p_wild_cluster": wild_cluster_p(x, y, g)})
grad = pd.DataFrame(grad)
grad.to_csv(TABLE_DIR / "behavioural_tasks_gradient.csv", index=False)

# ============================================================
# B. Behavioural tasks: consistency and keying shift
# ============================================================

interitem = (sub[sub["experiment"].isin(BEHAVIOURAL)]
             .pivot(index="experiment", columns="population", values="mean_interitem_corr"))


def keying_shift(df, unit, value):
    """Per-unit b = (forward mean + reverse mean)/2 - .5 on the 0-1 choice scale and
    its item-noise variance (s_f^2/n_f + s_r^2/n_r)/4, with s^2 taken on responses
    centred within item x keying (item difficulty is not noise; human LOT keying
    varies per trial), as in 02_capability_and_bias/utils.py. Called with one
    population at a time, so items are centred within that population."""
    df = df.assign(resid=df[value] - df.groupby(["item_id", "reverse_coded"])[value].transform("mean"))
    g = df.groupby([unit, "reverse_coded"]).agg(
        mean=(value, "mean"), var=("resid", "var"), count=(value, "count")).unstack().dropna()
    b = (g[("mean", False)] + g[("mean", True)]) / 2 - 0.5
    err = (g[("var", False)] / g[("count", False)] + g[("var", True)] / g[("count", True)]) / 4
    return b, err


def attenuation_correct(b, err):
    """Pull b toward its mean by sqrt(reliability) so the spread equals the
    estimated true spread (same correction as Fig 3, 02_capability_and_bias/01)."""
    var_obs = b.var()
    rel = max(var_obs - err.mean(), 0.0) / var_obs
    return b.mean() + np.sqrt(rel) * (b - b.mean()), rel


llm = profiles.load_combined_reference()
human = profiles.load_human_items()

shift = []
for task in ["LOT", "DFD"]:
    b_l, e_l = keying_shift(llm[llm["experiment"] == task], "model", "model_answer")
    # Human LOT: first-stage trials only (same 25 gambles as the LLMs; the adaptive
    # second stage shows person-specific gambles).
    b_h, e_h = keying_shift(profiles.lot_first_stage(human[human["experiment"] == task]),
                            "person_id", "response")
    c_l, rel_l = attenuation_correct(b_l, e_l)
    c_h, rel_h = attenuation_correct(b_h, e_h)
    h95 = np.percentile(np.abs(c_h), 95)
    for pop, b, c, rel in [("LLMs", b_l, c_l, rel_l), ("Humans", b_h, c_h, rel_h)]:
        shift.append({
            "task": task, "population": pop, "n": len(b),
            "mean_signed_b": b.mean(), "median_abs_b": np.median(np.abs(b)),
            "median_abs_b_corr": np.median(np.abs(c)), "reliability": rel,
            "share_above_human_p95_corr": float(np.mean(np.abs(c) > h95)),
            "mean_interitem_corr": interitem.loc[task, pop],
        })
shift = pd.DataFrame(shift)

# DFD reflection effect in humans: risky-choice share by keying (= by domain).
dfd_h = human[human["experiment"] == "DFD"]
chose_risky = np.where(dfd_h["reverse_coded"], 1 - dfd_h["response"], dfd_h["response"])
risky_share = pd.Series(chose_risky).groupby(dfd_h["reverse_coded"].to_numpy()).mean()
shift["human_dfd_risky_share_gain"] = risky_share[False]   # risky first = gain problems
shift["human_dfd_risky_share_loss"] = risky_share[True]    # risky second = loss problems
shift.to_csv(TABLE_DIR / "behavioural_tasks_keying_shift.csv", index=False)

# ============================================================
# C. Fig 5 without DFD
# ============================================================

ALL_ROWS = list(profiles.SUBSCALES)
NO_DFD = [r for r in ALL_ROWS if r[0] != "DFD"]
BIG5 = [r for r in ALL_ROWS if r[0] == "IPIP-NEO-300"]


def mu_abs(rows):
    """Per-model mean |normalised b| over `rows`, item-noise corrected as in Fig 5."""
    profiles.SUBSCALES = rows
    try:
        return profiles.compute_mu_a_corrected(llm)
    finally:
        profiles.SUBSCALES = ALL_ROWS


def human_mu_abs(rows):
    profiles.SUBSCALES = rows
    try:
        return profiles.compute_human_mu_a_corrected(human)
    finally:
        profiles.SUBSCALES = ALL_ROWS


prop_names = profiles.load_proprietary_model_names()
mu = pd.DataFrame({"with_dfd": mu_abs(ALL_ROWS), "without_dfd": mu_abs(NO_DFD)})
is_prop = mu.index.isin(prop_names)
fig5 = []
for col, rows in [("with_dfd", ALL_ROWS), ("without_dfd", NO_DFD)]:
    risk_rows = [r for r in rows if r[0] != "IPIP-NEO-300"]
    fig5.append({
        "sample": col.replace("_", " "), "n_subscales": len(rows),
        "min": mu[col].min(), "max": mu[col].max(),
        "mean_proprietary": mu.loc[is_prop, col].mean(),
        "mean_open_weight": mu.loc[~is_prop, col].mean(),
        "mean_personality": mu_abs(BIG5).mean(),
        "mean_risk": mu_abs(risk_rows).mean(),
        "human_median_individual": human_mu_abs(rows),
    })
fig5 = pd.DataFrame(fig5)
fig5["spearman_rank_with_vs_without"] = stats.spearmanr(mu["with_dfd"], mu["without_dfd"])[0]
fig5["max_rank_change"] = int((mu["with_dfd"].rank() - mu["without_dfd"].rank()).abs().max())
fig5.to_csv(TABLE_DIR / "behavioural_tasks_fig5_without_dfd.csv", index=False)

# ============================================================
# D. Behavioural-only gradient in the weighted condition (BART, CCT, DFE added)
# ============================================================
# DFE uses the same eight problems and keying as DFD, so it carries the same
# gain/loss confound; the clean set is BART, CCT, MPL (one keying) and LOT.

by_cond = pd.read_csv(TABLE_DIR / "orthogonality_subscales_by_condition.csv")
BEHAV_ALL = ["BART", "CCT", "MPL", "LOT", "DFE", "DFD"]
weighted = by_cond[(by_cond["condition"] == "LLM weighted (no flip)")
                   & by_cond["experiment"].isin(BEHAV_ALL)].set_index("experiment")
humans_b = pd.concat([
    by_cond[(by_cond["condition"] == "Humans") & by_cond["experiment"].isin(BEHAV_ALL)],
    by_cond[by_cond["condition"] == "Humans (BART, CCT, DFE)"],
]).set_index("experiment")
behav = pd.DataFrame({
    "p_r": weighted["pct_reversed_keyed_items"],
    "interitem_llm_weighted": weighted["mean_interitem_corr"],
    "interitem_humans": humans_b["mean_interitem_corr"],
}).reindex(BEHAV_ALL)

behav_grad = []
for label, tasks in [("all six", BEHAV_ALL),
                     ("without DFD and DFE", ["BART", "CCT", "MPL", "LOT"])]:
    d = behav.loc[tasks]
    x, y = d["p_r"].to_numpy(), d["interitem_llm_weighted"].to_numpy()
    g = np.array(tasks)   # one point per instrument, so each is its own cluster
    lo, hi = cluster_bootstrap_ci(x, y, g)[0]
    behav_grad.append({"sample": label, "n_tasks": len(tasks), "r": stats.pearsonr(x, y)[0],
                       "ci_cluster_lo": lo, "ci_cluster_hi": hi,
                       "p_wild_cluster": wild_cluster_p(x, y, g)})
behav_grad = pd.DataFrame(behav_grad)
behav.to_csv(TABLE_DIR / "behavioural_tasks_weighted.csv")
behav_grad.to_csv(TABLE_DIR / "behavioural_tasks_weighted_gradient.csv", index=False)

# ============================================================
# E. LOT vs self-report bias within model: sign and magnitude
# ============================================================
# Per model, normalised signed b on every (sub-)scale (Fig 5 function, one row at
# a time). On LOT, + = favours the first-listed option; on the rating scales,
# + = favours the high end of the scale, so sign agreement is compared with the
# agreement expected if the two signs were independent.

def signed_b(row):
    profiles.SUBSCALES = [row]
    try:
        return profiles.compute_mu_a_risk(llm)["mu_a_risk"]
    finally:
        profiles.SUBSCALES = ALL_ROWS


SELF_REPORT = [r for r in ALL_ROWS if r[0] not in ("LOT", "DFD")]
b_sr = pd.concat([signed_b(r) for r in SELF_REPORT], axis=1)
b_lot = signed_b(next(r for r in ALL_ROWS if r[0] == "LOT"))
sr_mean = b_sr.mean(axis=1)
pos_lot, pos_sr = (b_lot > 0).mean(), (sr_mean > 0).mean()
sr_abs = b_sr.abs().mean(axis=1)
within = pd.DataFrame([{
    "n_models": len(b_lot),
    "share_lot_positive": pos_lot,
    "share_selfreport_positive": pos_sr,
    "sign_agreement": float((np.sign(b_lot) == np.sign(sr_mean)).mean()),
    "sign_agreement_if_independent": pos_lot * pos_sr + (1 - pos_lot) * (1 - pos_sr),
    "pearson_signed": stats.pearsonr(b_lot, sr_mean)[0],
    "spearman_signed": stats.spearmanr(b_lot, sr_mean)[0],
    "median_abs_lot": b_lot.abs().median(),
    "median_abs_selfreport": sr_abs.median(),
    "share_lot_ge_selfreport": float((b_lot.abs() >= sr_abs).mean()),
    "wilcoxon_p_abs": stats.wilcoxon(b_lot.abs(), sr_abs).pvalue,
}])
within.to_csv(TABLE_DIR / "behavioural_tasks_lot_vs_selfreport.csv", index=False)

# ============================================================
# LaTeX: panel A (gradient) and panel B (behavioural tasks)
# ============================================================

def r_ci(row):
    return f"{row['r']:.2f} [{row['ci_cluster_lo']:.2f}, {row['ci_cluster_hi']:.2f}]"


tab_a = grad.pivot(index="sample", columns="population").reindex(
    ["all", "without DFD", "without LOT", "without DFD and LOT"])
rows_a = []
for s in tab_a.index:
    row = {"Instruments": s, "$n$": int(tab_a.loc[s, ("n_subscales", "LLMs")])}
    for pop in ["LLMs", "Humans"]:
        g = grad[(grad["sample"] == s) & (grad["population"] == pop)].iloc[0]
        row[f"$r$ {pop} [95\\% CI]"] = r_ci(g)
        p = g["p_wild_cluster"]
        row[f"$p$ {pop}"] = "$<$ .001" if p < .001 else f"{p:.3f}"
    rows_a.append(row)

rows_b = []
for task in BEHAVIOURAL:
    row = {"Task": task,
           "inter-item $r$ LLMs": f"{interitem.loc[task, 'LLMs']:.2f}",
           "inter-item $r$ humans": f"{interitem.loc[task, 'Humans']:.2f}"}
    s = shift[shift["task"] == task].set_index("population")
    if len(s):
        row["median $|\\hat b|$ LLMs"] = f"{s.loc['LLMs', 'median_abs_b_corr']:.2f}"
        row["median $|\\hat b|$ humans"] = f"{s.loc['Humans', 'median_abs_b_corr']:.2f}"
        row["LLMs $>$ human P95"] = f"{100 * s.loc['LLMs', 'share_above_human_p95_corr']:.0f}\\%"
    else:   # MPL: one keying only, no keying shift
        row.update({"median $|\\hat b|$ LLMs": "--", "median $|\\hat b|$ humans": "--",
                    "LLMs $>$ human P95": "--"})
    rows_b.append(row)

f5 = fig5.set_index("sample")
caption = (
    "Robustness of the behavioural-task results. (A) Cross-instrument correlation of "
    "$p_r$ with mean inter-item correlation (Fig.~4A), with instrument-clustered "
    "bootstrap 95\\% CI and wild-cluster bootstrap $p$. (B) Behavioural tasks: mean "
    "inter-item correlation, and median keying shift $|\\hat b|$ on the 0--1 choice "
    "scale (share choosing the first-listed option minus .5), corrected for item noise "
    "with the same error model for models and persons; last column: share of models "
    "above the 95th percentile of individual humans. DFD keying coincides with outcome "
    "domain (risky option first: gain problems; second: loss problems), so its keying "
    "shift is confounded with the reflection effect (humans choose the risky option on "
    f"{100 * shift['human_dfd_risky_share_gain'].iloc[0]:.0f}\\% of gain and "
    f"{100 * shift['human_dfd_risky_share_loss'].iloc[0]:.0f}\\% of loss problems). "
    "Without DFD, the per-model mean $|\\mu_b|$ of Fig.~5 keeps its ordering "
    f"(Spearman $\\rho$ = {f5['spearman_rank_with_vs_without'].iloc[0]:.2f}); "
    f"proprietary {f5.loc['with dfd', 'mean_proprietary']:.2f} $\\to$ "
    f"{f5.loc['without dfd', 'mean_proprietary']:.2f}, open-weight "
    f"{f5.loc['with dfd', 'mean_open_weight']:.2f} $\\to$ "
    f"{f5.loc['without dfd', 'mean_open_weight']:.2f}, risk block "
    f"{f5.loc['with dfd', 'mean_risk']:.2f} $\\to$ {f5.loc['without dfd', 'mean_risk']:.2f}. "
    "(C) All behavioural tasks in the weighted condition, the only one that administered "
    "BART, CCT and DFE; DFE uses the same problems and keying as DFD and shares its "
    "confound. Across the four unconfounded tasks, $p_r$ vs mean inter-item correlation: "
    f"$r$ = {behav_grad['r'].iloc[1]:.2f} (all six: {behav_grad['r'].iloc[0]:.2f}); with "
    "a single balanced unconfounded task (LOT) this is descriptive. "
    "Within model, the normalised LOT keying shift and the mean self-report bias have "
    f"median magnitudes {within['median_abs_lot'].iloc[0]:.2f} and "
    f"{within['median_abs_selfreport'].iloc[0]:.2f} (LOT $\\geq$ self-report in "
    f"{100 * within['share_lot_ge_selfreport'].iloc[0]:.0f}\\% of models); their signs "
    f"agree in {100 * within['sign_agreement'].iloc[0]:.0f}\\% of models against "
    f"{100 * within['sign_agreement_if_independent'].iloc[0]:.0f}\\% expected if independent."
)
rows_c = [{"Task": t, "$p_r$": f"{r['p_r']:.2f}",
           "inter-item $r$ LLMs (weighted)": f"{r['interitem_llm_weighted']:.2f}",
           "inter-item $r$ humans": f"{r['interitem_humans']:.2f}"}
          for t, r in behav.iterrows()]
with open(TABLE_DIR / "behavioural_tasks_robustness.tex", "w", encoding="utf-8") as f:
    f.write("\\begin{table}\n\\caption{" + caption + "}\n\\label{tab:behavioural_tasks}\n")
    f.write("\\par\\medskip\\noindent\\textbf{A}\\par\n")
    f.write(pd.DataFrame(rows_a).to_latex(index=False, escape=False, column_format="lrcccc"))
    f.write("\\par\\medskip\\noindent\\textbf{B}\\par\n")
    f.write(pd.DataFrame(rows_b).to_latex(index=False, escape=False, column_format="lccccc"))
    f.write("\\par\\medskip\\noindent\\textbf{C}\\par\n")
    f.write(pd.DataFrame(rows_c).to_latex(index=False, escape=False, column_format="lccc"))
    f.write("\\end{table}\n")

pd.set_option("display.width", 250)
for df in (grad, shift, fig5, behav, behav_grad, within):
    print(df.round(3).to_string())
    print()
