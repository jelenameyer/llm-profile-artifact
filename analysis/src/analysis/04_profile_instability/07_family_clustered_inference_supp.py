"""SI: sensitivity of the model-level inferences to clustering by model series and
developer (models of one family may not be independent).

  A. Table 1 / pi_b: forward-reverse correlation rho per Big Five trait (all models
     and the non-constant ones), percentile cluster bootstrap over series and over
     developers (whole clusters resampled); pi_b = (1 + rho) / 2 on the bounds.
  B. Size regression (Fig 3 bottom row): OLS of item-noise corrected |b_hat| on
     log10 total parameters, alone and with release date, open-weight models. CR1
     cluster-robust CI with t(G - 1) critical values, and wild cluster bootstrap-t p
     (null imposed, Rademacher weights, 99,999 draws).
  C. Open-weight vs proprietary (Fig 3 bottom row): Mann-Whitney on series means of
     |b_hat|. Developers mix open-weight and proprietary models, so only series can
     be used. No cluster-robust regression: with few treated clusters its wild
     bootstrap is unreliable (MacKinnon & Webb 2017).

Reads the reference-condition IPIP-NEO-300 data (02_capability_and_bias/utils.py)
and the family map of 03_clustering_icc_supp.py.
Writes results/tables/family_clustered_inference.{csv,tex}.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "02_capability_and_bias"))
from panel import MODEL_METADATA, MODEL_SIZE, bare_model_name  # noqa: E402
from utils import TRAITS, attenuation_correct, load_llm_reference, per_respondent_contrasts  # noqa: E402

_spec = importlib.util.spec_from_file_location("icc", HERE / "03_clustering_icc_supp.py")
icc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(icc)
FAMILY = icc.FAMILY

TABLE_DIR = HERE.parents[2] / "results/tables"
N_BOOT = 10_000
N_WILD = 99_999
SEED = 13


def fmt_p(p):
    return "$<$.001" if p < .001 else f"{p:.3f}".lstrip("0")


def num(x, d=2):
    return f"{x:.{d}f}".replace("-", "$-$")


# ============================================================
# Per-model frame
# ============================================================

raw = load_llm_reference()
contrasts = per_respondent_contrasts(raw, "model", "model_answer", "item_id")
attenuation_correct(contrasts)

models = pd.Index(sorted(raw["model"].unique()), name="model")
bare = models.map(bare_model_name)
missing = sorted(set(bare) - set(FAMILY))
assert not missing, f"Models without a family label: {missing}"
series = pd.Series([FAMILY[b][0] for b in bare], index=models)
developer = pd.Series([FAMILY[b][1] for b in bare], index=models)
assert len(models) == 60

# constant responders: zero variance over all IPIP answers (as in Table 1)
constant = raw.groupby("model")["model_answer"].var().fillna(0).loc[models] == 0


# ============================================================
# A. Table 1: rho(R_f, R_r) per trait, cluster bootstrap
# ============================================================

def fr_means(trait):
    sub = raw[raw["traits"] == trait]
    g = sub.groupby(["model", "reverse_coded"])["model_answer"].mean().unstack()
    return g[False].loc[models].to_numpy(), g[True].loc[models].to_numpy()


def cluster_boot_r(x, y, groups, rng):
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rs = []
    for _ in range(N_BOOT):
        idx = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
        if np.std(x[idx]) > 0 and np.std(y[idx]) > 0:
            rs.append(np.corrcoef(x[idx], y[idx])[0, 1])
    return np.percentile(rs, [2.5, 97.5])


def fisher_ci(r, n):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return np.tanh([z - 1.96 * se, z + 1.96 * se])


rows = []
for sample, keep in [("all", np.ones(len(models), bool)), ("excl. constant", ~constant.to_numpy())]:
    for t in TRAITS:
        rf, rr = (a[keep] for a in fr_means(t))
        r = np.corrcoef(rf, rr)[0, 1]
        row = {"analysis": "A. rho (Table 1)", "sample": sample, "trait": t, "n": int(keep.sum()),
               "estimate": r, "naive_lo": fisher_ci(r, keep.sum())[0], "naive_hi": fisher_ci(r, keep.sum())[1]}
        for level, grp in [("series", series), ("developer", developer)]:
            g = grp.to_numpy()[keep]
            lo, hi = cluster_boot_r(rf, rr, g, np.random.default_rng(SEED))
            row.update({f"{level}_lo": lo, f"{level}_hi": hi, f"{level}_k": len(np.unique(g))})
        rows.append(row)


# ============================================================
# B. Size regression, cluster-robust + wild cluster bootstrap
# ============================================================

abs_b = contrasts.assign(abs_b=contrasts["b_corr"].abs()).pivot(
    index="model", columns="trait", values="abs_b").loc[models, TRAITS]
abs_b["mean"] = abs_b[TRAITS].mean(axis=1)
def release_date(model):
    """Same matching as 03_capability_regression_supp.metadata_row: exact key, else
    the longest panel_key contained in the model string."""
    keys = MODEL_METADATA["panel_key"]
    hit = keys[keys == model]
    if hit.empty:
        hit = keys[[k in model for k in keys]]
        hit = hit[hit.str.len() == hit.str.len().max()]
    assert len(hit) == 1, (model, list(hit))
    return MODEL_METADATA.loc[hit.index[0], "release_date"]


release = pd.to_datetime(pd.Series([release_date(m) for m in models], index=models))
df = abs_b.assign(
    log_total=np.log10(models.map(MODEL_SIZE.get).astype(float)),
    release_year=2020 + (release - pd.Timestamp("2020-01-01")).dt.days / 365.25,
    series=series, developer=developer,
)
open_df = df[df["log_total"].notna()]
assert len(open_df) == 50


def cr1(X, y, g_idx, n_g, j):
    """Coefficient j and its CR1 cluster-robust standard error."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    S = np.zeros((n_g, X.shape[1]))
    np.add.at(S, g_idx, X * u[:, None])
    n, k = X.shape
    V = n_g / (n_g - 1) * (n - 1) / (n - k) * XtX_inv @ (S.T @ S) @ XtX_inv
    return beta[j], np.sqrt(V[j, j])


def cr1_t(X, y, g_idx, n_g, j):
    b, se = cr1(X, y, g_idx, n_g, j)
    return b / se


def wild_p(X, y, groups, j, rng):
    """Wild cluster bootstrap-t (Cameron, Gelbach & Miller 2008): null imposed by
    dropping column j, Rademacher weights per cluster."""
    labels, g_idx = np.unique(groups, return_inverse=True)
    n_g = len(labels)
    t_obs = cr1_t(X, y, g_idx, n_g, j)
    Xr = np.delete(X, j, axis=1)
    br = np.linalg.lstsq(Xr, y, rcond=None)[0]
    fit_r, e_r = Xr @ br, y - Xr @ br
    w = rng.choice([-1.0, 1.0], size=(N_WILD, n_g))
    t_star = np.array([cr1_t(X, fit_r + w[b, g_idx] * e_r, g_idx, n_g, j) for b in range(N_WILD)])
    return float((1 + np.sum(np.abs(t_star) >= np.abs(t_obs))) / (N_WILD + 1))


for spec, rhs in [("size", ["log_total"]), ("size + date", ["log_total", "release_year"])]:
    for outcome in ["mean"] + TRAITS:
        fit = smf.ols(f"Q('{outcome}') ~ " + " + ".join(rhs), data=open_df).fit()
        ci = fit.conf_int().loc["log_total"]
        row = {"analysis": f"B. |b| ~ {spec}", "sample": "open-weight", "trait": outcome,
               "n": len(open_df), "estimate": fit.params["log_total"],
               "naive_lo": ci[0], "naive_hi": ci[1], "naive_p": fit.pvalues["log_total"]}
        X = np.column_stack([np.ones(len(open_df))] + [open_df[c].to_numpy() for c in rhs])
        y = open_df[outcome].to_numpy()
        for level in ["series", "developer"]:
            g = open_df[level].to_numpy()
            labels, g_idx = np.unique(g, return_inverse=True)
            b, se = cr1(X, y, g_idx, len(labels), 1)
            q = stats.t.ppf(0.975, len(labels) - 1)
            lo, hi = b - q * se, b + q * se
            row.update({f"{level}_lo": lo, f"{level}_hi": hi, f"{level}_k": len(np.unique(g)),
                        f"{level}_p": wild_p(X, y, g, 1, np.random.default_rng(SEED))})
        rows.append(row)


# ============================================================
# C. Open-weight vs proprietary on series means
# ============================================================

df["proprietary"] = df["log_total"].isna()
ser = df.groupby("series").agg(**{t: (t, "mean") for t in ["mean"] + TRAITS},
                               proprietary=("proprietary", "first"),
                               mixed=("proprietary", "nunique"))
assert (ser["mixed"] == 1).all(), "a series mixes open-weight and proprietary models"
for outcome in ["mean"] + TRAITS:
    o, c = df.loc[~df["proprietary"], outcome], df.loc[df["proprietary"], outcome]
    so, sc = ser.loc[~ser["proprietary"], outcome], ser.loc[ser["proprietary"], outcome]
    rows.append({"analysis": "C. open vs proprietary", "sample": "all", "trait": outcome,
                 "n": len(df), "estimate": np.median(o) - np.median(c),
                 "naive_p": stats.mannwhitneyu(o, c, alternative="two-sided").pvalue,
                 "series_estimate": np.median(so) - np.median(sc),
                 "series_p": stats.mannwhitneyu(so, sc, alternative="two-sided").pvalue,
                 "series_k": len(ser), "series_k_open": len(so), "series_k_prop": len(sc)})

res = pd.DataFrame(rows)
res.to_csv(TABLE_DIR / "family_clustered_inference.csv", index=False)


# ============================================================
# LaTeX
# ============================================================

def ci(r, pre, pi=False):
    lo, hi = r[f"{pre}_lo"], r[f"{pre}_hi"]
    if pi:
        lo, hi = (1 + lo) / 2, (1 + hi) / 2
    return f"[{num(lo)}, {num(hi)}]"


body = ["\\multicolumn{5}{@{}l}{\\textit{A. Forward--reverse correlation $\\rho$ (Table 1) and "
        "$\\pi_b = (1+\\rho)/2$}} \\\\",
        "Sample, trait & $\\rho$ & Fisher-$z$ CI & series CI & developer CI \\\\", "\\midrule"]
for _, r in res[res["analysis"].str.startswith("A")].iterrows():
    lab = f"{int(r['n'])} {'models' if r['sample'] == 'all' else 'non-constant'}, {r['trait']}"
    body.append(f"{lab} & {num(r['estimate'])} & {ci(r, 'naive')} & {ci(r, 'series')} & "
                f"{ci(r, 'developer')} \\\\")
a = res[res["analysis"].str.startswith("A")]
pi_lo = (1 + a[["series_lo", "developer_lo"]].min(axis=1)) / 2
body += ["\\midrule",
         "\\multicolumn{5}{@{}l}{\\textit{B. Size slope of $|\\hat b|$ per decade, open-weight "
         f"models ($n$ = {len(open_df)})}}}} \\\\",
         "Model, outcome & $\\beta$ & OLS CI, $p$ & series CI, wild $p$ & developer CI, wild $p$ \\\\",
         "\\midrule"]
for _, r in res[res["analysis"].str.startswith("B")].iterrows():
    spec = r["analysis"].split("~ ")[1]
    lab = f"{spec}, {'mean' if r['trait'] == 'mean' else r['trait']}"
    body.append(f"{lab} & {num(r['estimate'])} & {ci(r, 'naive')}, {fmt_p(r['naive_p'])} & "
                f"{ci(r, 'series')}, {fmt_p(r['series_p'])} & "
                f"{ci(r, 'developer')}, {fmt_p(r['developer_p'])} \\\\")
c_rows = res[res["analysis"].str.startswith("C")]
kc = c_rows.iloc[0]
body += ["\\midrule",
         f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{C. Open-weight minus proprietary median $|\\hat b|$: "
         f"models ({int((~df['proprietary']).sum())} vs {int(df['proprietary'].sum())}) and series ({int(kc['series_k_open'])} vs {int(kc['series_k_prop'])})}}}} \\\\",
         "Outcome & models & Mann--Whitney $p$ & series & Mann--Whitney $p$ \\\\", "\\midrule"]
for _, r in c_rows.iterrows():
    body.append(f"{r['trait']} & {num(r['estimate'])} & {fmt_p(r['naive_p'])} & "
                f"{num(r['series_estimate'])} & {fmt_p(r['series_p'])} \\\\")

caption = (
    "Model-level inferences with uncertainty clustered by model family. "
    f"Families: {int(a.iloc[0]['series_k'])} series and {int(a.iloc[0]['developer_k'])} developers "
    "(Table~\\ref{tab:clustering_icc}). (A) Percentile cluster bootstrap (10{,}000 resamples of "
    "whole series or developers); $\\pi_b$ bounds follow as $(1+\\rho)/2$, the lowest clustered "
    f"lower bound of $\\pi_b$ being {pi_lo.min():.2f}. (B) OLS slope of item-noise corrected "
    "$|\\hat b|$ on $\\log_{10}$ total parameters, alone or with release date; clustered CIs are "
    "CR1 cluster-robust with $t(G-1)$ critical values, wild $p$ from a wild cluster bootstrap-$t$ test with the null imposed "
    "and Rademacher weights (99{,}999 draws). (C) Mann--Whitney test on models (as in Table~\\ref{tab:a_open_vs_prop}) "
    "and on series means; developers mix open-weight and proprietary models and cannot be used."
)
with open(TABLE_DIR / "family_clustered_inference.tex", "w", encoding="utf-8") as f:
    f.write("\\begin{table}\n\\caption{" + caption + "}\n\\label{tab:family_clustered}\n"
            "\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
            "\\begin{tabular}{@{}lllll@{}}\n\\toprule\n"
            + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

pd.set_option("display.width", 250)
print(res.round(3).to_string(index=False))
