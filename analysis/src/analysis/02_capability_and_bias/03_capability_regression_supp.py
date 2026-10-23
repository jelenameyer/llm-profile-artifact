"""SI: response bias against model size, release date and published capability
scores (IPIP-NEO-300, reference condition).

Outcome: |b_hat| per model, item-noise corrected exactly as in Fig 3, per Big Five
trait and averaged over the five traits. Predictors come from ../model_metadata.csv
(release date of the weights administered; nominal parameter counts as in Fig 3;
ECI downloaded 2026-10-22, Arena snapshot 2026-10-13). OLS with 95% CIs; Spearman rho for single predictors.

  1. log10(total parameters) + release date        open-weight models (n = 50)
  2. log10(active parameters) + release date       open-weight models (n = 50)
  3. as 1, without the four >=300B models           (n = 46, leverage check)
  4. release date + proprietary indicator           all models (n = 60)
  5. Epoch Capabilities Index (ECI)                 models with a score
  6. ECI + release date                             models with a score
  7. MMLU-Pro (Open LLM Leaderboard v2, normalised) models with a score
     (covers the small open-weight models ECI misses)
  8. LMArena text Elo (style-controlled), per 100 points, models with a score
Each model is fitted on all models and again without the constant responders
(rule in ../panel.py).

Writes results/tables/capability_regression.{csv,tex}.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panel import (LARGE_OPEN_MODELS, MODEL_METADATA, MODEL_SIZE,  # noqa: E402
                   MODEL_SIZE_ACTIVE, bare_model_name, low_variability_models)
from utils import TRAITS, attenuation_correct, load_llm_reference, per_respondent_contrasts  # noqa: E402

TABLE_DIR = Path(__file__).resolve().parents[3] / "results/tables"

# ============================================================
# Outcome: |b_hat| per model (Fig 3 values)
# ============================================================

contrasts = per_respondent_contrasts(load_llm_reference(), "model", "model_answer", "item_id")
attenuation_correct(contrasts)
abs_b = contrasts.assign(abs_b=contrasts["b_corr"].abs()).pivot(
    index="model", columns="trait", values="abs_b")[TRAITS]
abs_b["mean"] = abs_b[TRAITS].mean(axis=1)


def metadata_row(model: str) -> pd.Series:
    """Metadata row of an analysis-frame model key: exact match, else the longest
    panel_key contained in it (API keys carry provider prefix / condition suffix)."""
    keys = MODEL_METADATA["panel_key"]
    hit = keys[keys == model]
    if hit.empty:
        hit = keys[[k in model for k in keys]]
        hit = hit[hit.str.len() == hit.str.len().max()]
    assert len(hit) == 1, (model, list(hit))
    return MODEL_METADATA.loc[hit.index[0]]


meta = pd.DataFrame([metadata_row(m) for m in abs_b.index], index=abs_b.index)
release = pd.to_datetime(meta["release_date"])
df = abs_b.assign(
    release_year=2020 + (release - pd.Timestamp("2020-01-01")).dt.days / 365.25,
    log_total=np.log10(abs_b.index.map(MODEL_SIZE.get).astype(float)),
    log_active=np.log10(abs_b.index.map(MODEL_SIZE_ACTIVE.get).astype(float)),
    proprietary=(~abs_b.index.isin(list(MODEL_SIZE))).astype(int),
    large=abs_b.index.isin(list(LARGE_OPEN_MODELS)),
    eci=pd.to_numeric(meta["eci"], errors="coerce"),
    mmlu_pro=pd.to_numeric(meta["oll_mmlu_pro"], errors="coerce"),
    arena=pd.to_numeric(meta["arena_text_elo"], errors="coerce") / 100,
    constant=abs_b.index.map(bare_model_name).isin(low_variability_models()[0]["constant"]),
)
assert len(df) == 60 and df["release_year"].notna().all()

# ============================================================
# Regressions
# ============================================================

open_w = lambda d: d[d["proprietary"] == 0]
BLOCKS = {"A": "Open-weight models", "B": "All models", "C": "Models with a capability score"}
SPECS = [  # (block, label, rhs, subset): single predictors first, then joint fits
    ("A", "size", "log_total", open_w),
    ("A", "date", "release_year", open_w),
    ("A", "size + date", "log_total + release_year", open_w),
    ("A", "active size", "log_active", open_w),
    ("A", "active size + date", "log_active + release_year", open_w),
    ("A", "size + date, without >=300B", "log_total + release_year",
     lambda d: d[(d["proprietary"] == 0) & ~d["large"]]),
    ("B", "date", "release_year", lambda d: d),
    ("B", "proprietary", "proprietary", lambda d: d),
    ("B", "date + proprietary", "release_year + proprietary", lambda d: d),
    ("C", "ECI", "eci", lambda d: d[d["eci"].notna()]),
    ("C", "MMLU-Pro", "mmlu_pro", lambda d: d[d["mmlu_pro"].notna()]),
    ("C", "Arena Elo", "arena", lambda d: d[d["arena"].notna()]),
]

rows = []
for sample in ["all", "excl. constant"]:
    for block, spec, rhs, subset in SPECS:
        d = subset(df)
        if sample == "excl. constant":
            d = d[~d["constant"]]
        for outcome in ["mean"] + TRAITS:
            fit = smf.ols(f"Q('{outcome}') ~ {rhs}", data=d).fit()
            ci = fit.conf_int()
            for pred in rhs.split(" + "):
                row = {"sample": sample, "block": block, "spec": spec, "outcome": outcome, "predictor": pred,
                       "n": int(fit.nobs),
                       "coef": fit.params[pred], "ci_lo": ci.loc[pred, 0], "ci_hi": ci.loc[pred, 1],
                       "p": fit.pvalues[pred], "r2": fit.rsquared}
                if " + " not in rhs:
                    rho, p_rho = stats.spearmanr(d[pred], d[outcome])
                    row.update(spearman_rho=rho, spearman_p=p_rho)
                rows.append(row)
res = pd.DataFrame(rows)
res.to_csv(TABLE_DIR / "capability_regression.csv", index=False)

# ============================================================
# LaTeX: mean |b_hat| per spec and predictor, with the per-trait p range
# ============================================================

LABEL = {"log_total": "log$_{10}$ total parameters", "log_active": "log$_{10}$ active parameters",
         "release_year": "release date (years)", "proprietary": "proprietary",
         "eci": "ECI", "mmlu_pro": "MMLU-Pro", "arena": "Arena Elo (per 100)"}


def fmt_p(p):
    return "$<$.001" if p < .001 else f"{p:.3f}".lstrip("0")


def num(x, d=3):
    return f"{x:.{d}f}".replace("-", "$-$")   # typeset minus, not hyphen


HEADER = ["Model", "$n$", "Predictor", "$\\beta$ [95\\% CI]", "$p$", "$p$ per trait",
          "Spearman $\\rho$ ($p$)", "$\\beta$ excl.\\ const.", "$p$ excl.\\ const."]
main = res[res["sample"] == "all"]
excl = res[(res["sample"] == "excl. constant") & (res["outcome"] == "mean")].set_index(
    ["block", "spec", "predictor"])
body, prev_block, prev_spec = [], None, None
for (block, spec, pred), g in main.groupby(["block", "spec", "predictor"], sort=False):
    if block != prev_block:
        if prev_block is not None:
            body.append("\\midrule")
        body.append(f"\\multicolumn{{{len(HEADER)}}}{{@{{}}l}}{{\\textit{{{BLOCKS[block]}}}}} \\\\")
    m = g[g["outcome"] == "mean"].iloc[0]
    tr = g[g["outcome"] != "mean"]["p"]
    e = excl.loc[(block, spec, pred)]
    first = (block, spec) != (prev_block, prev_spec)   # model name and n once per fit
    rho = "" if pd.isna(m.get("spearman_rho")) else \
        f"{num(m['spearman_rho'], 2)} ({fmt_p(m['spearman_p'])})"
    body.append(" & ".join([
        spec.replace(">=", "$\\geq$") if first else "", str(m["n"]) if first else "",
        LABEL[pred],
        f"{num(m['coef'])} [{num(m['ci_lo'])}, {num(m['ci_hi'])}]",
        fmt_p(m["p"]),
        f"{fmt_p(tr.min())}--{fmt_p(tr.max())}",
        rho,
        num(e["coef"]),
        fmt_p(e["p"]),
    ]) + " \\\\")
    prev_block, prev_spec = block, spec

r_size_date = stats.pearsonr(*df.loc[df["proprietary"] == 0, ["log_total", "release_year"]].T.values)[0]
caption = (
    "Regressions of absolute response bias on model size, release date and capability "
    "scores (OLS). Outcome: $|\\hat b|$ averaged over the Big Five (IPIP-NEO-300, "
    "item-noise corrected as in main-text Fig.~3); $p$ per trait gives the range of the "
    "predictor's $p$-value over the five single-trait fits. $\\beta$ is the change in "
    "$|\\hat b|$ per tenfold increase in parameters, per year of release, per score point "
    "(Arena Elo: per 100 points), or for proprietary vs open-weight models. Size is public "
    "for open-weight models only (nominal counts; active equals total parameters for "
    "dense models). Release date: first public release of the weights administered "
    "(proprietary models: API availability). Capability scores: ECI, Epoch Capabilities "
    "Index (downloaded 22 Oct 2026); MMLU-Pro, Open LLM Leaderboard v2 (normalised), "
    "available for small open-weight models only; Arena Elo, LMArena text leaderboard "
    "(style-controlled, snapshot of 13 Oct 2026). Each score regression uses only the models that have that "
    "score. Spearman $\\rho$ is given for single-predictor fits. Last two columns: the "
    "same fit without the constant responders. Among open-weight models, log size "
    f"and release date correlate $r$ = {num(r_size_date, 2)}."
)
with open(TABLE_DIR / "capability_regression.tex", "w", encoding="utf-8") as f:
    f.write("\\begin{table}\n\\caption{" + caption + "}\n\\label{tab:capability_regression}\n"
            "\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
            "\\begin{tabular}{@{}llllccccc@{}}\n\\toprule\n"
            + " & ".join(HEADER) + " \\\\\n\\midrule\n"
            + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

pd.set_option("display.width", 250)
print(res[res["outcome"] == "mean"].round(4).to_string(index=False))
print()
print(res.groupby(["sample", "spec", "predictor"], sort=False)["p"].agg(["min", "max"]).round(3).to_string())
