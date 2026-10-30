"""SI: response bias against model size for all 60 models, with estimated sizes for
the proprietary models (IPIP-NEO-300, reference condition).

Outcome: |b_hat| per model and Big Five trait, item-noise corrected as in Fig 3,
and its mean over the five traits. Size: log10 total parameters; nominal counts
for the open-weight models. Proprietary sizes are not disclosed; ../model_metadata.csv
holds them in columns params_est_*_B (sources there): developer statements for
Grok 4.20 and Qwen3.5-Plus and secondary sources for Qwen3.5-Flash; for the other
seven, the estimates of three LLMs asked the same question (ChatGPT, Claude, Gemini)
and their arithmetic mean (primary).

  A. primary set, per trait: OLS of |b_hat| on size (plus Pearson r and Spearman
     rho), and in two separate fits on size + release date, size + proprietary
  B. robustness: Pearson r per trait with each LLM's estimates alone, and with every
     proprietary size of the primary set multiplied by a random factor between 1/3
     and 3 (log-uniform, N_DRAWS draws)

Writes results/tables/size_with_proprietary_estimates.{csv,tex}; the .tex holds A only,
B (quoted in the SI text) is in the CSV.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panel import MODEL_METADATA, MODEL_SIZE  # noqa: E402
from utils import TRAITS, attenuation_correct, load_llm_reference, per_respondent_contrasts  # noqa: E402

TABLE_DIR = Path(__file__).resolve().parents[3] / "results/tables"
OUTCOMES = TRAITS + ["mean"]
SIZE_SETS = {  # label -> metadata column
    "Mean (A)": "params_est_B",
    "ChatGPT": "params_est_chatgpt_B",
    "Claude": "params_est_claude_B",
    "Gemini": "params_est_gemini_B",
}
PRIMARY = "Mean (A)"
ESTIMATED = {  # panel_key -> name in the caption, for the seven sizes estimated by LLMs
    "claude-haiku-4-5": "Claude Haiku 4.5", "claude-sonnet-4-6": "Sonnet 4.6",
    "claude-opus-4-6": "Opus 4.6", "gpt-5.4": "GPT-5.4", "gpt-5.4-mini": "mini",
    "gpt-5.4-nano": "nano", "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash-Lite",
}
N_DRAWS = 2000
JITTER = 3          # random factor between 1/JITTER and JITTER

# ============================================================
# Data: |b_hat| per model (Fig 3 values), sizes and release date
# ============================================================

contrasts = per_respondent_contrasts(load_llm_reference(), "model", "model_answer", "item_id")
attenuation_correct(contrasts)
abs_b = contrasts.assign(abs_b=contrasts["b_corr"].abs()).pivot(
    index="model", columns="trait", values="abs_b")[TRAITS]
abs_b["mean"] = abs_b[TRAITS].mean(axis=1)

meta_by_key = MODEL_METADATA.set_index("panel_key")


def metadata_row(model: str) -> pd.Series:
    """Metadata row of an analysis-frame model key: exact match, else the longest
    panel_key contained in it (API keys carry provider prefix / condition suffix,
    and gpt-5.4 is contained in gpt-5.4-mini)."""
    if model in meta_by_key.index:
        return meta_by_key.loc[model]
    hits = [k for k in meta_by_key.index if k in model]
    assert hits, model
    return meta_by_key.loc[max(hits, key=len)]


rows = []
for m in abs_b.index:
    md = metadata_row(m)
    proprietary = int(m not in MODEL_SIZE)
    row = {"model": m, "proprietary": proprietary, "release": pd.Timestamp(md["release_date"])}
    for label, col in SIZE_SETS.items():
        size = md[col] if proprietary else MODEL_SIZE[m]
        row[label] = np.log10(float(size))
    rows.append(row)
df = abs_b.join(pd.DataFrame(rows).set_index("model"))
df["log_size"] = df[PRIMARY]
df["release_year"] = 2020 + (df["release"] - pd.Timestamp("2020-01-01")).dt.days / 365.25
assert len(df) == 60 and df[list(SIZE_SETS) + ["release_year"]].notna().all().all()

# ============================================================
# A. Primary set: regressions per trait
# ============================================================

out = []
for y in OUTCOMES:
    row = {"outcome": y, "n": len(df)}
    row["r"], row["r_p"] = stats.pearsonr(df["log_size"], df[y])
    row["spearman_rho"], row["spearman_p"] = stats.spearmanr(df["log_size"], df[y])
    for tag, rhs, cov in [("", "log_size", None),
                          ("_wdate", "log_size + release_year", "release_year"),
                          ("_wprop", "log_size + proprietary", "proprietary")]:
        fit = smf.ols(f"Q('{y}') ~ {rhs}", data=df).fit()
        ci = fit.conf_int().loc["log_size"]
        row.update({f"slope{tag}": fit.params["log_size"], f"ci_lo{tag}": ci[0],
                    f"ci_hi{tag}": ci[1], f"p{tag}": fit.pvalues["log_size"]})
        if cov:
            row.update({f"coef_{cov}": fit.params[cov], f"p_{cov}": fit.pvalues[cov]})
    out.append(row)
res = pd.DataFrame(out)

# ============================================================
# B. Robustness: alternative size sets and random perturbation
# ============================================================

robust = []
for label in SIZE_SETS:
    for y in OUTCOMES:
        r, p = stats.pearsonr(df[label], df[y])
        robust.append({"size_set": label, "outcome": y, "r": r, "p": p})

rng = np.random.default_rng(1)
prop = df["proprietary"].to_numpy() == 1
draws = {y: [] for y in OUTCOMES}
for _ in range(N_DRAWS):
    x = df["log_size"].to_numpy().copy()
    x[prop] += rng.uniform(-np.log10(JITTER), np.log10(JITTER), prop.sum())
    for y in OUTCOMES:
        draws[y].append(stats.pearsonr(x, df[y])[:2])
for y in OUTCOMES:
    r, p = np.array(draws[y]).T
    lo, hi = np.percentile(r, [2.5, 97.5])
    robust.append({"size_set": f"random x1/{JITTER}..x{JITTER}", "outcome": y, "r": np.median(r),
                   "r_lo": lo, "r_hi": hi, "share_p_below_05": (p < .05).mean()})
robust = pd.DataFrame(robust)

pd.concat([res.assign(block="A"), robust.assign(block="B")], ignore_index=True).to_csv(
    TABLE_DIR / "size_with_proprietary_estimates.csv", index=False)

# ============================================================
# LaTeX
# ============================================================


def fmt_p(p):
    return "$<$.001" if p < .001 else f"{p:.3f}".lstrip("0")


def num(x, d=2):
    s = f"{x:.{d}f}".removeprefix("-" if round(x, d) == 0 else "")   # no negative zero
    return s.replace("-", "$-$")   # typeset minus, not hyphen


def slope(r, tag=""):
    return f"{num(r['slope' + tag])} [{num(r['ci_lo' + tag])}, {num(r['ci_hi' + tag])}]"


TRAIT_NAME = {"O": "O", "C": "C", "E": "E", "A": "A", "N": "N", "mean": "Mean"}
body_a = []
for _, r in res.iterrows():
    if r["outcome"] == "mean":
        body_a.append("\\midrule")
    body_a.append(" & ".join([
        TRAIT_NAME[r["outcome"]], num(r["r"]), slope(r), fmt_p(r["p"]),
        f"{num(r['spearman_rho'])} ({fmt_p(r['spearman_p'])})",
        slope(r, "_wdate"), fmt_p(r["p_wdate"]),
        slope(r, "_wprop"), fmt_p(r["p_wprop"]),
    ]) + " \\\\")

def billions(x):
    return f"{x / 1000:.1f}T" if x >= 1000 else f"{x:.0f}B"


est_sizes = "; ".join(f"{name} {billions(meta_by_key.loc[k, 'params_est_B'])}"
                      for k, name in ESTIMATED.items())
n_prop = int(df["proprietary"].sum())
caption = (
    "Absolute response bias and model size across all LLMs, with estimated sizes for the "
    f"proprietary models ($n$ = {len(df)}: {len(df) - n_prop} open-weight, {n_prop} "
    "proprietary). Outcome: $|\\hat b|$ per Big Five trait (IPIP-NEO-300, item-noise "
    "corrected as in main-text Fig.~3) and its mean over the five traits. Size: "
    "$\\log_{10}$ total parameters, nominal for open-weight models. Proprietary developers "
    "do not disclose parameter counts. Grok 4.20 (500B) and Qwen3.5-Plus (397B, the "
    "hosted version of Qwen3.5-397B-A17B) are sized from developer statements, "
    "Qwen3.5-Flash (35B, hosted Qwen3.5-35B-A3B) from secondary sources. The other "
    "seven were estimated by asking three LLMs (ChatGPT, Claude, Gemini) the same "
    "question in fresh chats and averaged (see text): "
    f"{est_sizes}. $b$: unstandardized OLS "
    "change in $|\\hat b|$ per tenfold increase in parameters, with 95\\% CI, alone and "
    "controlling for release date or for a proprietary indicator (two separate "
    "regressions). All size estimates for proprietary models are uncertain."
)
HEADER_A = ["Trait", "$r$", "$b$ [95\\% CI]", "$p$", "Spearman $\\rho$ ($p$)",
            "$b$ + date", "$p$", "$b$ + proprietary", "$p$"]
with open(TABLE_DIR / "size_with_proprietary_estimates.tex", "w", encoding="utf-8") as f:
    f.write(
        "\\begin{table}\n\\caption{" + caption + "}\n\\label{tab:size_with_proprietary_estimates}\n"
        "\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{@{}lcccccccc@{}}\n\\toprule\n"
        + " & ".join(HEADER_A) + " \\\\\n\\midrule\n"
        + "\n".join(body_a) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

pd.set_option("display.width", 250)
print(res.round(3).to_string(index=False))
print()
print(robust.round(3).to_string(index=False))
