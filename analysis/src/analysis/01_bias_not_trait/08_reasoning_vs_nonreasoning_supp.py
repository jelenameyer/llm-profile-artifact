"""SI table: reasoning (_r) vs non-reasoning (_nr) runs of the same model on the
IPIP-NEO-300, for the proprietary models and the large open-weight models.
Per model: items answered, temperature in _r, % identical answers, mean |b_hat|
over the Big Five in each run, and number of traits whose bias sign flips.
Per trait: paired Wilcoxon test of |b_hat| (_nr vs _r), for all models and for
the large open-weight models alone.
Writes results/tables/reasoning_vs_nonreasoning.{csv,tex} and
reasoning_vs_nonreasoning_tests.csv."""
import numpy as np
import pandas as pd
from scipy import stats
from utils import clean_model_label, is_proprietary

TRAITS = ["O", "C", "E", "A", "N"]
M = 3  # scale midpoint
OUT = "../../../results/tables/"

api = pd.read_csv("../../../data/ipipneo300_data/api_data/ipipneo_api_data_raw.csv", low_memory=False)
api = api[api["context_mode"] == "no_context"].copy()
api["cond"] = np.where(api["model"].str.endswith("__nr"), "nr", "r")
# pair the two runs of a model; Grok exposes them as two model IDs
api["base"] = (api["model"].str.replace(r"__n?r$", "", regex=True)
               .str.replace(r"-0309-(non-)?reasoning$", "", regex=True))
api["group"] = np.where(api["model"].map(is_proprietary), "Proprietary", "Large open-weight")

# b_hat per (model, run, trait): (forward mean + reverse mean) / 2 - m, raw answers
means = api.groupby(["base", "cond", "traits", "reverse_coded"])["model_answer"].mean().unstack()
b = ((means[False] + means[True]) / 2 - M).unstack("cond")  # index (base, trait), cols nr / r

# % identical answers, compared numerically over items answered in both runs
wide = api.pivot_table(index=["base", "item_id"], columns="cond", values="model_answer", aggfunc="first")
both = wide.dropna()
identical = (both["nr"] == both["r"]).groupby("base").mean()

rows = []
for base, g in api.groupby("base", sort=False):
    r_run = g[g["cond"] == "r"]
    temps = sorted(r_run["param_temperature"].dropna().unique())
    bb = b.loc[base]
    rows.append({
        "group": g["group"].iloc[0],
        "model": clean_model_label(base),
        "n_nr": int(g.loc[g["cond"] == "nr", "model_answer"].notna().sum()),
        "n_r": int(r_run["model_answer"].notna().sum()),
        "temp_r": f"{temps[0]:g}" if len(temps) == 1 else "n/a",
        "pct_identical": 100 * identical[base],
        "mean_abs_b_nr": bb["nr"].abs().mean(),
        "mean_abs_b_r": bb["r"].abs().mean(),
        "delta_mean_abs_b": bb["r"].abs().mean() - bb["nr"].abs().mean(),
        "n_sign_flips": int((np.sign(bb["nr"]) != np.sign(bb["r"])).sum()),
        **{f"b_nr_{t}": bb.loc[t, "nr"] for t in TRAITS},
        **{f"b_r_{t}": bb.loc[t, "r"] for t in TRAITS},
    })
tab = (pd.DataFrame(rows).sort_values(["group", "model"], ascending=[False, True])
       .reset_index(drop=True))
tab.to_csv(OUT + "reasoning_vs_nonreasoning.csv", index=False)

# paired tests of |b_hat| per trait
group_of = api.groupby("base")["group"].first()
tests = []
for sample, bases in [("all", group_of.index),
                      ("large_open_weight", group_of[group_of == "Large open-weight"].index)]:
    for t in TRAITS:
        s = b.xs(t, level="traits").loc[bases].abs()
        tests.append({
            "sample": sample, "trait": t, "n_models": len(s),
            "mean_abs_b_nr": s["nr"].mean(), "mean_abs_b_r": s["r"].mean(),
            "n_lower_with_reasoning": int((s["r"] < s["nr"]).sum()),
            "wilcoxon_p": stats.wilcoxon(s["nr"], s["r"]).pvalue,
        })
tests = pd.DataFrame(tests)
tests.to_csv(OUT + "reasoning_vs_nonreasoning_tests.csv", index=False)

# LaTeX: one row per model under a group label row; trait-level tests in the caption
GROUP_LABEL = {
    "Proprietary": "Proprietary (\\_r at temperature 1 or provider default)",
    "Large open-weight": "Large open-weight (temperature 0 in both runs)",
}
incomplete = tab[tab["n_r"] < tab["n_nr"]]

def fmt_delta(x):
    return f"{x:+.2f}".replace("-", "$-$")

def p_range(s):
    return f"{s.min():.2f}--{s.max():.2f}".replace("0.", ".")

all_t = tests[tests["sample"] == "all"]
low_t = tests[tests["sample"] == "large_open_weight"]
caption = (
    "Reasoning (\\_r) vs non-reasoning (\\_nr) runs of the same model on the IPIP-NEO-300. "
    "$|\\hat b|$ is averaged over the Big Five; $\\Delta = |\\hat b|_r - |\\hat b|_{nr}$; "
    "sign flips count traits whose $\\hat b$ changes sign. "
    f"Paired Wilcoxon tests of $|\\hat b|$ per trait: all {all_t['n_models'].iloc[0]} models "
    f"$p$ = {p_range(all_t['wilcoxon_p'])}; large open-weight models only "
    f"$p$ = {p_range(low_t['wilcoxon_p'])} ($n$ = {low_t['n_models'].iloc[0]} cannot reach "
    "$p < .125$). Grok's two runs are distinct model IDs. "
    "$^\\dagger$ \\_r items with an extractable answer: "
    + "; ".join(f"{r.model.replace('_', chr(92) + '_')} {r.n_r} of {r.n_nr}"
                for r in incomplete.itertuples())
    + "."
)
lines = [
    "\\begin{table}", f"\\caption{{{caption}}}", "\\label{tab:reasoning_vs_nonreasoning}",
    "\\begin{tabular}{lccccc}", "\\toprule",
    "Model & $|\\hat b|$ \\_nr & $|\\hat b|$ \\_r & $\\Delta$ & Identical (\\%) "
    "& Sign flips \\\\",
]
for group, g in tab.groupby("group", sort=False):
    lines += ["\\midrule", f"\\multicolumn{{6}}{{l}}{{\\textit{{{GROUP_LABEL[group]}}}}} \\\\"]
    for _, r in g.iterrows():
        mark = "$^\\dagger$" if r["n_r"] < r["n_nr"] else ""
        lines.append(
            f"{r['model'].replace('_', chr(92) + '_')}{mark} & {r['mean_abs_b_nr']:.2f} & "
            f"{r['mean_abs_b_r']:.2f} & {fmt_delta(r['delta_mean_abs_b'])} & "
            f"{r['pct_identical']:.0f} & {r['n_sign_flips']} \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
with open(OUT + "reasoning_vs_nonreasoning.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

pd.set_option("display.width", 250)
print(tab[["group", "model", "n_nr", "n_r", "temp_r", "pct_identical",
           "mean_abs_b_nr", "mean_abs_b_r", "n_sign_flips"]].round(3).to_string(index=False))
print()
print(tests.round(3).to_string(index=False))
