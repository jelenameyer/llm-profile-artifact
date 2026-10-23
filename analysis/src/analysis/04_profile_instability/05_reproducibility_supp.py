"""SI: reproducibility across two identical administrations of the large
open-weight models (pass 1 and rep2, same requests, temperature 0), per reasoning
condition, over IPIP-NEO-300 + the 12 risk instruments (598 items per pass).

Per model x condition: items answered in each pass, % identical answers among
items answered in both, and on the IPIP-NEO-300 the mean |b_hat| over the Big Five
in each pass plus the largest per-trait change in b_hat between passes, both on the
items answered in both passes (so a change reflects changed answers, not a changed
item set). Per task:
% identical answers (reproducibility tracks response-scale granularity).

Reads data/reproducibility/rep_pairs.csv (src/preprocessing/08_preprocess_reproducibility.py).
Writes results/tables/reproducibility.{csv,tex} and reproducibility_by_task.csv.
"""
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from panel import bare_model_name  # noqa: E402

DATA = HERE.parents[2] / "data/reproducibility/rep_pairs.csv"
TABLE_DIR = HERE.parents[2] / "results/tables"
M = 3   # IPIP-NEO-300 scale midpoint


def label(name: str) -> str:
    # NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 -> Nemotron-3-Ultra
    return re.sub(r"^NVIDIA-(.*)-\d+B-A\d+B-BF16$", r"\1", bare_model_name(name))


pairs = pd.read_csv(DATA, low_memory=False)
pairs["model"] = pairs["model"].map(label)
both = pairs.dropna(subset=["answer_1", "answer_2"])
# compare numerically (a missing value elsewhere in a column makes it float)
both = both.assign(same=both["answer_1"].astype(float) == both["answer_2"].astype(float))

# b_hat per (model, condition, pass, trait) on the IPIP-NEO-300, items answered in both passes
ipip = both[both["task"] == "ipipneo"]
b = {}
for p in ("1", "2"):
    means = ipip.groupby(["model", "condition", "traits", "reverse_coded"])[f"answer_{p}"].mean().unstack()
    b[p] = (means[True] + means[False]) / 2 - M
b = pd.DataFrame(b)  # index (model, condition, trait)

keys = ["model", "condition"]
res = pd.DataFrame({
    "items": pairs.groupby(keys).size(),
    "answered_1": pairs.groupby(keys)["answer_1"].count(),
    "answered_2": pairs.groupby(keys)["answer_2"].count(),
    "comparable": both.groupby(keys).size(),
    "identical_pct": 100 * both.groupby(keys)["same"].mean(),
    "mean_abs_b_1": b["1"].abs().groupby(level=keys).mean(),
    "mean_abs_b_2": b["2"].abs().groupby(level=keys).mean(),
    "max_abs_diff_b": (b["2"] - b["1"]).abs().groupby(level=keys).max(),
}).reset_index()
res.to_csv(TABLE_DIR / "reproducibility.csv", index=False)

by_task = (100 * both.groupby(["task", "condition", "model"])["same"].mean()).unstack(["condition", "model"])
by_task.to_csv(TABLE_DIR / "reproducibility_by_task.csv")

# Compact LaTeX: one row per model, the two conditions side by side
wide = res.set_index(keys).unstack("condition")
tex = pd.DataFrame({
    "Model": wide.index,
    r"Identical \_nr (\%)": wide[("identical_pct", "nr")].map("{:.1f}".format).values,
    r"Identical \_r (\%)": wide[("identical_pct", "r")].map("{:.1f}".format).values,
    r"Items compared \_r": wide[("comparable", "r")].astype(int).values,
    r"max $\Delta\hat b$ \_nr": wide[("max_abs_diff_b", "nr")].map("{:.2f}".format).values,
    r"max $\Delta\hat b$ \_r": wide[("max_abs_diff_b", "r")].map("{:.2f}".format).values,
})
caption = (
    "Reproducibility across two identical administrations (temperature 0, same "
    f"requests) of the large open-weight models, {int(res['items'].iloc[0])} items per "
    "condition (IPIP-NEO-300 and 12 risk instruments). Identical: share of items with "
    "the same answer in both passes, among items answered in both. "
    r"max $\Delta\hat b$: largest change in $\hat b$ over the Big Five between passes, "
    "computed on the items answered in both; in the reasoning condition up to "
    f"{res.loc[res['condition'] == 'r', 'max_abs_diff_b'].max():.2f}."
)
with open(TABLE_DIR / "reproducibility.tex", "w", encoding="utf-8") as f:
    f.write(tex.to_latex(index=False, escape=False, column_format="lccccc",
                         caption=caption, label="tab:reproducibility"))

pd.set_option("display.width", 250)
print(res.round(3).to_string(index=False))
print()
print(by_task.round(0).to_string())
