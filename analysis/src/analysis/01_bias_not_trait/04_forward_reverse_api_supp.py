"""SI Fig S2: forward-reverse item-mean scatter for the proprietary API models,
reasoning vs non-reasoning sub-conditions, per Big Five trait (IPIP-NEO-300).
Also prints the per-trait forward-reverse correlations cited in the SI text.
Writes results/figures/forward_reverse_api_supp.pdf."""
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_corrs_and_ci, plot_corr_panel

TRAITS = ["O", "C", "E", "A", "N"]

# load API data; split reasoning (__r) vs non-reasoning (__nr) sub-conditions
api_data = pd.read_csv(
    "../../../data/ipipneo300_data/api_data/ipipneo_api_data_raw.csv",
    low_memory=False,
)
api_nr = api_data[api_data["model"].str.contains("_nr", regex=False)]
api_r = api_data[api_data["model"].str.contains("_r", regex=False)]

# per-trait forward-reverse correlations (the numbers quoted in the SI text)
rows = []
for condition_name, df in [("non-reasoning", api_nr), ("reasoning", api_r)]:
    for trait in TRAITS:
        subset = df[df["traits"] == trait]
        r, lo, hi, _ = get_corrs_and_ci(subset, entity_var="model", dependent_var="model_answer")
        rows.append({
            "condition": condition_name,
            "trait": trait,
            "r": round(r, 3),
            "ci_lo": round(lo, 3),
            "ci_hi": round(hi, 3),
            "n_models": subset["model"].nunique(),
        })
print(pd.DataFrame(rows).to_string(index=False))

# scatter: non-reasoning (top row) vs reasoning (bottom row), one column per trait
fig, axes = plt.subplots(2, 5, figsize=(16, 7), sharey=True, sharex=True)
for j, trait in enumerate(TRAITS):
    for i, (condition_name, df, color) in enumerate([
        ("Non-reasoning", api_nr, "#1e847f"),
        ("Reasoning", api_r, "#3b528b"),
    ]):
        subset = df[df["traits"] == trait]
        ax = axes[i, j]
        if subset["model"].nunique() < 2:
            ax.set_visible(False)
            continue
        plot_corr_panel(ax, subset, entity_var="model", dependent_var="model_answer",
                        color=color, FS_STATS=8, show_stats=True)
        if j == 0:
            ax.set_ylabel(f"{condition_name}\nMean reversed", fontsize=9)
        if i == 0:
            ax.set_title(trait, fontsize=11, fontweight="bold")
        if i == 1:
            ax.set_xlabel("Mean forward", fontsize=9)

plt.tight_layout()
plt.savefig("../../../results/figures/forward_reverse_api_supp.pdf", bbox_inches="tight")
