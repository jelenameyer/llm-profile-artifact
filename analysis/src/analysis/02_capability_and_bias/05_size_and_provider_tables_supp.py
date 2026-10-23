"""SI tables for the two Fig. 3 bottom-row contrasts, read from
results/tables/b_per_trait_stats.csv (written by 01_decomposition_per_llm.py):

  size_correlation.tex     per-trait Pearson r of |b_hat| with log10(total parameters)
                           and the OLS slope per decade, for all open-weight models
                           and for the enumerated 1-70B stratum alone
  open_vs_proprietary.tex  per-trait median |b_hat| of open-weight and proprietary
                           models with the Mann-Whitney p between them, and the human
                           median as reference

All |b_hat| values are item-noise corrected, as in Fig. 3.
"""
from pathlib import Path

import pandas as pd

TABLE_DIR = Path(__file__).resolve().parents[3] / "results/tables"
stats = pd.read_csv(TABLE_DIR / "b_per_trait_stats.csv")


def fmt_p(p):
    return "$<$.001" if p < .001 else f"{p:.3f}".lstrip("0")


def fmt(x):
    return f"${x:.2f}$"


n_all, n_small = int(stats["size_n"].iloc[0]), int(stats["size_n_excl_large"].iloc[0])
size = pd.DataFrame({
    "Trait": stats["trait"],
    f"$r$ ($n$ = {n_all})": stats["size_r"].map(fmt),
    "slope": stats["size_slope"].map(fmt),
    "$p$": stats["size_p"].map(fmt_p),
    f"$r$ ($n$ = {n_small})": stats["size_r_excl_large"].map(fmt),
    "slope ": stats["size_slope_excl_large"].map(fmt),
    "$p$ ": stats["size_p_excl_large"].map(fmt_p),
})
size_caption = (
    "Absolute response bias and model size. Per-trait Pearson $r$ between $|\\hat b|$ "
    "(item-noise corrected) and $\\log_{10}$ total parameters, OLS slope (change in "
    "$|\\hat b|$ per tenfold increase in parameters) and $p$, for all open-weight models "
    f"($n$ = {n_all}, 1B--1T) and for those of 1--70B ($n$ = {n_small})."
)
with open(TABLE_DIR / "size_correlation.tex", "w", encoding="utf-8") as f:
    f.write(size.to_latex(index=False, escape=False, column_format="lcccccc", position="h!",
                          caption=size_caption, label="tab:a_size_correlation"))

n_open, n_prop = int(stats["n_open"].iloc[0]), int(stats["n_closed"].iloc[0])
prov = pd.DataFrame({
    "Trait": stats["trait"],
    f"open-weight ($n$ = {n_open})": stats["median_abs_b_open"].map(fmt),
    f"proprietary ($n$ = {n_prop})": stats["median_abs_b_closed"].map(fmt),
    "MW $p$": stats["mw_p_open_vs_prop"].map(fmt_p),
    f"human ($n$ = {int(stats['n_human'].iloc[0]):,})".replace(",", "{,}"): stats["median_abs_b_human_corr"].map(fmt),
})
prov_caption = (
    "Median absolute response bias $|\\hat b|$ per trait for open-weight LLMs, proprietary "
    "LLMs and humans, all item-noise corrected as in main-text Fig.~3. MW $p$: two-sided "
    "Mann--Whitney $U$ test, open-weight vs proprietary, uncorrected; humans are shown "
    "as reference."
)
with open(TABLE_DIR / "open_vs_proprietary.tex", "w", encoding="utf-8") as f:
    f.write(prov.to_latex(index=False, escape=False, column_format="lcccc", position="h!",
                          caption=prov_caption, label="tab:a_open_vs_prop"))

print(size.to_string(index=False))
print()
print(prov.to_string(index=False))
