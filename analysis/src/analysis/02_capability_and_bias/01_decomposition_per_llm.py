"""Figure 2: per-trait (theta_hat - m, b_hat) decomposition (row 0) and |b_hat|
vs model size + open/proprietary/human reference (row 1), for the Big Five.

Reads ipipneo300_data (llm/api/human, raw + rereversed, no-context).
Writes:
  results/figures/decomposition_per_llm.pdf
  results/tables/b_per_trait_stats.csv
"""
import warnings
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# Constants
# ============================================================

TRAITS = ["O", "C", "E", "A", "N"]
TRAIT_NAMES = {
    "O": "Openness",
    "C": "Conscientiousness",
    "E": "Extraversion",
    "A": "Agreeableness",
    "N": "Neuroticism",
}
SCALE_MIN, SCALE_MAX = 1, 5
M = (SCALE_MIN + SCALE_MAX) / 2

C_HUMAN = "#440154"
C_LLM   = "#1e847f"
C_LLM_API = "#287c8e" 

# Per-origin point style: proprietary models use the blue fill, semi-transparent and sit on top
LLM_POINT_STYLE = [
    ("open",   dict(c=C_LLM,     alpha=1.0, zorder=4)),
    ("closed", dict(c=C_LLM_API, alpha=0.7, zorder=5)),
]

_RGB_HUMAN = mcolors.to_rgb(C_HUMAN)
CMAP_HUMAN = mcolors.LinearSegmentedColormap.from_list(
    "purple_fade",
    [(*_RGB_HUMAN, 0.05), (*_RGB_HUMAN, 0.70)],
    N=256,
)
KDE_GRID = 140
KDE_LEVELS = 7

# Typography — matched to 01_forward_reverse_overview.py
FS_LABEL     = 15   # axis labels
FS_TITLE     = 17   # subplot titles
FS_STATS     = 13    # in-plot stats boxes / annotations
FS_TICK      = 13    # tick labels
FS_TRAITHEAD = 18   # bold trait letter column header

LLM_SIZE_TOP   = 38      # decomposition scatter
LLM_SIZE_BOT   = 38      # |b̂| scatter (bottom panels are narrower)
AX_LIM         = 2.15
GRID_ALPHA     = 0.15

FIGURE_NAME = "decomposition_per_llm.pdf"

LLM_RAW_PATH    = "../../../data/intermediate/ipipneo300_data/llm_data/ipipneo_no_flip_data_raw.csv"
API_RAW_PATH    = "../../../data/intermediate/ipipneo300_data/api_data/ipipneo_api_data_raw.csv"
HUMAN_RAW_PATH  = "../../../data/intermediate/ipipneo300_data/human_data/ipipneo_human_raw.csv"

OUT_FIG_DIR   = Path("../../../results/figures")
OUT_TABLE_DIR = Path("../../../results/tables")
for d in [OUT_FIG_DIR, OUT_TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Open-source model parameter counts (B)
# ============================================================

MODEL_SIZE = {
    'Qwen2.5-1.5B-Instruct': 1.5, 'OLMo-2-7B-Instruct': 7, 'gemma-3-12b-it': 12,
    'Qwen2.5-7B-Instruct': 7, 'Llama-3.1-70B-Instruct': 70, 'gemma-3-27b-it': 27,
    'gpt-oss-20b': 20, 'SmolLM3-3B': 3, 'gemma-3-1b-it': 1, 'Qwen3-32B': 32,
    'Mistral-Small-24B-Instruct-2501': 24, 'LFM2-8B-A1B': 8,
    'Llama-3.1-8B-Instruct': 8, 'gemma-2-27b-it': 27, 'Llama-3.2-1B-Instruct': 1,
    'Apertus-70B-Instruct-2509': 70, 'Ministral-8B-Instruct-2410': 8,
    'Apertus-8B-Instruct-2509': 8, 'Qwen3-4B': 4, 'granite-3.3-2b-instruct': 2,
    'Qwen2.5-32B-Instruct': 32, 'Qwen3-8B': 8, 'LFM2-2.6B': 2.6, 'zephyr-7b-beta': 7,
    'gemma-2-9b-it': 9, 'Qwen3-30B-A3B-Instruct-2507': 30,
    'Qwen2.5-14B-Instruct': 14, 'Mistral-7B-Instruct-v0.3': 7,
    'gemma-2-2b-it': 2, 'gemma-3-4b-it': 4, 'Falcon-3-1B-Instruct': 1,
    'Qwen3-1.7B': 1.7, 'Llama-3.3-70B-Instruct': 70, 'Falcon-3-7B-Instruct': 7,
    'Qwen2.5-3B-Instruct': 3, 'granite-3.3-8b-instruct': 8, 'LFM2-1.2B': 1.2,
    'Llama-3.2-3B-Instruct': 3, 'Falcon-3-10B-Instruct': 10, 'Qwen3-14B': 14,
    'bloomz-3b': 3, 'bloomz-7b1': 7,
    'Phi-3.5-mini-instruct': 4, 'Phi-3-mini-128k-instruct': 4,
    'Phi-3-medium-128k-instruct': 14, 'TildeOpen-30b': 30,
}

def model_origin(name):
    return "open" if name in MODEL_SIZE else "closed"

# ============================================================
# Data loading
# ============================================================

raw_llms  = pd.read_csv(LLM_RAW_PATH, low_memory=False)
raw_api   = pd.read_csv(API_RAW_PATH, low_memory=False)
api_nr    = raw_api[raw_api["model"].str.contains("_nr", regex=False)]
raw_data  = pd.concat([raw_llms, api_nr], ignore_index=True)
raw_data  = raw_data[raw_data["context_mode"] == "no_context"].copy()

raw_human = pd.read_csv(HUMAN_RAW_PATH, low_memory=False)
raw_human["response"] = pd.to_numeric(
    raw_human["response"].replace({"0": None, 0: None, " ": None}), errors="coerce"
)
raw_human = raw_human.dropna(subset=["response"])
raw_human = raw_human[raw_human["response"].between(SCALE_MIN, SCALE_MAX)].copy()

n_humans = raw_human["person_id"].nunique()
n_llms   = raw_data["model"].nunique()
print(f"Loaded: humans = {n_humans:,},  LLMs = {n_llms}")

# ============================================================
# Helpers: block means and (θ̂, b̂) contrasts
# ============================================================

def _block_means(df, group_col, val_col, trait):
    sub = df[df["traits"] == trait]
    Rf  = sub[~sub["reverse_coded"]].groupby(group_col)[val_col].mean()
    Rr  = sub[ sub["reverse_coded"]].groupby(group_col)[val_col].mean()
    idx = Rf.index.intersection(Rr.index)
    return Rf.loc[idx], Rr.loc[idx]

def per_respondent_contrasts(raw, group_col, val_col):
    out = []
    for t in TRAITS:
        Rf, Rr = _block_means(raw, group_col, val_col, t)
        out.append(pd.DataFrame({
            group_col:   Rf.index,
            "trait":     t,
            "b_hat":     (Rf.values + Rr.values) / 2 - M,
            "theta_hat": M + (Rf.values - Rr.values) / 2,
        }))
    return pd.concat(out, ignore_index=True)

llm_contrasts   = per_respondent_contrasts(raw_data,  "model",     "model_answer")
human_contrasts = per_respondent_contrasts(raw_human, "person_id", "response")

llm_contrasts["origin"] = llm_contrasts["model"].map(model_origin)
llm_contrasts["size"]   = llm_contrasts["model"].map(MODEL_SIZE.get)

# ============================================================
# π_b per trait  (Eq. 4: π_b = (1 + ρ) / 2)
# ============================================================

def pi_b_from_data(raw, group_col, val_col, trait):
    Rf, Rr = _block_means(raw, group_col, val_col, trait)
    return (1 + float(np.corrcoef(Rf, Rr)[0, 1])) / 2

pi_b = {
    t: {
        "human": pi_b_from_data(raw_human, "person_id", "response",     t),
        "llm":   pi_b_from_data(raw_data,  "model",     "model_answer", t),
    }
    for t in TRAITS
}

abs_b_max = max(
    float(np.nanpercentile(np.abs(human_contrasts["b_hat"]), 99.5)),
    float(np.nanmax(np.abs(llm_contrasts["b_hat"]))),
) * 1.05

# ============================================================
# Figure layout: 2 rows × 5 columns; Row 1 cells subdivided [2.5, 1]
# ============================================================

fig = plt.figure(figsize=(18, 9))

gs = fig.add_gridspec(
    2, 5,
    height_ratios=[1.0, 1.0],
    hspace=0.20,
    wspace=0.18,
    left=0.06, right=0.995, top=0.93, bottom=0.15,
)

trait_stats = []

for i, t in enumerate(TRAITS):

    sub_h     = human_contrasts[human_contrasts["trait"] == t]
    sub_l     = llm_contrasts[llm_contrasts["trait"] == t]
    sub_h_abs = np.abs(sub_h["b_hat"].values)
    h_med, h_hi = np.percentile(sub_h_abs, [50, 95])

    # Noise-free human reference (matches 04_profile_instability/01): mean signed
    # b̂ across persons, THEN magnitude (per-person |b̂| would carry response noise).
    b_avg_signed_h = float(np.mean(sub_h["b_hat"].values))
    abs_b_avg_h    = abs(b_avg_signed_h)

    # ----------------------------------------------------------
    # ROW 0: decomposition scatter
    # ----------------------------------------------------------
    ax_top = fig.add_subplot(gs[0, i])

    # Human density (background): KDE over the full human sample
    x_h = sub_h["theta_hat"].values - M
    y_h = sub_h["b_hat"].values
    kde = gaussian_kde(np.vstack([x_h, y_h]))
    xx_g = np.linspace(-AX_LIM, AX_LIM, KDE_GRID)
    yy_g = np.linspace(-AX_LIM, AX_LIM, KDE_GRID)
    Xg, Yg = np.meshgrid(xx_g, yy_g)
    Zg = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
    z_max = float(Zg.max())
    if z_max > 0:
        levels = np.linspace(z_max * 0.02, z_max, KDE_LEVELS)
        ax_top.contourf(Xg, Yg, Zg, levels=levels, cmap=CMAP_HUMAN, zorder=2)

    # Full LLM scatter (foreground): proprietary models in see-through blue, on top.
    for origin, style in LLM_POINT_STYLE:
        sub_o = sub_l[sub_l["origin"] == origin]
        ax_top.scatter(
            sub_o["theta_hat"] - M, sub_o["b_hat"],
            s=LLM_SIZE_TOP, edgecolor="white", linewidth=0.5, **style,
        )

    ax_top.axhline(0, color="#aa3300", linestyle=":", linewidth=1.4, zorder=3)
    ax_top.set_xlim(-AX_LIM, AX_LIM)
    ax_top.set_ylim(-AX_LIM, AX_LIM)
    ax_top.set_xticks([-2, -1, 0, 1, 2])
    ax_top.grid(alpha=GRID_ALPHA)
    ax_top.tick_params(labelsize=FS_TICK)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # Column header = full trait name above each top panel
    ax_top.set_title(TRAIT_NAMES[t], fontsize=FS_TITLE, fontweight="bold", pad=8)

    # y-axis label only on leftmost column
    if i == 0:
        ax_top.set_ylabel(r"$\hat b$  (response bias)", fontsize=FS_LABEL)
    else:
        plt.setp(ax_top.get_yticklabels(), visible=False)

    # x-axis label only on middle column
    if i == 2:
        ax_top.set_xlabel(r"$\hat\theta - m$  (latent trait)",
                          fontsize=FS_LABEL)

    # π_b annotations: colour-coded, bottom corners
    ax_top.text(
        0.03, 0.04,
        rf"$\pi_b^\mathrm{{H}}\!=\!{pi_b[t]['human']:.2f}$",
        transform=ax_top.transAxes, ha="left", va="bottom",
        fontsize=FS_STATS, color=C_HUMAN,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.82, ec="none"),
    )
    ax_top.text(
        0.97, 0.04,
        rf"$\pi_b^\mathrm{{L}}\!=\!{pi_b[t]['llm']:.2f}$",
        transform=ax_top.transAxes, ha="right", va="bottom",
        fontsize=FS_STATS, color=C_LLM,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.82, ec="none"),
    )

    # ----------------------------------------------------------
    # ROW 1: |b̂| — size subpanel + open/prop strip
    # ----------------------------------------------------------
    inner = gs[1, i].subgridspec(1, 2, width_ratios=[2.5, 1], wspace=0.25)
    ax_sz = fig.add_subplot(inner[0, 0])
    ax_oc = fig.add_subplot(inner[0, 1], sharey=ax_sz)

    # Optimal-zero reference + spine cleanup (both subpanels)
    for ax_ in [ax_sz, ax_oc]:
        ax_.axhline(0, color="#aa3300", linestyle=":", linewidth=1.0, zorder=2)
        ax_.spines["top"].set_visible(False)
        ax_.spines["right"].set_visible(False)

    # |b̂| vs log10(size), open-source LLMs
    sub_open = sub_l[sub_l["origin"] == "open"].dropna(subset=["size"])
    xs_o = np.log10(sub_open["size"].astype(float).values)
    ys_o = np.abs(sub_open["b_hat"].values)
    ax_sz.scatter(xs_o, ys_o, c=C_LLM, s=LLM_SIZE_BOT,
                  edgecolor="white", linewidth=0.4, zorder=4)

    slope = r_val = p_val = np.nan
    if len(xs_o) >= 3:
        slope, intercept, r_val, p_val, _ = stats.linregress(xs_o, ys_o)
        xx = np.linspace(xs_o.min() - 0.05, xs_o.max() + 0.05, 100)
        ax_sz.plot(xx, slope * xx + intercept, "-", color="black",
                   linewidth=1.2, zorder=3)

    size_ticks = [1, 10, 100]
    ax_sz.set_xticks(np.log10(size_ticks))
    ax_sz.set_xticklabels([f"{s}B" for s in size_ticks], fontsize=FS_TICK)
    ax_sz.set_ylim(-abs_b_max * 0.03, abs_b_max)
    ax_sz.set_xlabel("size (log)", fontsize=FS_LABEL)
    ax_sz.tick_params(axis="y", labelsize=FS_TICK)
    ax_sz.grid(alpha=GRID_ALPHA)

    if i == 0:
        ax_sz.set_ylabel(r"$|\hat b|$  (0 = optimal)", fontsize=FS_LABEL)
    else:
        plt.setp(ax_sz.get_yticklabels(), visible=False)

    # Open vs Prop strip
    rng_jit = np.random.default_rng(i + 100)
    o_v = np.abs(sub_l[sub_l["origin"] == "open"]["b_hat"].values)
    c_v = np.abs(sub_l[sub_l["origin"] == "closed"]["b_hat"].values)

    for x_pos, vals, fill, alpha in [(0, o_v, C_LLM, 0.95),
                                     (1, c_v, C_LLM_API, 0.7)]:
        if len(vals) == 0:
            continue
        jitter = rng_jit.uniform(-0.14, 0.14, len(vals))
        ax_oc.scatter(
            np.full(len(vals), x_pos) + jitter, vals,
            c=fill, s=26, edgecolor="white", linewidth=0.4,
            alpha=alpha, zorder=3,
        )
        ax_oc.plot(
            [x_pos - 0.26, x_pos + 0.26], [np.median(vals)] * 2,
            "k-", linewidth=2.0, zorder=4,
        )

    # Human |b̂| violin on the far right (next to proprietary models)
    h_v = np.abs(sub_h["b_hat"].values)
    h_v = h_v[h_v <= abs_b_max]
    if len(h_v) > 0:
        parts = ax_oc.violinplot(
            [h_v], positions=[2], widths=0.85,
            showmedians=True, showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(C_HUMAN)
            body.set_edgecolor(C_HUMAN)
            body.set_alpha(0.55)
            body.set_linewidth(0.6)
        if "cmedians" in parts:
            med = parts["cmedians"]
            med.set_color(C_HUMAN)
            med.set_linewidth(2.6)
            med.set_zorder(5)
            # widen median tick so it shows under the diamond when they coincide
            segs = med.get_segments()
            if segs:
                y_med = segs[0][0][1]
                med.set_segments([[[2 - 0.38, y_med], [2 + 0.38, y_med]]])

    # Noise-free human |b̄| as a diamond on the human violin column.
    if abs_b_avg_h <= abs_b_max:
        ax_oc.scatter(
            [2], [abs_b_avg_h], marker="D", s=22,
            facecolor=C_HUMAN, edgecolor="white", linewidth=0.9, zorder=6,
        )

    ax_oc.set_xticks([0, 1, 2])
    ax_oc.set_xticklabels(["Op.", "Pr.", "H."], fontsize=FS_TICK)
    ax_oc.set_xlim(-0.55, 2.6)
    plt.setp(ax_oc.get_yticklabels(), visible=False)
    ax_oc.grid(axis="y", alpha=GRID_ALPHA)

    p_mw = np.nan
    if len(o_v) > 1 and len(c_v) > 1:
        _, p_mw = stats.mannwhitneyu(o_v, c_v, alternative="two-sided")

    trait_stats.append({
        "trait": t,
        "size_r": r_val, "size_p": p_val, "size_slope": slope,
        "median_abs_b_open":   float(np.median(o_v)) if len(o_v) else np.nan,
        "median_abs_b_closed": float(np.median(c_v)) if len(c_v) else np.nan,
        "median_abs_b_human":  float(h_med),
        "human_95pct_abs_b":   float(h_hi),
        "b_avg_human_signed":  b_avg_signed_h,
        "abs_b_avg_human":     float(abs_b_avg_h),
        "mw_p_open_vs_prop":   p_mw,
    })

# ============================================================
# Figure legend — two boxes, one per row (b̂=0 baseline appears in both)
# ============================================================

top_handles = [
    Patch(facecolor=C_HUMAN, alpha=0.55, edgecolor="none",
          label=fr"Humans, KDE of $(\hat\theta - m,\, \hat b)$  ($n$ = {n_humans:,})"),
    Line2D([0], [0], marker="o", linestyle="", markersize=9,
           markerfacecolor=C_LLM, markeredgecolor="white", markeredgewidth=0.5,
           label=fr"Open-source LLMs, per-model $(\hat\theta - m,\, \hat b)$"),
    Line2D([0], [0], marker="o", linestyle="", markersize=9, alpha=0.7,
           markerfacecolor=C_LLM_API, markeredgecolor="white", markeredgewidth=0.5,
           label=fr"Proprietary LLMs, per-model $(\hat\theta - m,\, \hat b)$"),
    Line2D([0], [0], color="#aa3300", linestyle=":", linewidth=1.8,
           label=r"$\hat b = 0$ (bias-free baseline)"),
]

bottom_handles = [
    Patch(facecolor=C_HUMAN, alpha=0.55, edgecolor=C_HUMAN, linewidth=0.6,
          label=fr"Humans, KDE of $|\hat b|$; inner bar = median)"),
    Line2D([0], [0], color="black", linestyle="-", linewidth=2.0,
           label=r"LLM group median $|\hat b|$"),
    Line2D([0], [0], marker="D", linestyle="", markersize=8,
           markerfacecolor=C_HUMAN, markeredgecolor="white", markeredgewidth=0.6,
           label=r"Humans, $|\bar b|$ of mean responses"),
    Line2D([0], [0], color="black", linestyle="-", linewidth=1.2,
           label="OLS fit, open-source LLMs"),
]

leg_top = fig.legend(
    handles=top_handles,
    title=r"Top row:",
    loc="upper left", ncol=2,
    fontsize=FS_STATS, title_fontsize=FS_LABEL,
    frameon=True, framealpha=0.90, edgecolor="grey",
    bbox_to_anchor=(0.055, 0.075),
    handlelength=1.6, handletextpad=0.6, borderpad=0.8,
    labelspacing=0.5, columnspacing=1.4,
)
leg_top._legend_box.align = "left"

leg_bot = fig.legend(
    handles=bottom_handles,
    title=r"Bottom row:",
    loc="upper left", ncol=2,
    fontsize=FS_STATS, title_fontsize=FS_LABEL,
    frameon=True, framealpha=0.90, edgecolor="grey",
    bbox_to_anchor=(0.54, 0.075),
    handlelength=2.0, handletextpad=0.7, borderpad=0.8,
    labelspacing=0.5, columnspacing=2.2,
)
leg_bot._legend_box.align = "left"

# ============================================================
# Save + export table
# ============================================================

fig.savefig(OUT_FIG_DIR / FIGURE_NAME, bbox_inches="tight", dpi=200)
print(f"Saved: {OUT_FIG_DIR / FIGURE_NAME}")

stats_df = pd.DataFrame(trait_stats)
stats_df.to_csv(OUT_TABLE_DIR / "b_per_trait_stats.csv", index=False)
print("\nPer-trait summary:")
print(stats_df.round(3).to_string(index=False))