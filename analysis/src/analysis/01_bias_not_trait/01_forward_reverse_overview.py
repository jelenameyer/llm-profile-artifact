"""Main-text Fig 1: forward-reverse item-mean scatter for Neuroticism (the median
trait) with the schematic trait/bias reference panel, humans vs LLMs on the
IPIP-NEO-300. Writes results/figures/forward_reverse_overview.pdf."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils import plot_corr_panel, clean_model_label
# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

TRAIT     = "N"
SAVE_PATH = "../../../results/figures/forward_reverse_overview.pdf"

# colors
C_LLM   = "#1e847f"
C_HUMAN = "#440154"
C_LLM_API = "#287c8e"

# typography
FS_LABEL    = 11   # axis labels
FS_TITLE    = 13   # subplot titles
FS_STATS    = 9    # in-plot stats box
FS_SUBLABEL = 11   # panel letters (A), (B), …

# panel letters — set to None to suppress
SUBPLOT_LABELS = {
    (0): "A",
    (1): "B",
    (2): "C",
}

# panel titles — set to None to suppress
SUBPLOT_TITLES = {
    (0): "Expectations",
    (1): "Human",
    (2): "LLM",
}

# axis labels — set to None to suppress
SUBPLOT_XLABELS = {
    
    (0): "Mean of forward items",
    (1): "Mean of forward items",
    (2): "Mean of forward items",
}

SUBPLOT_YLABELS = {
    (0): "Mean of reversed items",
    (1): "Mean of reversed items",
    (2): "Mean of reversed items",
}

# labelpad per panel — None uses matplotlib's default (~4 pt)
SUBPLOT_XLABELPADS = {
    (0): 20,
    (1): None,
    (2): None,
}
SUBPLOT_YLABELPADS = {
    (0): None,
    (1): None,
    (2): None,
}

# figure layout
FIG_SIZE = (12, 4.5)
H_PAD    = 3.5
LINE_WIDTH = 3

# ══════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════

raw_humans = pd.read_csv(
    "../../../data/intermediate/ipipneo300_data/human_data/ipipneo_human_raw.csv",
    low_memory=False,
)
raw_humans["response"] = pd.to_numeric(
    raw_humans["response"].replace({"0": None, 0: None, " ": None}),
    errors="coerce",
)
raw_humans = raw_humans.dropna(subset=["response"])

raw_llms = pd.read_csv(
    "../../../data/intermediate/ipipneo300_data/llm_data/ipipneo_no_flip_data_raw.csv",
    low_memory=False,
)
# add API non-reasoning models
api_data = pd.read_csv(
    "../../../data/intermediate/ipipneo300_data/api_data/ipipneo_api_data_raw.csv",
    low_memory=False,
)
raw_llms = pd.concat([raw_llms, api_data[api_data["model"].str.contains("_nr", regex=False)]], ignore_index=True)

raw_llms = (raw_llms[raw_llms["context_mode"] == "no_context"]
            .rename(columns={"model_answer": "response"}))

llm_trait   = raw_llms[raw_llms["traits"] == TRAIT]
human_trait = raw_humans[raw_humans["traits"] == TRAIT]


# ══════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════

SCHEMATIC_PANELS = [(0)]

fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE, sharey=False)

# ── schematic panel ──────────────────────────────────────────────────────

x = np.arange(10)

# Shaded region between the two reference lines (bowtie shape)
x_fill = np.linspace(0, 9, 300)
axes[0].fill_between(x_fill,
                     np.minimum(9 - x_fill, x_fill),   # lower envelope
                     np.maximum(9 - x_fill, x_fill),   # upper envelope
                     alpha=0.12, color='grey', zorder=0)

# Reference lines on top of shading
axes[0].plot(x, np.arange(9, -1, -1), color="grey", linewidth=LINE_WIDTH)
axes[0].plot(x, np.arange(10),        color="grey", linewidth=LINE_WIDTH)

# Labels
axes[0].text(1.4, 2,   "response bias", rotation=45,  fontsize=FS_LABEL,
             ha="left", va="bottom")
axes[0].text(0.84, 7.7, "trait",       rotation=-44, fontsize=FS_LABEL,
             ha="left", va="bottom")

# ── data panels ───────────────────────────────────────────────────────────
plot_corr_panel(axes[1], human_trait, entity_var="person_id",
                dependent_var="response", color=C_HUMAN, FS_STATS=FS_STATS, point_alpha=0.4, point_size=8, show_stats=False)
plot_corr_panel(axes[2], llm_trait,   entity_var="model",
                dependent_var="response", color=C_LLM,  FS_STATS=FS_STATS, point_alpha=0.7, point_size=40, show_stats=False)

# add model labels in LLM panel (label all models once)
llm_divided = (
    llm_trait
    .groupby(["model", "reverse_coded"], as_index=False)["response"]
    .mean()
    .pivot(index="model", columns="reverse_coded", values="response")
    .reset_index()
)


target_models = {
    "gpt-oss-20b",
    "Llama-3.3-70B-Instruct",
    "Qwen3-8B",
    "gemma-3-4b-it",
    "openai__gpt-5.4__nr",
    "anthropic__claude-opus-4-6__nr",
    "x-ai__grok-4.20-0309-non-reasoning__nr",
} 

label_offsets = {
    "gpt-oss-20b": (0.3, 0.3),
    "Llama-3.3-70B-Instruct": (-0.005, 0.52),
    "Qwen3-8B": (-0.0, -0.3),
    "gemma-3-4b-it": (0.0001, -0.7),
    "openai__gpt-5.4__nr": (0.3, 0.5),
    "anthropic__claude-opus-4-6__nr": (0.0, -0.5),
    "x-ai__grok-4.20-0309-non-reasoning__nr": (0.00001, -0.9),
}



for _, row in llm_divided.iterrows():
    model = row["model"]
    if model not in target_models:
        continue
    x_val = row[False]
    y_val = row[True]
    if pd.isna(x_val) or pd.isna(y_val):
        continue
    dx, dy = label_offsets.get(model, (0.15, 0.15))
    label = clean_model_label(model)
    text_x = x_val + dx
    text_y = y_val + dy
    axes[2].annotate(
        label,
        xy=(x_val, y_val),
        xytext=(text_x, text_y),
        textcoords="data",
        fontsize=FS_STATS,
        ha="left" if dx > 0 else "right",
        va="bottom" if dy > 0 else "top",
        arrowprops=dict(arrowstyle="-", color="grey", linewidth=0.8,
                        shrinkA=0, shrinkB=0),
    )

# add api models in different colour (to show difference)
api_mask = llm_divided["model"].str.contains("__", regex=False)
api_pts  = llm_divided[api_mask]
axes[2].scatter(
    api_pts[False], api_pts[True],
    color=C_LLM_API, alpha=0.7, s=30,
)
# ── unified theme + labels ────────────────────────────────────────────────
for j in range(3):
    ax = axes[j]
    if j not in SCHEMATIC_PANELS:
        ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if j in SCHEMATIC_PANELS:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(0)

    if SUBPLOT_LABELS.get(j):
        ax.text(-0.08, 1.05, SUBPLOT_LABELS[j],
                transform=ax.transAxes, fontsize=FS_SUBLABEL,
                fontweight="bold", va="top", ha="left")

    if SUBPLOT_TITLES.get(j):
        ax.set_title(SUBPLOT_TITLES[j], fontsize=FS_TITLE,
                     fontweight="bold", pad=10)

    if SUBPLOT_XLABELS.get(j):
        pad = SUBPLOT_XLABELPADS.get(j)
        ax.set_xlabel(SUBPLOT_XLABELS[j], fontsize=FS_LABEL,
                    **({'labelpad': pad} if pad is not None else {}))

    if SUBPLOT_YLABELS.get(j):
        pad = SUBPLOT_YLABELPADS.get(j)
        ax.set_ylabel(SUBPLOT_YLABELS[j], fontsize=FS_LABEL,
                    **({'labelpad': pad} if pad is not None else {}))

# ── layout & save ─────────────────────────────────────────────────────────
plt.tight_layout(rect=(0, 0.05, 1, 1), h_pad=H_PAD)

# add shared "Empirical" header with lines above human/LLM panels
pos_left = axes[1].get_position()
pos_right = axes[2].get_position()
y_header = max(pos_left.y1, pos_right.y1) + 0.1
x_left = pos_left.x0
x_right = pos_right.x1
x_center = (x_left + x_right) / 2
gap = 0.06
fig.text(x_center, y_header, "Empirical", ha="center", va="center",
         fontsize=FS_TITLE, fontweight="bold")
fig.add_artist(Line2D([x_left, x_center - gap], [y_header, y_header],
                      transform=fig.transFigure, color="black", linewidth=1))
fig.add_artist(Line2D([x_center + gap, x_right], [y_header, y_header],
                      transform=fig.transFigure, color="black", linewidth=1))

fig.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
