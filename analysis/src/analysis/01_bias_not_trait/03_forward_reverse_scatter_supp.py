"""SI Fig S1: forward-reverse item-mean scatter for the four non-Neuroticism traits
(O, C, E, A), humans vs LLMs on the IPIP-NEO-300. Companion to main-text Fig 1.
Writes results/figures/forward_reverse_scatter_supp.pdf."""
import pandas as pd
import matplotlib.pyplot as plt
from utils import plot_corr_panel

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

TRAITS    = ["O", "C", "E", "A"]
SAVE_PATH = "../../../results/figures/forward_reverse_scatter_supp.pdf"

# colors
C_LLM   = "#1e847f"
C_HUMAN = "#440154"
C_LLM_API = "#287c8e"

# typography
FS_LABEL    = 11
FS_TITLE    = 11
FS_STATS    = 9
FS_SUBLABEL = 11

# figure layout
FIG_SIZE = (9, 16)

# panel letters: column 0 = Humans, column 1 = LLMs
# auto-generated as A/B, C/D, E/F, G/H for 4 traits
SUBPLOT_LABELS = {
    (i, j): chr(65 + i * 2 + j)     
    for i in range(len(TRAITS))
    for j in range(2)
}

# titles and axis labels are built dynamically from TRAITS in the loop below

# ══════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════

raw_humans = pd.read_csv(
    "../../../data/ipipneo300_data/human_data/ipipneo_human_raw.csv",
    low_memory=False,
)
raw_humans["response"] = pd.to_numeric(
    raw_humans["response"].replace({"0": None, 0: None, " ": None}),
    errors="coerce",
)
raw_humans = raw_humans.dropna(subset=["response"])

raw_llms = pd.read_csv(
    "../../../data/ipipneo300_data/llm_data/ipipneo_no_flip_data_raw.csv",
    low_memory=False,
)
api_data = pd.read_csv(
    "../../../data/ipipneo300_data/api_data/ipipneo_api_data_raw.csv",
    low_memory=False,
)
api_nr = api_data[api_data["model"].str.contains("_nr", regex=False)]
raw_llms = pd.concat([raw_llms, api_nr], ignore_index=True)
raw_llms = (raw_llms[raw_llms["context_mode"] == "no_context"]
            .rename(columns={"model_answer": "response"}))

# ══════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(len(TRAITS), 2, figsize=FIG_SIZE, sharey=False)

for i, trait in enumerate(TRAITS):
    llm_trait   = raw_llms[raw_llms["traits"] == trait]
    human_trait = raw_humans[raw_humans["traits"] == trait]

    plot_corr_panel(axes[i, 0], human_trait, entity_var="person_id",
                    dependent_var="response", color=C_HUMAN, FS_STATS=FS_STATS,
                    point_alpha=0.4, point_size=8,  show_stats=True)

    plot_corr_panel(axes[i, 1], llm_trait,   entity_var="model",
                    dependent_var="response", color=C_LLM,   FS_STATS=FS_STATS,
                    point_alpha=0.7, point_size=40, show_stats=True)
    
    # overlay API models
    llm_divided = (
        llm_trait
        .groupby(["model", "reverse_coded"], as_index=False)["response"]
        .mean()
        .pivot(index="model", columns="reverse_coded", values="response")
        .reset_index()
    )
    api_pts = llm_divided[llm_divided["model"].str.contains("__", regex=False)]
    axes[i, 1].scatter(api_pts[False], api_pts[True],
                       color=C_LLM_API, alpha=0.7, s=40, zorder=5)

# ── unified theme + labels ────────────────────────────────────────────────
for i, trait in enumerate(TRAITS):
    for j, (entity, color) in enumerate([("Humans", C_HUMAN), ("LLMs", C_LLM)]):
        ax = axes[i, j]

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # panel letter
        ax.text(-0.08, 1.05, SUBPLOT_LABELS[(i, j)],
                transform=ax.transAxes, fontsize=FS_SUBLABEL,
                fontweight="bold", va="top", ha="left")

        # title
        ax.set_title(f"{entity} - Trait {trait}",
                     fontsize=FS_TITLE, fontweight="bold", pad=6)

        # axis labels
        ax.set_ylabel("Mean of reversed items", fontsize=FS_LABEL)
        ax.set_xlabel("Mean of forward items", fontsize=FS_LABEL)

# ── layout & save ─────────────────────────────────────────────────────────
plt.tight_layout()
fig.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)