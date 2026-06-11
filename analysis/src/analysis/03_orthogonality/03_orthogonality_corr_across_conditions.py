"""Fig 3b: cross-instrument p_r vs mean-inter-item correlation (Pearson r, 95% CI)
shown per prompting condition, with small/large size-split markers (forest plot).

Reads results/tables/orthogonality_corr_by_condition.csv (from script 02).
Writes results/figures/orthogonality_across_conditions.pdf.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

DATA_PATH   = "../../../results/tables/orthogonality_corr_by_condition.csv"
SAVE_PATHS  = [
    "../../../results/figures/orthogonality_across_conditions.pdf",
]

# colors
C_LLM   = "#1e847f"
C_HUMAN = "#440154"

# markers
MARKER_PEARSON  = "o"
MARKERSIZE      = 8

# S / L dots + letter annotations, drawn slightly above each row
SUB_DOT_SIZE    = 5         # marker size for the S / L dots
LETTER_SIZE     = 8        # font size for the S / L letter annotations
LETTER_DX       = 0.020     # gap (data units) between a dot and its outside letter
SUB_Y_OFFSET    = 0.28      # how far above the row's center the dots sit (axis is inverted)
SUB_ALPHA       = 0.75

# alpha
ALPHA_LLM   = 0.75
ALPHA_HUMAN   = 0.35

# typography
FS_YTICKS  = 13
FS_XLABEL  = 13

# layout — saved aspect ratio tuned to match panel (c) after LaTeX rescaling
ROW_HEIGHT = 0.565
FIG_EXTRA  = 0.5

# axes
XLIM = (-1.08, 0.05)
XLABEL = "Pearson r" # (95% CI)"

# error bar style
ELINEWIDTH = 1.3
CAPSIZE    = 3
CAPTHICK   = 1.3

# separator between LLM and Human rows
SEP_COLOR  = "gray"
SEP_LW     = 0.7
SEP_LS     = ":"

# ══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def parse_ci(val):
    """Parse CI from tuple, list, array, or string representation."""
    if isinstance(val, (tuple, list, np.ndarray)):
        return float(val[0]), float(val[1])
    nums = re.findall(r'-?\d+\.\d+', str(val))
    return float(nums[0]), float(nums[1])


def make_label(row):
    """Build a two-line condition label. Asterisk marks conditions
    that include the IPIP-NEO-300 scales (LLM direct and Humans)."""
    if row["condition"] == "Humans":
        return "Humans*"

    if "LLM direct" in row["condition"]:
        cond_short, ipip = "Direct", "*"
    else:
        cond_short, ipip = "Weighted", ""

    ctx = "context" if row["context"] == "with_context" else "no context"
    score_label = "textgen" if row["score"] == "model_answer" else "logit"
    flip = "no flip" if "no flip" in row["condition"] else "flip"

    return f"{cond_short}{ipip}, {ctx}\n{score_label}, {flip}"

# ══════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════

df = pd.read_csv(DATA_PATH)

plot_rows = []
for _, row in df.iterrows():
    label = make_label(row)
    lo, hi = parse_ci(row["pearson_ci"])
    plot_rows.append({
        "label": label,
        "r": row["pearson_r"],
        "r_small": row.get("pearson_r_small", np.nan),
        "r_large": row.get("pearson_r_large", np.nan),
        "lo": lo,
        "hi": hi,
        "human": row["condition"] == "Humans",
    })

plot_df = pd.DataFrame(plot_rows)
labels = list(dict.fromkeys(plot_df["label"]))
n = len(labels)
y_pos = {lbl: i for i, lbl in enumerate(labels)}

# ══════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, n * ROW_HEIGHT + FIG_EXTRA))
# add horizontal row discriminator lines 
for i in range(n):
    if i % 2 == 0:
        ax.axhspan(
            i - 0.5,
            i + 0.5,
            color="grey",
            alpha=0.06,
            zorder=0
        )
for _, row in plot_df.iterrows():
    y = y_pos[row["label"]]
    color = C_HUMAN if row["human"] else C_LLM

    # main "all-models" point with 95% CI error bar (original style)
    ax.errorbar(
        row["r"], y,
        xerr=[[row["r"] - row["lo"]], [row["hi"] - row["r"]]],
        fmt=MARKER_PEARSON, color=color,
        markersize=MARKERSIZE, alpha=ALPHA_HUMAN if row["human"] else ALPHA_LLM,
        elinewidth=ELINEWIDTH, capsize=CAPSIZE, capthick=CAPTHICK, ecolor=color,
    )

    # S and L dots + letter annotations, slightly above the row (LLM rows only)
    if not row["human"]:
        y_sub = y - SUB_Y_OFFSET
        rs = row["r_small"] if np.isfinite(row["r_small"]) else None
        rl = row["r_large"] if np.isfinite(row["r_large"]) else None

        for x in (rs, rl):
            if x is not None:
                ax.plot(x, y_sub, MARKER_PEARSON, color=color, alpha=SUB_ALPHA,
                        markersize=SUB_DOT_SIZE, markeredgewidth=0)

        # Annotate each dot with its letter on the OUTSIDE of the pair,
        # so the two letters can never overlap each other.
        if rs is not None and rl is not None:
            if rs <= rl:
                left_x,  left_lbl  = rs, "S"
                right_x, right_lbl = rl, "L"
            else:
                left_x,  left_lbl  = rl, "L"
                right_x, right_lbl = rs, "S"
            ax.text(left_x - LETTER_DX, y_sub, left_lbl,
                    color=color, alpha=SUB_ALPHA,
                    fontsize=LETTER_SIZE, fontweight="bold",
                    ha="right", va="center")
            ax.text(right_x + LETTER_DX, y_sub, right_lbl,
                    color=color, alpha=SUB_ALPHA,
                    fontsize=LETTER_SIZE, fontweight="bold",
                    ha="left", va="center")
        elif rs is not None:
            ax.text(rs + LETTER_DX, y_sub, "S",
                    color=color, alpha=SUB_ALPHA,
                    fontsize=LETTER_SIZE, fontweight="bold",
                    ha="left", va="center")
        elif rl is not None:
            ax.text(rl + LETTER_DX, y_sub, "L",
                    color=color, alpha=SUB_ALPHA,
                    fontsize=LETTER_SIZE, fontweight="bold",
                    ha="left", va="center")

# separator line above the first Humans row
human_labels = [l for l in labels if "Humans" in l]
if human_labels:
    human_y = y_pos[human_labels[0]]
    ax.axhline(human_y - 0.5, color=SEP_COLOR, linewidth=SEP_LW, linestyle=SEP_LS)

# axes styling
ax.grid(axis="x", alpha=0.2)
ax.set_yticks(range(n))
ax.set_yticklabels(labels, fontsize=FS_YTICKS)
ax.invert_yaxis()
ax.set_xlabel(XLABEL, fontsize=FS_XLABEL)
ax.set_xlim(XLIM)
ax.spines[["top", "right"]].set_visible(False)

# panel label (figure-relative for consistent placement across figures)
fig.text(0.02, 0.98, "B", fontsize=17, fontweight="bold",
         va="top", ha="left")

# ── layout & save ─────────────────────────────────────────────────────────
plt.tight_layout()
for path in SAVE_PATHS:
    fig.savefig(path, dpi=200, bbox_inches="tight")

