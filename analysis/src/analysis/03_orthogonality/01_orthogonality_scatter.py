"""Fig 3a: mean inter-item correlation vs proportion of reverse-keyed items, one
point per instrument, LLMs vs humans (risk battery + IPIP-NEO-300 traits).

Reads risk_data + ipipneo300_data (LLM/api/human).
Writes results/figures/orthogonality_scatter.pdf.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from pathlib import Path
from adjustText import adjust_text
from utils import (
    compute_mean_interitem_corr,
    compute_pct_reversed_map,
    add_pct_reversed,
    apply_domain_fixes,
)

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent

LLM_PATH = BASE_DIR / "../../../data/intermediate/risk_data/LLM_data_proc_prompts_direct/LLM_no_flip_data_rereversed.csv"
LLM_API_PATH = BASE_DIR / "../../../data/intermediate/risk_data/api_data/LLM_api_no_flip_data_rereversed.csv"
HUMAN_PATH = BASE_DIR / "../../../data/intermediate/risk_data/human_data_proc/items_per_person.csv"
IPIP_LLM_PATH = BASE_DIR / "../../../data/intermediate/ipipneo300_data/llm_data/ipipneo_all_data_reflipped_and_rereversed.csv"
IPIP_API_PATH = BASE_DIR / "../../../data/intermediate/ipipneo300_data/api_data/ipipneo_api_data_rereversed.csv"
IPIP_HUMAN_PATH = BASE_DIR / "../../../data/intermediate/ipipneo300_data/human_data/ipipneo_human.csv"

SAVE_PATHS = [
    BASE_DIR / "../../../results/figures/orthogonality_scatter.pdf",
]

DOMAIN_EXCLUDE = {"SOEP scale", "SOEP"}

LABEL_OVERRIDES = {
    "decision": "PRI",
    "Ethical": "Deth",
    "Gambling": "Dgam",
    "Health": "Dhea",
    "Investment": "Dinv",
    "Recreational": "Drec",
    "Social": "Dsoc",
}

# jitter
RNG_SEED = 13
JITTER_X = 0.6
JITTER_Y = 0.01

# colors
C_LLM = "#21918c"
C_HUMAN = "#4D025F"
C_LLM_LINE = "#1e847f"
C_HUMAN_LINE = "#440154"
C_LLM_TEXT = "#0d4f4b"
C_HUMAN_TEXT = "#3b0f6e"

# markers
POINT_SIZE = 65
LW_POINTS = 1.0

# behavioural (task-based) vs survey-based instruments — shape encodes task type
BEHAVIOURAL = {"MPL", "DFD", "LOT"}
MARKER_SURVEY = "o"
MARKER_BEHAV = "D"
C_NEUTRAL = "#555555"

# alpha
ALPHA_LLM = 0.90
ALPHA_HUMAN = 0.30
ALPHA_LLM_TEXT = 0.85
ALPHA_HUMAN_TEXT = 0.55

# regression lines
LW_LLM = 2.0
LW_HUMAN = 1.8
ALPHA_LLM_LINE = 0.95
ALPHA_HUMAN_LINE = 0.35

# typography
FS_BASE = 9
FS_TITLE = 11
FS_LABEL = 11
FS_LEGEND = 11
FS_TEXT = 7

# figure
FIGSIZE = (10.4, 6.2)

# adjustText
ARROWPROPS = dict(arrowstyle="-", color="gray", lw=0.35, alpha=0.55)
EXPAND = (1.3, 1.5)
FORCE_TEXT = (0.4, 0.6)
FORCE_POINTS = (0.2, 0.3)
MIN_ARROW_LEN = 8

# legend labels
LEGEND_LLM_POINTS = "LLMs [no context, textgen, no flip]"
LEGEND_HUMAN_POINTS = "Humans"
LEGEND_LLM_LINE = "LLMs regression"
LEGEND_HUMAN_LINE = "Human regression"

# ══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def _plot_regression(ax, x, y, color, lw=1.8, ls="-", line_alpha=1.0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return None
    slope, intercept, r, p, _ = stats.linregress(x, y)
    xg = np.linspace(x.min(), x.max(), 200)
    ax.plot(xg, intercept + slope * xg, color=color, lw=lw, ls=ls, alpha=line_alpha, zorder=3)
    return r, p, len(x)


def _corr_stats(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float("nan"), float("nan"), len(x)
    return stats.pearsonr(x, y)[0], stats.spearmanr(x, y)[0], len(x)


# ══════════════════════════════════════════════════════════════════════════
# LOAD & PROCESS
# ══════════════════════════════════════════════════════════════════════════

llm = pd.read_csv(LLM_PATH, low_memory=False)
llm = llm[llm["context_mode"] == "no_context"]

# Add the 10 proprietary API models (no-reasoning / no_context / no-flip — the
# only risk condition collected for them, matching the open-weights slice above).
llm_api = pd.read_csv(LLM_API_PATH, low_memory=False)
llm_api = llm_api[
    llm_api["model"].str.endswith("__nr")
    & (llm_api["context_mode"] == "no_context")
    & (llm_api["flipped"] == False)
]
llm = pd.concat([llm, llm_api], ignore_index=True)

hum = pd.read_csv(HUMAN_PATH, low_memory=False)
hum["experiment"] = (hum["experiment"].str.replace(r"\s.+$", "", regex=True)
                                      .str.replace("BARRAT", "BARRATT", regex=False))

# IPIP-NEO-300 (LLM + Human)
ipip_llm = pd.read_csv(IPIP_LLM_PATH, low_memory=False)
ipip_llm = ipip_llm[(ipip_llm["context_mode"] == "no_context") & (ipip_llm["flipped"] == False)].copy()

# Same 10 proprietary API models on the IPIP-NEO side (no-reasoning condition),
# so the Big Five points use the same 56-model roster as the risk points.
ipip_api = pd.read_csv(IPIP_API_PATH, low_memory=False)
ipip_api = ipip_api[
    ipip_api["model"].str.endswith("__nr")
    & (ipip_api["context_mode"] == "no_context")
    & (ipip_api["flipped"] == False)
].copy()
ipip_llm = pd.concat([ipip_llm, ipip_api], ignore_index=True)
ipip_llm["category"] = ipip_llm["traits"]

ipip_hum = pd.read_csv(IPIP_HUMAN_PATH, low_memory=False)
ipip_hum["response"] = pd.to_numeric(ipip_hum["response"].replace({"0": None, 0: None, " ": None}))
ipip_hum = ipip_hum.dropna(subset=["response"]).copy()
ipip_hum["experiment"] = "IPIP-NEO-300"
ipip_hum["category"] = ipip_hum["traits"]
ipip_hum = ipip_hum.rename(columns={"response": "score", "person_id": "partid"})

llm_all = pd.concat([llm, ipip_llm], ignore_index=True)
hum_all = pd.concat([hum, ipip_hum], ignore_index=True)

mean_llm = apply_domain_fixes(compute_mean_interitem_corr(
    llm_all, score="model_answer", item_name="item_id",
    individual="model", domain_exclude=DOMAIN_EXCLUDE), is_llm=True)
mean_hum = apply_domain_fixes(compute_mean_interitem_corr(
    hum_all, score="score", item_name="item",
    individual="partid", domain_exclude=DOMAIN_EXCLUDE), is_llm=False)

llm_experiments = set(llm_all["experiment"].dropna().unique())
mean_llm = mean_llm[mean_llm["experiment"].isin(llm_experiments)]
mean_hum = mean_hum[mean_hum["experiment"].isin(llm_experiments)]

pct_map = compute_pct_reversed_map(llm_all)
mean_llm = add_pct_reversed(mean_llm, pct_map)
# LOT presentation order was randomised per trial for humans, so the LLM-instrument
# p_r (0.40 after folding) doesn't reflect what humans experienced (0.50 by design).
# Override the LOT entry on a humans-only copy of the map.
pct_map_hum = pct_map.copy()
pct_map_hum.loc[pct_map_hum["experiment"] == "LOT", "pct_reversed_keyed_items"] = 0.5
mean_hum = add_pct_reversed(mean_hum, pct_map_hum)

# scale_label: subdomain name if present, else experiment name
def scale_label(row):
    return row["domain"] if pd.notna(row.get("domain")) and row.get("domain") != "total" else row["experiment"]

for df in (mean_llm, mean_hum):
    df["scale_label"] = df.apply(scale_label, axis=1).replace(LABEL_OVERRIDES)
    df["pct_reversed_pct"] = df["pct_reversed_keyed_items"] * 100


# ══════════════════════════════════════════════════════════════════════════
# JITTER
# ══════════════════════════════════════════════════════════════════════════

rng = np.random.default_rng(RNG_SEED)
mean_llm = mean_llm[np.isfinite(mean_llm["mean_interitem_corr"])].copy()
mean_hum = mean_hum[np.isfinite(mean_hum["mean_interitem_corr"])].copy()

mean_llm["x_j"] = mean_llm["pct_reversed_pct"] + rng.normal(0, JITTER_X, len(mean_llm))
mean_llm["y_j"] = mean_llm["mean_interitem_corr"] + rng.normal(0, JITTER_Y, len(mean_llm))
mean_hum["x_j"] = mean_hum["pct_reversed_pct"] + rng.normal(0, JITTER_X, len(mean_hum))
mean_hum["y_j"] = mean_hum["mean_interitem_corr"] + rng.normal(0, JITTER_Y, len(mean_hum))

# ══════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════

plt.rcParams.update({"font.size": FS_BASE, "axes.titlesize": FS_TITLE, "axes.labelsize": FS_BASE})
fig, ax = plt.subplots(figsize=FIGSIZE)

def _scatter_by_type(ax, df, color, alpha):
    for marker, subset in (
        (MARKER_SURVEY, df[~df["experiment"].isin(BEHAVIOURAL)]),
        (MARKER_BEHAV,  df[df["experiment"].isin(BEHAVIOURAL)]),
    ):
        ax.scatter(subset["x_j"], subset["y_j"], s=POINT_SIZE, alpha=alpha,
                   color=color, lw=LW_POINTS, marker=marker, zorder=4)

_scatter_by_type(ax, mean_llm, C_LLM, ALPHA_LLM)
_scatter_by_type(ax, mean_hum, C_HUMAN, ALPHA_HUMAN)

_plot_regression(ax, mean_llm["pct_reversed_pct"], mean_llm["mean_interitem_corr"],
                 color=C_LLM_LINE, lw=LW_LLM, line_alpha=ALPHA_LLM_LINE)
_plot_regression(ax, mean_hum["pct_reversed_pct"], mean_hum["mean_interitem_corr"],
                 color=C_HUMAN_LINE, lw=LW_HUMAN, line_alpha=ALPHA_HUMAN_LINE)

# ── scale labels on every point ──────────────────────────────────────────
texts_llm = [ax.text(row["x_j"], row["y_j"], row["scale_label"],
                     fontsize=FS_TEXT, color=C_LLM_TEXT, alpha=ALPHA_LLM_TEXT)
             for _, row in mean_llm.iterrows()]
texts_hum = [ax.text(row["x_j"], row["y_j"], row["scale_label"],
                     fontsize=FS_TEXT, color=C_HUMAN_TEXT, alpha=ALPHA_HUMAN_TEXT)
             for _, row in mean_hum.iterrows()]

all_texts = texts_llm + texts_hum
all_x = pd.concat([mean_llm["x_j"], mean_hum["x_j"]]).values
all_y = pd.concat([mean_llm["y_j"], mean_hum["y_j"]]).values

adjust_text(all_texts, x=all_x, y=all_y, ax=ax,
            arrowprops=ARROWPROPS,
            expand=EXPAND, force_text=FORCE_TEXT, force_points=FORCE_POINTS,
            min_arrow_len=MIN_ARROW_LEN)

# ── annotations, labels, legend ──────────────────────────────────────────

r_llm, rho_llm, n_llm = _corr_stats(mean_llm["pct_reversed_pct"], mean_llm["mean_interitem_corr"])
r_hum, rho_hum, n_hum = _corr_stats(mean_hum["pct_reversed_pct"], mean_hum["mean_interitem_corr"])

print(f"LLM: r = {r_llm:+.2f}, $\\rho$ = {rho_llm:+.2f} (N = {n_llm})\n")
print(f"Human: r = {r_hum:+.2f},  $\\rho$ = {rho_hum:+.2f} (N = {n_hum})")

ax.set_xlabel("Proportion of reversed-keyed items (%)", fontsize=FS_LABEL)
ax.set_ylabel("Mean inter-item correlation (within instrument)", fontsize=FS_LABEL)
ax.spines[["top", "right"]].set_visible(False)

ax.legend(handles=[
    Line2D([0], [0], marker="o", color=C_LLM, markerfacecolor=C_LLM, markersize=6, label=LEGEND_LLM_POINTS, linestyle="None"),
    Line2D([0], [0], marker="o", color=C_HUMAN, alpha=ALPHA_HUMAN, markerfacecolor=C_HUMAN, markersize=6, label=LEGEND_HUMAN_POINTS, linestyle="None"),
    Line2D([0], [0], marker=MARKER_SURVEY, color=C_NEUTRAL, markerfacecolor=C_NEUTRAL, markersize=6, label="Self-report scale", linestyle="None"),
    Line2D([0], [0], marker=MARKER_BEHAV, color=C_NEUTRAL, markerfacecolor=C_NEUTRAL, markersize=6, label="Behavioural task", linestyle="None"),
    Line2D([0], [0], color=C_LLM_LINE, lw=LW_LLM, label=LEGEND_LLM_LINE),
    Line2D([0], [0], color=C_HUMAN_LINE, lw=LW_HUMAN, alpha=ALPHA_HUMAN_LINE, label=LEGEND_HUMAN_LINE),
], loc="best", fontsize=FS_LEGEND, framealpha=0.9)

# panel label (figure-relative for consistent placement across figures)
fig.text(0.0, 0.98, "A", fontsize=12, fontweight="bold",
         va="top", ha="left")

plt.tight_layout()
for path in SAVE_PATHS:
    fig.savefig(path, dpi=300, bbox_inches="tight")

plt.close(fig)
print(f"Saved figure to: {SAVE_PATHS[0]}")
