"""Fig 3c: distribution of mean inter-item correlation across instruments in the
0-10% vs 40-50% reverse-keyed bins, per prompting condition (ridgeline + means).

Reads risk_data + ipipneo300_data (LLM/api/human, all conditions).
Writes results/figures/orthogonality_interitem_extremes.pdf.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from utils import (
    compute_mean_interitem_corr,
    normalize_experiment_names,
    apply_domain_fixes as _apply_domain_fixes,
    compute_pct_reversed_map,
    add_pct_reversed,
    pick_item_col,
    pick_individual_col,
)

DATA_DIR = (BASE_DIR / "../../../data/intermediate/risk_data").resolve()
IPIP_DIR = (BASE_DIR / "../../../data/intermediate/ipipneo300_data").resolve()

SAVE_PATHS = [
    BASE_DIR / "../../../results/figures/orthogonality_interitem_extremes.pdf",
]

# colors
C_LLM = "#1e847f"
C_HUMAN = "#440154"

# markers
MARKER_LOW = "D"
MARKER_HIGH = "s"
MARKERSIZE = 6

# alpha
ALPHA_LLM = 0.95
ALPHA_HUMAN = 0.35

# typography
FS_YTICKS = 13
FS_XLABEL = 13
FS_LEGEND = 10

# layout (matches script 05 so the two figures align side-by-side in the paper)
ROW_HEIGHT = 0.58
FIG_EXTRA = 0.6

# ridgeline density curves
DENSITY_HEIGHT = 0.36       # max half-height of each density curve (in row units)
DENSITY_BW = 0.4            # gaussian_kde bandwidth scale (relative to Scott's rule)
RUG_HEIGHT = 0.06           # rug-tick height per data point
DENSITY_FILL_ALPHA = 0.35
TAPER_FRAC = 0.10           # fraction of curve length to taper to 0 at each end
MARKER_Y_OFFSET = 0.12      # mean marker offset into the density area (row units)
N_LABEL_FS = 8              # fontsize for "n=…" annotations
N_LABEL_OFFSET_Y = 0.06     # vertical offset for "n=…" annotation (row units)
N_LABEL_ALPHA = 0.9         # opacity for "n=…" annotation

# separator between LLM and Human rows
SEP_COLOR = "gray"
SEP_LW = 0.7
SEP_LS = ":"

# reversed-keying bins (proportions)
LOW_BIN = (0.00, 0.10)
HIGH_BIN = (0.40, 0.50)

def make_label(condition, context, score):
    ctx = "ctx" if context == "with_context" else "no ctx"
    if condition == "Humans":
        return "Humans*"
    score_label = "textgen" if score == "model_answer" else "logit"
    condition_short = (
        condition.replace("LLM direct", "LLM d.")
        .replace("LLM weighed", "LLM w.")
        .replace(" (no flip)", "")
        .replace(" (flip)", "")
    )
    flip = "nf" if "no flip" in condition else "f"
    label = f"{condition_short}  [{ctx}, {score_label}, {flip}]"
    if "LLM d." in condition_short:
        label = f"{label}*"
    return label


def mean_range(values):
    vals = np.asarray(values, float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(np.mean(vals))
    return mean, float(np.min(vals)), float(np.max(vals)), n


def summarize_condition_bin(df, score_col, pct_map, context_mode, condition_label):
    data = df.copy()
    if "context_mode" in data.columns:
        data = data[data["context_mode"] == context_mode].copy()

    item_col = pick_item_col(data)
    individual_col = pick_individual_col(data)

    mean_corr = compute_mean_interitem_corr(
        data,
        score=score_col,
        item_name=item_col,
        individual=individual_col,
    )
    mean_corr = _apply_domain_fixes(mean_corr, is_llm=(individual_col == "model"))
    mean_corr = add_pct_reversed(mean_corr, pct_map)
    mean_corr = mean_corr.dropna(subset=["mean_interitem_corr", "pct_reversed_keyed_items"])

    low_mask = (
        mean_corr["pct_reversed_keyed_items"] >= LOW_BIN[0]
    ) & (
        mean_corr["pct_reversed_keyed_items"] <= LOW_BIN[1]
    )
    high_mask = (
        mean_corr["pct_reversed_keyed_items"] >= HIGH_BIN[0]
    ) & (
        mean_corr["pct_reversed_keyed_items"] <= HIGH_BIN[1]
    )

    out_rows = []
    for band, mask in [("0-10% reversed", low_mask), ("40-50% reversed", high_mask)]:
        values = mean_corr.loc[mask, "mean_interitem_corr"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        mean, lo, hi, n = mean_range(values)
        out_rows.append(
            {
                "condition": condition_label,
                "band": band,
                "mean": mean,
                "lo": lo,
                "hi": hi,
                "n_subscales": n,
                "values": values,
            }
        )
    return out_rows


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

risk_data_LLM_direct_prompting = pd.read_csv(
    DATA_DIR / "LLM_data_proc_prompts_direct/LLM_no_flip_data_rereversed.csv",
    low_memory=False,
)
risk_data_flip_LLM_direct_prompting = pd.read_csv(
    DATA_DIR / "LLM_data_proc_prompts_direct/LLM_flip_data_rereversed.csv",
    low_memory=False,
)
risk_data_LLM_weighed_prompting = pd.read_csv(
    DATA_DIR / "LLM_data_proc_prompts_weighed/LLM_weighed_no_flip_data_rereversed.csv",
    low_memory=False,
)
risk_data_flip_LLM_weighed_prompting = pd.read_csv(
    DATA_DIR / "LLM_data_proc_prompts_weighed/LLM_weighed_flip_data_rereversed.csv",
    low_memory=False,
)
risk_data_humans = pd.read_csv(
    DATA_DIR / "human_data_proc/items_per_person.csv",
    low_memory=False,
)
risk_data_humans["experiment"] = risk_data_humans["experiment"].replace({"DM scale": "Dm scale"})

ipip_llm = pd.read_csv(
    IPIP_DIR / "llm_data/ipipneo_all_data_reflipped_and_rereversed.csv",
    low_memory=False,
)
ipip_llm = ipip_llm.copy()
ipip_llm["category"] = ipip_llm["traits"]

ipip_hum = pd.read_csv(
    IPIP_DIR / "human_data/ipipneo_human.csv",
    low_memory=False,
)
ipip_hum["response"] = pd.to_numeric(
    ipip_hum["response"].replace({"0": None, 0: None, " ": None})
)
ipip_hum = ipip_hum.dropna(subset=["response"]).copy()
ipip_hum["experiment"] = "IPIP-NEO-300"
ipip_hum["category"] = ipip_hum["traits"]
ipip_hum = ipip_hum.rename(columns={"response": "score", "person_id": "partid"})

# Proprietary API models (risk + IPIP), no-reasoning (__nr) / no_context /
# no-flip — the only slice the API runs collected. Added ONLY to the headline
# condition below (direct / no flip / no context / textgen).
risk_api_direct = pd.read_csv(
    DATA_DIR / "api_data/LLM_api_no_flip_data_rereversed.csv", low_memory=False
)
risk_api_direct = risk_api_direct[
    risk_api_direct["model"].str.endswith("__nr")
    & (risk_api_direct["context_mode"] == "no_context")
    & (risk_api_direct["flipped"] == False)
].copy()

ipip_api = pd.read_csv(
    IPIP_DIR / "api_data/ipipneo_api_data_rereversed.csv", low_memory=False
)
ipip_api = ipip_api[
    ipip_api["model"].str.endswith("__nr")
    & (ipip_api["context_mode"] == "no_context")
    & (ipip_api["flipped"] == False)
].copy()
ipip_api["category"] = ipip_api["traits"]


# -----------------------------------------------------------------------------
# Name harmonization
# -----------------------------------------------------------------------------

risk_data_LLM_direct_prompting = normalize_experiment_names(risk_data_LLM_direct_prompting)
risk_data_flip_LLM_direct_prompting = normalize_experiment_names(risk_data_flip_LLM_direct_prompting)
risk_data_LLM_weighed_prompting = normalize_experiment_names(risk_data_LLM_weighed_prompting)
risk_data_flip_LLM_weighed_prompting = normalize_experiment_names(risk_data_flip_LLM_weighed_prompting)
risk_data_humans = normalize_experiment_names(risk_data_humans)
ipip_llm = normalize_experiment_names(ipip_llm)
ipip_hum = normalize_experiment_names(ipip_hum)
risk_api_direct = normalize_experiment_names(risk_api_direct)
ipip_api = normalize_experiment_names(ipip_api)


# -----------------------------------------------------------------------------
# Reversed-keying maps
# -----------------------------------------------------------------------------

pct_map_direct = compute_pct_reversed_map(
    pd.concat([risk_data_LLM_direct_prompting, ipip_llm], ignore_index=True),
    item_col="item_id",
)
pct_map_direct = _apply_domain_fixes(pct_map_direct, is_llm=True)

pct_map_flip_direct = compute_pct_reversed_map(
    pd.concat([risk_data_flip_LLM_direct_prompting, ipip_llm], ignore_index=True),
    item_col="item_id",
)
pct_map_flip_direct = _apply_domain_fixes(pct_map_flip_direct, is_llm=True)

pct_map_weighed = compute_pct_reversed_map(risk_data_LLM_weighed_prompting, item_col="item")
pct_map_weighed = _apply_domain_fixes(pct_map_weighed, is_llm=True)

pct_map_weighed_flip = compute_pct_reversed_map(
    risk_data_flip_LLM_weighed_prompting, item_col="item"
)
pct_map_weighed_flip = _apply_domain_fixes(pct_map_weighed_flip, is_llm=True)

humans_with_ipip = pd.concat([risk_data_humans, ipip_hum], ignore_index=True)
pct_map_humans = _apply_domain_fixes(pct_map_direct, is_llm=False).copy()
# LOT presentation order was randomised per trial for humans, so the LLM-instrument
# p_r (0.40 after folding) doesn't reflect what humans experienced (0.50 by design).
# Override the LOT entry on the humans-only map.
pct_map_humans.loc[
    pct_map_humans["experiment"] == "LOT", "pct_reversed_keyed_items"
] = 0.5


# -----------------------------------------------------------------------------
# Conditions (same ordering as generalisation table/figure)
# -----------------------------------------------------------------------------

conditions = []
for ctx in ["no_context", "with_context"]:
    for score_col in ["model_answer", "logit_score"]:
        ipip_ctx = ipip_llm[ipip_llm["context_mode"] == ctx]
        direct_noflip_parts = [
            risk_data_LLM_direct_prompting,
            ipip_ctx[ipip_ctx["flipped"] == False],
        ]
        # Headline condition only: add the 10 proprietary API models to
        # direct / no flip / no context / textgen — the single condition the
        # API runs cover.
        if ctx == "no_context" and score_col == "model_answer":
            direct_noflip_parts += [risk_api_direct, ipip_api]
        conditions.append(
            (
                "LLM direct (no flip)",
                pd.concat(direct_noflip_parts, ignore_index=True),
                ctx,
                score_col,
                pct_map_direct,
            )
        )
        conditions.append(
            (
                "LLM direct (flip)",
                pd.concat(
                    [risk_data_flip_LLM_direct_prompting, ipip_ctx[ipip_ctx["flipped"] == True]],
                    ignore_index=True,
                ),
                ctx,
                score_col,
                pct_map_flip_direct,
            )
        )

conditions.append(
    (
        "LLM weighed (no flip)",
        risk_data_LLM_weighed_prompting,
        "with_context",
        "score",
        pct_map_weighed,
    )
)
conditions.append(
    (
        "LLM weighed (flip)",
        risk_data_flip_LLM_weighed_prompting,
        "with_context",
        "score",
        pct_map_weighed_flip,
    )
)
conditions.append(("Humans", humans_with_ipip, "with_context", "score", pct_map_humans))


# -----------------------------------------------------------------------------
# Summaries by condition and reversed-keying band
# -----------------------------------------------------------------------------

rows = []
labels_in_order = []
for condition, data, ctx, score_col, pct_map in conditions:
    label = make_label(condition=condition, context=ctx, score=score_col)
    labels_in_order.append(label)
    rows.extend(
        summarize_condition_bin(
            df=data,
            score_col=score_col,
            pct_map=pct_map,
            context_mode=ctx,
            condition_label=label,
        )
    )

plot_df = pd.DataFrame(rows)
print(plot_df[["condition", "band", "n_subscales"]].to_string(index=False))
label_order = list(dict.fromkeys(labels_in_order))
y_pos = {lbl: i for i, lbl in enumerate(label_order)}


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, len(label_order) * ROW_HEIGHT + FIG_EXTRA))

for _, row in plot_df.iterrows():
    n = row["n_subscales"]
    if n == 0 or not np.isfinite(row["mean"]):
        continue

    is_human = "Humans" in row["condition"]
    color = C_HUMAN if is_human else C_LLM
    alpha = ALPHA_HUMAN if is_human else ALPHA_LLM

    base_y = y_pos[row["condition"]]
    if row["band"] == "0-10% reversed":
        marker = MARKER_LOW
        density_dir = +1   # density grows visually downward (invert_yaxis on)
    else:
        marker = MARKER_HIGH
        density_dir = -1   # density grows visually upward

    values = row["values"]

    # Ridgeline KDE — clipped to [min, max] with a smooth taper at each end
    # so the curve meets the baseline at the actual data extremes without the
    # vertical "cut" you'd get from clipping a Gaussian KDE directly.
    if n >= 2 and float(np.std(values)) > 1e-9:
        try:
            kde = gaussian_kde(values, bw_method=DENSITY_BW)
            xs = np.linspace(values.min(), values.max(), 256)
            ys = kde(xs)
            t = np.linspace(0, 1, len(xs))
            window = np.minimum(np.minimum(t / TAPER_FRAC,
                                            (1 - t) / TAPER_FRAC), 1.0)
            ys = ys * window
            ymax = ys.max()
            if ymax > 0:
                ys = ys / ymax * DENSITY_HEIGHT
            top = base_y + density_dir * ys
            ax.fill_between(xs, base_y, top, color=color,
                            alpha=alpha * DENSITY_FILL_ALPHA, linewidth=0)
            ax.plot(xs, top, color=color, alpha=alpha, lw=0.9)
        except np.linalg.LinAlgError:
            pass

    # Rug ticks — one short tick per scale.
    for v in values:
        ax.plot([v, v], [base_y, base_y + density_dir * RUG_HEIGHT],
                color=color, alpha=alpha * 0.8, lw=0.7)

    # Mean marker, offset into the density area so the two bands are
    # visually separated even when their x-ranges overlap (e.g. Humans).
    ax.plot(row["mean"], base_y + density_dir * MARKER_Y_OFFSET,
            marker=marker, color=color, markersize=MARKERSIZE, alpha=alpha,
            linestyle="None", markeredgewidth=0.5)

    # n label at the right edge of the density, on the band's side.
    ax.annotate(
        f"n={n}",
        xy=(values.max(), base_y + density_dir * N_LABEL_OFFSET_Y),
        xytext=(3, 0), textcoords="offset points",
        fontsize=N_LABEL_FS, color=color, alpha=N_LABEL_ALPHA,
        va="center", ha="left",
    )

human_labels = [l for l in label_order if "Humans" in l]
if human_labels:
    ax.axhline(
        y_pos[human_labels[0]] - 0.5,
        color=SEP_COLOR,
        linewidth=SEP_LW,
        linestyle=SEP_LS,
    )

# Use the actual data limits (which include the KDE tails) for x-axis,
# then add a small pad. y-axis is left untouched.
ax.relim()
ax.autoscale_view(scalex=True, scaley=False)
xmin, xmax = ax.get_xlim()
span = xmax - xmin
pad = 0.04 * span if span > 0 else 0.05
ax.set_xlim(xmin - pad, xmax + pad)

# add horizontal row discriminator lines    
for i in range(len(label_order)):
    if i % 2 == 0:
        ax.axhspan(
            i - 0.5,
            i + 0.5,
            color="grey",
            alpha=0.06,
            zorder=0
        )
ax.grid(axis="x", alpha=0.2)
ax.set_yticks([])
#ax.set_yticklabels(label_order, fontsize=FS_YTICKS)
ax.invert_yaxis()
ax.set_xlabel("Mean inter-item correlation (min-max across subscales)", fontsize=FS_XLABEL)
ax.spines[["top", "right"]].set_visible(False)

legend_handles = [
    Line2D([0], [0], marker=MARKER_LOW, color="black", linestyle="None", markersize=7, label="Tasks with: 0-10% reversed"),
    Line2D([0], [0], marker=MARKER_HIGH, color="black", linestyle="None", markersize=7, label="Tasks with: 40-50% reversed"),
]
ax.legend(handles=legend_handles, fontsize=FS_LEGEND, loc="lower right", framealpha=0.9)

fig.text(-0.02, 0.98, "C", fontsize=17, fontweight="bold", va="top", ha="left")

plt.tight_layout()
for path in SAVE_PATHS:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    except PermissionError:
        print(f"Skipping save (no permission): {path}")
plt.close(fig)

print(f"Saved figure to: {SAVE_PATHS[0]}")
