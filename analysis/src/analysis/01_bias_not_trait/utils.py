# packages 
import pandas as pd
import os
import re
import glob
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# -----------------------------------Helper for analysis------------------------------------------------

# function for vis:
def get_corrs_and_vis_points(data, entity_var, dependent_var):
    # calculate participants' means, divided by reversed vs. forward items
    divided_mean_df = (
        data
        .groupby([entity_var, "reverse_coded"], as_index=False)[dependent_var] 
        .mean()
    )

    divided_mean_df = (
        divided_mean_df
        .pivot(index=entity_var, columns="reverse_coded", values=dependent_var)
        .reset_index()
    )

    x = divided_mean_df[False]
    y = divided_mean_df[True]
    r, p = stats.pearsonr(x, y)
    return x, y, r


def pearsonr_ci(x, y, alpha=0.05):
    x, y = pd.Series(x), pd.Series(y)
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 2:
        return float("nan"), (float("nan"), float("nan")), n
    r, _ = stats.pearsonr(x, y)
    if n < 4:
        return r, (float("nan"), float("nan")), n
    z    = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se   = 1 / np.sqrt(n - 3)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    lo, hi = np.tanh(z - zcrit * se), np.tanh(z + zcrit * se)
    return r, (lo, hi), n


def forward_reverse_means(data, entity_var, dependent_var):
    divided = (
        data.groupby([entity_var, "reverse_coded"], as_index=False)[dependent_var]
        .mean()
        .pivot(index=entity_var, columns="reverse_coded", values=dependent_var)
        .reset_index()
    )
    return divided[False], divided[True]


def plot_corr_panel(ax, data, entity_var, dependent_var, color, FS_STATS,
                    point_alpha=0.7, point_size=40, show_stats=True):
    x_raw, y_raw = forward_reverse_means(data, entity_var, dependent_var)
    r, (lo, hi), n = pearsonr_ci(x_raw, y_raw)

    x_arr = np.asarray(x_raw, dtype=float)
    y_arr = np.asarray(y_raw, dtype=float)
    mask  = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[mask], y_arr[mask]

    ax.scatter(x_arr, y_arr, alpha=point_alpha, s=point_size,
               c=color, edgecolors="white", linewidth=0.3)

    if len(x_arr) >= 3:
        slope, intercept, *_ = stats.linregress(x_arr, y_arr)
        x_grid  = np.linspace(1, 5, 300)
        y_hat   = intercept + slope * x_grid
        y_fit   = intercept + slope * x_arr
        s_err   = np.sqrt(np.sum((y_arr - y_fit) ** 2) / (len(x_arr) - 2))
        x_mean  = x_arr.mean()
        sxx     = np.sum((x_arr - x_mean) ** 2)
        tcrit   = stats.t.ppf(0.975, df=len(x_arr) - 2)
        se_band = s_err * np.sqrt(1 / len(x_arr) + (x_grid - x_mean) ** 2 / sxx)
        ax.plot(x_grid, y_hat, color=color, linewidth=1.8, zorder=3)
        # ax.fill_between(x_grid, y_hat - tcrit * se_band, y_hat + tcrit * se_band,
        #                 color=color, alpha=0.15, zorder=2)

    if show_stats:
        sign = "+" if r > 0 else ""
        ax.text(0.05, 0.95,
                f"$r$ = {sign}{r:.2f} [{lo:.2f}, {hi:.2f}]\n$N$ = {n}",
                transform=ax.transAxes, va="top", ha="left", fontsize=FS_STATS,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.8))

    ax.set_xlim(0.75, 5.25)
    ax.set_ylim(0.75, 5.25)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([1, 2, 3, 4, 5])


def get_corrs_and_ci(data, entity_var, dependent_var, alpha=0.05):
    # calculate participants' means, divided by reversed vs. forward items
    divided_mean_df = (
        data
        .groupby([entity_var, "reverse_coded"], as_index=False)[dependent_var]
        .mean()
    )

    divided_mean_df = (
        divided_mean_df
        .pivot(index=entity_var, columns="reverse_coded", values=dependent_var)
        .reset_index()
    )

    x = divided_mean_df[False]
    y = divided_mean_df[True]
    r, (lo, hi), n = pearsonr_ci(x, y, alpha=alpha)
    return r, lo, hi, n


def format_r_ci(r, lo, hi, decimals=2, multiline=True):
    if pd.isna(r):
        return ""
    if pd.isna(lo) or pd.isna(hi):
        return f"{r:.{decimals}f}"
    return f"{r:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"

def fisher_r_to_z_test(r1, n1, r2, n2):
    if any(pd.isna(x) for x in [r1, n1, r2, n2]):
        return float("nan"), float("nan"), float("nan")
    if n1 < 4 or n2 < 4:
        return float("nan"), float("nan"), float("nan")
    r1c = np.clip(r1, -0.999999, 0.999999)
    r2c = np.clip(r2, -0.999999, 0.999999)
    z1 = np.arctanh(r1c)
    z2 = np.arctanh(r2c)
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    if se == 0:
        return float("nan"), float("nan"), float("nan")
    delta_z = z1 - z2
    z_stat = delta_z / se
    p = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return delta_z, z_stat, p


def format_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def format_number(value, decimals=2):
    if pd.isna(value):
        return ""
    return f"{value:.{decimals}f}"


def get_corrs_by_traits(
    data,
    table_name,
    entity_var,
    dependent_var,
    alpha=0.05,
    decimals=2,
    multiline_ci=True,
    category = "traits"
):
    corr_dict = {}
    corr_raw = {}
    for trait in data[category].unique():
        subset_data = data[data[category] == trait]
        r, lo, hi, n = get_corrs_and_ci(subset_data, entity_var, dependent_var, alpha=alpha)
        corr_dict[trait] = format_r_ci(r, lo, hi, decimals=decimals, multiline=multiline_ci)
        corr_raw[trait] = {"r": r, "n": n}

    return corr_dict, corr_raw


# build a dataframe with rows = conditions and columns = traits
def corrs_table_from_conditions(
    conditions,
    entity_var=None,
    dependent_var=None,
    condition_col="condition",
    trait_order=None,
    alpha=0.05,
    decimals=2,
    multiline_ci=True,
    compare_to=None,
    add_diff_rows=False,
    category = "traits"
):
    # baseline correlations for Fisher r-to-z comparison
    baseline_raw = None
    if compare_to is not None:
        baseline_item = None
        for item in conditions:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                cond_name = item[0]
                cond_df = item[1]
                cond_entity_var = entity_var
                cond_dependent_var = dependent_var
            else:
                cond_name = item.get("name")
                cond_df = item.get("df")
                cond_entity_var = item.get("entity_var", entity_var)
                cond_dependent_var = item.get("dependent_var", dependent_var)

            if cond_name == compare_to:
                baseline_item = {
                    "df": cond_df,
                    "entity_var": cond_entity_var,
                    "dependent_var": cond_dependent_var,
                }
                break

        if baseline_item is None:
            raise ValueError(f"compare_to='{compare_to}' not found in conditions.")

        _, baseline_raw = get_corrs_by_traits(
            baseline_item["df"],
            compare_to,
            baseline_item["entity_var"],
            baseline_item["dependent_var"],
            alpha=alpha,
            decimals=decimals,
            multiline_ci=multiline_ci,
            category = category
        )

    rows = []
    for item in conditions:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            condition_name, df = item
            row_entity_var = entity_var
            row_dependent_var = dependent_var
        else:
            condition_name = item["name"]
            df = item["df"]
            row_entity_var = item.get("entity_var", entity_var)
            row_dependent_var = item.get("dependent_var", dependent_var)

        if row_entity_var is None or row_dependent_var is None:
            raise ValueError(
                "entity_var and dependent_var must be provided either globally or per condition."
            )

        corr_dict, corr_raw = get_corrs_by_traits(
            df,
            condition_name,
            row_entity_var,
            row_dependent_var,
            alpha=alpha,
            decimals=decimals,
            multiline_ci=multiline_ci,
            category=category
        )
        row = {condition_col: condition_name, "n": df.loc[df[row_dependent_var].notna(), row_entity_var].nunique(), **corr_dict}
        rows.append(row)

        if add_diff_rows and compare_to is not None and condition_name != compare_to:
            use_short_labels = len(conditions) == 2
            if use_short_labels:
                delta_label = r"$\Delta z'$"
                z_label = "z-statistic"
            else:
                delta_label = f"$\\Delta z'$ ({condition_name}-{compare_to})"
                z_label = f"z-statistic ({condition_name}-{compare_to})"

            delta_row = {condition_col: delta_label}
            z_row = {condition_col: z_label}

            for trait in corr_dict.keys():
                base = baseline_raw.get(trait, {"r": float("nan"), "n": float("nan")})
                curr = corr_raw.get(trait, {"r": float("nan"), "n": float("nan")})
                delta_z, z_stat, p = fisher_r_to_z_test(
                    curr["r"], curr["n"], base["r"], base["n"]
                )
                delta_row[trait] = f"{format_number(delta_z, decimals=decimals)}{format_stars(p)}"
                z_row[trait] = format_number(z_stat, decimals=decimals)

            rows.append(delta_row)
            rows.append(z_row)

    table_df = pd.DataFrame(rows)

    if trait_order is None:
        trait_order = [t for t in ["O", "C", "E", "A", "N"] if t in table_df.columns]

    ordered_cols = [condition_col, "n"] + trait_order
    table_df = table_df.reindex(columns=ordered_cols)

    return table_df


def to_latex_table(
    df,
    float_format="%.2f",
    index=False,
    caption=None,
    label=None,
    escape=False,
):
    return df.to_latex(
        index=index,
        float_format=float_format,
        caption=caption,
        label=label,
        escape=escape,
    )


def clean_model_label(name: str) -> str:
    cleaned = name
    for token in ["-Instruct-2509", "-Instruct", "-it", "-IT", "-instruct", "-0309-non-reasoning", "-2026-02-23", "-2026-02-15", "anthropic__", "openai__", "x-ai__", "google__", "qwen__", "-preview", "__nr", "__r", "-0309-reasoning"]:
        cleaned = cleaned.replace(token, "")
    return cleaned


def strip_leading_zero(num_str):
    if num_str.startswith("-0."):
        return "-" + num_str[2:]
    if num_str.startswith("0."):
        return num_str[1:]
    return num_str


def format_ci_cell(cell, decimals=2):
    """Turn a "r [lo, hi]" string cell into a two-line LaTeX mini-table
    (point estimate above, CI below, leading zeros stripped from the CI)."""
    if not isinstance(cell, str) or "[" not in cell:
        return cell
    m = re.match(r"^\s*([+-]?\d*\.?\d+)\s*\[([+-]?\d*\.?\d+),\s*([+-]?\d*\.?\d+)\]\s*$", cell)
    if m is None:
        return cell
    r, lo, hi = float(m.group(1)), float(m.group(2)), float(m.group(3))
    r_str = f"{r:.{decimals}f}"
    lo_str = strip_leading_zero(f"{lo:.{decimals}f}")
    hi_str = strip_leading_zero(f"{hi:.{decimals}f}")
    return f"\\begin{{tabular}}{{c}}{r_str} \\\\ {{[{lo_str}, {hi_str}]}}\\end{{tabular}}"