import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import pingouin as pg


# ── instrument / domain name harmonization ──────────────────────────────────

DOSPERT_DOMAIN_NAMES = {
    "Deth": "Ethical", "Dgam": "Gambling", "Dhea": "Health",
    "Dinv": "Investment", "Drec": "Recreational", "Dsoc": "Social",
}

EXPERIMENT_NAME_MAP = {
    "DOSPERT scale": "DOSPERT", "DAST scale": "DAST", "BARRAT scale": "BARRATT",
    "SSSV scale": "SSSV", "GABS scale": "GABS", "CARE scale": "CARE",
    "PRI scale": "PRI", "Dm scale": "DM", "DM scale": "DM", "SOEP scale": "SOEP",
    "LOT task": "LOT", "MPL task": "MPL", "DFE task": "DFE", "DFD task": "DFD",
    "CCT task": "CCT", "BART task": "BART",
}


def normalize_experiment_names(df):
    out = df.copy()
    out["experiment"] = out["experiment"].replace(EXPERIMENT_NAME_MAP)
    return out


def apply_domain_fixes(df, is_llm=False):
    """Rename DOSPERT sub-domains and resolve the PRI domain
    (LLM: total/NaN -> 'decision'; human: drop the 'certainty' rows)."""
    out = df.copy()
    dospert = out["experiment"].isin(["DOSPERT", "DOSPERT scale"])
    if dospert.any():
        out.loc[dospert, "domain"] = out.loc[dospert, "domain"].replace(DOSPERT_DOMAIN_NAMES)
    pri = out["experiment"].isin(["PRI", "PRI scale"])
    if pri.any():
        if is_llm:
            out.loc[pri & (out["domain"].isna() | out["domain"].eq("total")), "domain"] = "decision"
        else:
            out = out[~(pri & out["domain"].eq("certainty"))]
    return out


def compute_pct_reversed_map(df, item_col="item_id"):
    """Proportion of reverse-keyed items per (experiment, domain), folded to <= 0.5."""
    if "reverse_coded" not in df.columns:
        raise ValueError("reverse_coded column missing in data")
    tmp = df[["experiment", "category", item_col, "reverse_coded"]].drop_duplicates()
    tmp = tmp.rename(columns={"category": "domain"})
    tmp["domain"] = tmp["domain"].fillna("total")
    pct = (tmp.groupby(["experiment", "domain"])["reverse_coded"]
              .mean().reset_index(name="pct_reversed_keyed_items"))
    pct["pct_reversed_keyed_items"] = pct["pct_reversed_keyed_items"].where(
        pct["pct_reversed_keyed_items"] <= 0.5, 1 - pct["pct_reversed_keyed_items"])
    return pct


def add_pct_reversed(df, pct_map):
    """Attach pct_reversed_keyed_items by (experiment, domain), falling back to the
    experiment-level value. DOSPERT sub-domains are renamed on the map first
    (idempotent if the map was already fixed)."""
    out = df.copy()
    pct = pct_map.copy()
    dospert = pct["experiment"].eq("DOSPERT")
    if dospert.any():
        pct.loc[dospert, "domain"] = pct.loc[dospert, "domain"].replace(DOSPERT_DOMAIN_NAMES)
    map_domain = dict(zip(zip(pct["experiment"], pct["domain"]), pct["pct_reversed_keyed_items"]))
    map_exp = dict(zip(pct["experiment"], pct["pct_reversed_keyed_items"]))
    out["pct_reversed_keyed_items"] = list(zip(out["experiment"], out["domain"]))
    out["pct_reversed_keyed_items"] = (out["pct_reversed_keyed_items"].map(map_domain)
        .combine_first(out["experiment"].map(map_exp)).fillna(0))
    return out


def pick_item_col(df):
    for c in ("item_id", "item"):
        if c in df.columns:
            return c
    raise ValueError("No known item column found in dataset.")


def pick_individual_col(df):
    for c in ("model", "partid"):
        if c in df.columns:
            return c
    raise ValueError("No known individual column found in dataset.")


# ── reliability / inter-item correlation ─────────────────────────────────────

def _iter_experiment_domains(
    exp_data,
    experiment_name,
    domain_col="category",
    domain_exclude=None,
    total_domain_label="total",
):
    has_domain = exp_data[domain_col].notna().any()
    domain_exclude = set() if domain_exclude is None else set(domain_exclude)
    if has_domain and experiment_name not in domain_exclude:
        for domain, domain_data in exp_data.groupby(domain_col):
            yield domain, domain_data
    else:
        yield total_domain_label, exp_data


def compute_cronbach_alpha(
    data,
    score="score",
    item_name="item_id",
    individual="model",
    experiment_col="experiment",
    domain_col="category",
    domain_exclude=None,
    total_domain_label="total",
):
    if domain_exclude is None:
        domain_exclude = {"SOEP scale", "SOEP"}
    results = []
    for exp, exp_data in data.groupby(experiment_col):
        for domain, domain_data in _iter_experiment_domains(
            exp_data, experiment_name=exp, domain_col=domain_col,
            domain_exclude=domain_exclude, total_domain_label=total_domain_label,
        ):
            df_wide = domain_data.pivot_table(index=individual, columns=item_name, values=score)
            if df_wide.shape[1] > 1:
                alpha, ci = pg.cronbach_alpha(df_wide)
            else:
                alpha, ci = None, (None, None)
            results.append({"experiment": exp, "domain": domain,
                            "k": len(df_wide.columns), "alpha": alpha, "alpha_CI": ci})
    return pd.DataFrame(results)


def mean_interitem_correlation(df_wide):
    """df_wide: rows = respondents/models, columns = items."""
    if df_wide.shape[1] < 2:
        return None
    corr = df_wide.corr()
    iu = np.triu_indices_from(corr, k=1)
    return corr.values[iu].mean()


def compute_mean_interitem_corr(
    data,
    score="model_answer",
    item_name="item_id",
    individual="model",
    experiment_col="experiment",
    domain_col="category",
    domain_exclude=None,
    total_domain_label="total",
):
    if domain_exclude is None:
        domain_exclude = {"SOEP scale", "SOEP"}
    results = []
    for exp, exp_data in data.groupby(experiment_col):
        for domain, domain_data in _iter_experiment_domains(
            exp_data, experiment_name=exp, domain_col=domain_col,
            domain_exclude=domain_exclude, total_domain_label=total_domain_label,
        ):
            df_wide = domain_data.pivot_table(index=individual, columns=item_name, values=score)
            results.append({"experiment": exp, "domain": domain,
                            "k": len(df_wide.columns),
                            "mean_interitem_corr": mean_interitem_correlation(df_wide)})
    return pd.DataFrame(results)


def mean_interitem_corr_ci(df_wide, n_boot=500, random_state=13):
    """Mean inter-item correlation with a respondent-level bootstrap 95% CI."""
    mean_corr = mean_interitem_correlation(df_wide)
    if mean_corr is None:
        return None, None, None
    n = df_wide.shape[0]
    if n < 2:
        return mean_corr, None, None
    rng = np.random.default_rng(random_state)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = mean_interitem_correlation(df_wide.iloc[idx])
        if m is not None and np.isfinite(m):
            boot.append(m)
    if len(boot) == 0:
        return mean_corr, None, None
    low, high = np.quantile(boot, [0.025, 0.975])
    return mean_corr, low, high


def compute_mean_interitem_corr_with_ci(
    data,
    score="model_answer",
    item_name="item_id",
    individual="model",
    experiment_col="experiment",
    domain_col="category",
    domain_exclude=None,
    total_domain_label="total",
    n_boot=500,
    random_state=13,
):
    if domain_exclude is None:
        domain_exclude = {"SOEP scale", "SOEP"}
    results = []
    for exp, exp_data in data.groupby(experiment_col):
        has_domain = exp_data[domain_col].notna().any()
        if has_domain and exp not in domain_exclude:
            domain_groups = exp_data.groupby(domain_col)
        else:
            domain_groups = [(total_domain_label, exp_data)]
        for domain, domain_data in domain_groups:
            df_wide = domain_data.pivot_table(index=individual, columns=item_name, values=score)
            mean_corr, ci_low, ci_high = mean_interitem_corr_ci(
                df_wide, n_boot=n_boot, random_state=random_state)
            results.append({"experiment": exp, "domain": domain, "k": len(df_wide.columns),
                            "mean_interitem_corr": mean_corr,
                            "mean_interitem_corr_ci_low": ci_low,
                            "mean_interitem_corr_ci_high": ci_high})
    return pd.DataFrame(results)


# ── LaTeX formatting for the alpha / inter-item tables ───────────────────────

def fmt_number(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if abs(x) < 0.005:
        x = 0.0
    return f"{x:.2f}"


def fmt_value_ci(value, ci_low, ci_high):
    v = fmt_number(value)
    if v is None:
        return ""
    lo, hi = fmt_number(ci_low), fmt_number(ci_high)
    if lo is None or hi is None:
        return v
    return f"\\begin{{tabular}}{{c}}{v} \\\\ {{[{lo}, {hi}]}}\\end{{tabular}}"


def format_cronbach_latex(df, fmt_k=False):
    """Format an alpha / mean-inter-item table (LLM + human columns) for LaTeX."""
    out = df.copy()
    for val, lo, hi in [
        ("avg_interitem_corr_llm", "avg_interitem_corr_llm_ci_low", "avg_interitem_corr_llm_ci_high"),
        ("cronbach_alpha_llm", "alpha_ci_low_llm", "alpha_ci_high_llm"),
        ("avg_interitem_corr_humans", "avg_interitem_corr_humans_ci_low", "avg_interitem_corr_humans_ci_high"),
        ("cronbach_alpha_humans", "alpha_ci_low_humans", "alpha_ci_high_humans"),
    ]:
        out[val] = out.apply(lambda r, v=val, l=lo, h=hi: fmt_value_ci(r[v], r[l], r[h]), axis=1)
    out["pct_reversed_keyed_items"] = out["pct_reversed_keyed_items"].map(
        lambda x: fmt_number(x) if pd.notna(x) else "")
    if fmt_k:
        out["k"] = out["k"].map(lambda x: fmt_number(x) if pd.notna(x) else "")
    out = out.rename(columns={
        "pct_reversed_keyed_items": "pct rev. items",
        "avg_interitem_corr_llm": "avg. inter-item-corr LLM",
        "cronbach_alpha_llm": "$\\alpha$ LLM",
        "avg_interitem_corr_humans": "avg. inter-item-corr humans",
        "cronbach_alpha_humans": "$\\alpha$ humans",
    })
    return out[["experiment", "domain", "k", "pct rev. items",
                "avg. inter-item-corr LLM", "$\\alpha$ LLM",
                "avg. inter-item-corr humans", "$\\alpha$ humans"]]
