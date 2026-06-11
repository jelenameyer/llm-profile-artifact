"""SI slope-based validation of the formal measurement model (Table tab:bis_slope_validation).

Compares the model-predicted OLS slope of recoded T on raw R (from rho and p_r)
with the empirical slope for BISa/BISm/BISn x {human, LLM}, with 1,000-replicate
person-level bootstrap CIs.

Reads risk_data (LLM/api no-flip raw, human raw_items_per_person).
Writes results/tables/bias_slope_validation_barratt.csv (and prints the table).
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

LLM_PATH   = "../../../data/intermediate/risk_data/LLM_data_proc_prompts_direct/LLM_no_flip_data_raw.csv"
API_PATH   = "../../../data/intermediate/risk_data/api_data/LLM_api_no_flip_data_raw.csv"
HUMAN_PATH = "../../../data/intermediate/risk_data/human_data_proc/raw_items_per_person.csv"

SCALE_MIN, SCALE_MAX = 1, 4          # BARRATT (BIS-11): 1-4 Likert
M, FLIP = (SCALE_MIN + SCALE_MAX) / 2, SCALE_MIN + SCALE_MAX
BIS_CELLS = ["BISa", "BISm", "BISn"]
N_BOOT, SEED = 1000, 42

# harmonize human experiment names to the LLM codes (from 06_..._risk_supp.py)
EXPERIMENT_MAP = {"BARRAT scale": "BARRATT", "BARRATT scale": "BARRATT"}


def _norm_item_id(x):
    if pd.isna(x):
        return None
    s = str(x)
    return s[:-2] if s.endswith(".0") else s


def _risk_trait(df):
    return df["category"].where(df["category"].notna() & (df["category"] != ""), df["experiment"])


# ---------------------------------------------------------------- load + harmonize
llm = pd.read_csv(LLM_PATH, low_memory=False)
api = pd.read_csv(API_PATH, low_memory=False)
# risk API runs cover only the no-reasoning, no-context, no-flip slice
api = api[api["model"].str.endswith("__nr")
          & (api["context_mode"] == "no_context")
          & (api["flipped"] == False)]
llm = pd.concat([llm, api], ignore_index=True)
llm = llm[llm["context_mode"] == "no_context"].copy()
llm["item_id"] = llm["item_id"].map(_norm_item_id)
llm["risk_trait"] = _risk_trait(llm)

hum = pd.read_csv(HUMAN_PATH, low_memory=False)
# binary risk tasks use 0 as a valid response, so only " " is a missing sentinel
hum["score"] = pd.to_numeric(hum["score"].replace({" ": None}), errors="coerce")
hum = hum.dropna(subset=["score"]).copy()
hum["experiment"] = hum["experiment"].map(EXPERIMENT_MAP).fillna(hum["experiment"])
hum["item_id"] = hum["item"].map(_norm_item_id)
# humans inherit the keying (reverse_coded) from the LLM item map
keying = (llm[["experiment", "item_id", "reverse_coded"]]
          .dropna(subset=["reverse_coded"]).drop_duplicates(["experiment", "item_id"]))
hum = hum.merge(keying, on=["experiment", "item_id"], how="left")
hum["risk_trait"] = _risk_trait(hum)


# ---------------------------------------------------------------- model + bootstrap
def cell_person_means(df, cell, person_col, response_col):
    """Per-respondent forward and reverse item means for one sub-scale, plus p_r."""
    sub = df[df["risk_trait"] == cell]
    rev_flag = sub["reverse_coded"].fillna(False)
    n_f = sub.loc[~rev_flag, "item_id"].nunique()
    n_r = sub.loc[rev_flag, "item_id"].nunique()
    p_r = n_r / (n_f + n_r)
    R_f = sub[~rev_flag].groupby(person_col)[response_col].mean()
    R_r = sub[rev_flag].groupby(person_col)[response_col].mean()
    common = R_f.index.intersection(R_r.index)
    return R_f.loc[common].values, R_r.loc[common].values, p_r


def slope_stats(R_f, R_r, p_r):
    """rho, pi_b, predicted slope, and empirical OLS slope of T on R."""
    rho = float(np.corrcoef(R_f, R_r)[0, 1])
    R = (1 - p_r) * R_f + p_r * R_r
    T = (1 - p_r) * R_f + p_r * (FLIP - R_r)
    coef = 1 - 2 * p_r
    beta_pred = 2 * coef / (coef ** 2 * (1 - rho) + (1 + rho))
    beta_emp = float(np.cov(R, T, ddof=1)[0, 1] / np.var(R, ddof=1))
    return rho, (1 + rho) / 2, beta_pred, beta_emp


def bootstrap_ci(R_f, R_r, p_r):
    """95% person-level percentile-bootstrap CIs for beta_pred and beta_emp."""
    rng = np.random.default_rng(SEED)
    n = len(R_f)
    bp, be = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        rf, rr = R_f[idx], R_r[idx]
        if np.var(rf) == 0 or np.var(rr) == 0:
            continue
        _, _, b_pred, b_emp = slope_stats(rf, rr, p_r)
        bp.append(b_pred); be.append(b_emp)
    q = lambda a: (float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975)))
    return q(bp), q(be)


rows = []
for cell in BIS_CELLS:
    for sample, df, pc, rc in [("human", hum, "partid", "score"),
                               ("LLM", llm, "model", "model_answer")]:
        R_f, R_r, p_r = cell_person_means(df, cell, pc, rc)
        rho, pi_b, beta_pred, beta_emp = slope_stats(R_f, R_r, p_r)
        (bp_lo, bp_hi), (be_lo, be_hi) = bootstrap_ci(R_f, R_r, p_r)
        rows.append({
            "sub_scale": cell, "p_r": round(p_r, 2), "sample": sample, "n": len(R_f),
            "rho": round(rho, 2), "pi_b": round(pi_b, 2),
            "beta_pred": round(beta_pred, 2), "betap_lo": round(bp_lo, 2), "betap_hi": round(bp_hi, 2),
            "beta_emp": round(beta_emp, 2), "betae_lo": round(be_lo, 2), "betae_hi": round(be_hi, 2),
        })

table = pd.DataFrame(rows)
print(table.to_string(index=False))
print(f"\nMax |beta_pred - beta_emp| across cells: {(table['beta_pred'] - table['beta_emp']).abs().max():.2f}")

out = "../../../results/tables/bias_slope_validation_barratt.csv"
table.to_csv(out, index=False)
print(f"Wrote {out}")
