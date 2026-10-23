"""Shared helpers for the capability-and-bias scripts: the IPIP-NEO-300 reference
frame, per-respondent (theta_hat, b_hat) contrasts, and the item-noise
(attenuation) correction of b_hat used in Fig 3."""
from pathlib import Path

import numpy as np
import pandas as pd

TRAITS = ["O", "C", "E", "A", "N"]
M = 3   # IPIP-NEO-300 scale midpoint (1-5)

DATA_DIR = Path(__file__).resolve().parents[3] / "data/ipipneo300_data"


def load_llm_reference() -> pd.DataFrame:
    """Reference condition: local models (no flip) + API models non-reasoning,
    no context."""
    raw_llms = pd.read_csv(DATA_DIR / "llm_data/ipipneo_no_flip_data_raw.csv", low_memory=False)
    raw_api = pd.read_csv(DATA_DIR / "api_data/ipipneo_api_data_raw.csv", low_memory=False)
    api_nr = raw_api[raw_api["model"].str.contains("_nr", regex=False)]
    raw_data = pd.concat([raw_llms, api_nr], ignore_index=True)
    return raw_data[raw_data["context_mode"] == "no_context"].copy()


def _block_means(df, group_col, val_col, trait):
    sub = df[df["traits"] == trait]
    Rf  = sub[~sub["reverse_coded"]].groupby(group_col)[val_col].mean()
    Rr  = sub[ sub["reverse_coded"]].groupby(group_col)[val_col].mean()
    idx = Rf.index.intersection(Rr.index)
    return Rf.loc[idx], Rr.loc[idx]


def _block_error_var(df, group_col, val_col, item_col, trait):
    """Item-noise variance of b̂ per respondent: (s_f²/n_f + s_r²/n_r) / 4,
    with s² the respondent's own item variance within each keying block,
    computed on ITEM-CENTRED responses (each item's mean across respondents
    removed). Item difficulty shifts every respondent equally, so it adds no
    between-respondent variance and is not noise. item_col=None skips the
    centring (raw within-respondent variance; sensitivity check only)."""
    sub = df[df["traits"] == trait]
    resid = sub[val_col] if item_col is None else \
        sub[val_col] - sub.groupby(item_col)[val_col].transform("mean")
    ev  = lambda x: x.var() / x.count()
    fwd = ~sub["reverse_coded"]
    Ef  = resid[fwd].groupby(sub.loc[fwd, group_col]).agg(ev)
    Er  = resid[~fwd].groupby(sub.loc[~fwd, group_col]).agg(ev)
    idx = Ef.index.intersection(Er.index)
    return (Ef.loc[idx] + Er.loc[idx]) / 4


def per_respondent_contrasts(raw, group_col, val_col, item_col):
    """item_col identifies the item, for item-centring the noise term; pass one
    group (humans or LLMs) at a time so items are centred within that group."""
    out = []
    for t in TRAITS:
        Rf, Rr = _block_means(raw, group_col, val_col, t)
        out.append(pd.DataFrame({
            group_col:   Rf.index,
            "trait":     t,
            "b_hat":     (Rf.values + Rr.values) / 2 - M,
            "theta_hat": M + (Rf.values - Rr.values) / 2,
            "b_err_var": _block_error_var(raw, group_col, val_col, item_col, t).loc[Rf.index].values,
        }))
    return pd.concat(out, ignore_index=True)


def attenuation_correct(contrasts):
    """Adds `b_corr`: b̂ with item noise removed from its spread (per trait).
    Var(b̂) = Var(b) + mean error variance, so Var(b) = Var(b̂) - mean(b_err_var).
    Each b̂ is pulled toward the trait mean by sqrt(reliability) so the corrected
    values have the estimated true spread; returns the per-trait reliability etc."""
    out = {}
    contrasts["b_corr"] = np.nan
    for t in TRAITS:
        m      = contrasts["trait"] == t
        b      = contrasts.loc[m, "b_hat"]
        var_obs = float(b.var())
        var_err = float(contrasts.loc[m, "b_err_var"].mean())
        var_true = max(var_obs - var_err, 0.0)
        rel    = var_true / var_obs
        contrasts.loc[m, "b_corr"] = b.mean() + np.sqrt(rel) * (b - b.mean())
        out[t] = {"var_obs": var_obs, "var_err": var_err,
                  "var_true": var_true, "reliability": rel}
    return out
