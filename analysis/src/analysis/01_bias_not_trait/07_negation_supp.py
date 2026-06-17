"""Negation check for the forward-reverse asymmetry (SI Table S16).

Writes to analysis/data/negation_analysis/:
  ipipneo300_negation_labels.csv
  negation_results_summary.csv
  forward_reverse_negation_free_subset_results.csv
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_corrs_and_ci

REPO = Path(__file__).resolve().parents[4]
ITEMS = REPO / "data_generation/direct/tasks/jsonl_data/ipipneo300_items.jsonl"
DATA_DIR = REPO / "analysis/data/ipipneo300_data"
OUT = REPO / "analysis/data/negation_analysis"


def phi(x, y):
    return float(np.corrcoef(x, y)[0, 1])


# 1. BARRATT (BIS-11): reverse-keying vs negation, hand-coded per item.
BARRATT_REVERSED = {1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 1, 10: 1,
                    11: 0, 12: 1, 13: 1, 14: 0, 15: 1, 16: 0, 17: 0, 18: 0, 19: 0, 20: 1,
                    21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 1, 30: 1}
BARRATT_NEGATED = {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0,
                   11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0,
                   21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0}

_keys = sorted(BARRATT_REVERSED)
b_rev = np.array([BARRATT_REVERSED[k] for k in _keys])
b_neg = np.array([BARRATT_NEGATED[k] for k in _keys])
barratt_r = phi(b_rev, b_neg)
print(f"BARRATT (BIS-11): {b_neg.sum()}/{len(b_neg)} negated, "
      f"phi(reverse, negation) = {barratt_r:+.3f}")


# 2. IPIP-NEO-300: regex-label negation, write labels + base-rate summary.
items = [json.loads(line) for line in ITEMS.read_text().splitlines() if line.strip()]
df = pd.DataFrame(items)
df["reverse_keyed"] = df["sign"].str.startswith("-").astype(int)
df["trait"] = df["sign"].str[1]
assert len(df) == 300

STRICT = [r"\bnot\b", r"n't\b", r"\bnever\b", r"\bno\b", r"\bnothing\b",
          r"\bnobody\b", r"\bnone\b", r"\bneither\b", r"\bnor\b", r"\bcannot\b"]
EXTENDED = STRICT + [r"\brarely\b", r"\bseldom\b", r"\bfew\b", r"\blittle\b"]


def _hit(text, patterns):
    return int(any(re.search(p, text, flags=re.IGNORECASE) for p in patterns))


df["negation_strict"] = df["item"].apply(lambda s: _hit(s, STRICT))
df["negation_extended"] = df["item"].apply(lambda s: _hit(s, EXTENDED))

ipip_r_strict = phi(df.reverse_keyed, df.negation_strict)
ipip_r_extended = phi(df.reverse_keyed, df.negation_extended)
_rev = df[df.reverse_keyed == 1]
print(f"IPIP-NEO-300: phi(reverse, negation_strict) = {ipip_r_strict:+.3f}; "
      f"{_rev.negation_strict.mean():.0%} of reverse items negated "
      f"({_rev.negation_strict.sum()}/{len(_rev)})")

OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "ipipneo300_negation_labels.csv", index=False)

summary = pd.DataFrame([
    {"Instrument": "BARRATT", "n_items": len(b_rev), "n_rev": int(b_rev.sum()),
     "n_neg_strict": int(b_neg.sum()), "r_strict": barratt_r,
     "n_neg_extended": np.nan, "r_extended": np.nan},
    {"Instrument": "IPIP-NEO-300", "n_items": len(df), "n_rev": int(df.reverse_keyed.sum()),
     "n_neg_strict": int(df.negation_strict.sum()), "r_strict": ipip_r_strict,
     "n_neg_extended": int(df.negation_extended.sum()), "r_extended": ipip_r_extended},
])
summary.to_csv(OUT / "negation_results_summary.csv", index=False)
print(summary.to_string(index=False))


# 3. Forward-reverse correlation on the negation-free reverse subset.
negated_reverse_ids = set(
    df[(df.reverse_keyed == 1) & (df.negation_strict == 1)]["id"].tolist()
)

llm = pd.read_csv(DATA_DIR / "llm_data/ipipneo_no_flip_data_raw.csv", low_memory=False)
api = pd.read_csv(DATA_DIR / "api_data/ipipneo_api_data_raw.csv", low_memory=False)
api_nr = api[api["model"].str.contains("_nr", regex=False)]
llm = pd.concat([llm, api_nr], ignore_index=True)
llm_nc = llm[llm["context_mode"] == "no_context"].copy()

hum = pd.read_csv(DATA_DIR / "human_data/ipipneo_human_raw.csv", low_memory=False)
hum["response"] = pd.to_numeric(
    hum["response"].replace({"0": None, 0: None, " ": None}), errors="coerce"
)  # '0' = missing for IPIP Likert (1-5)

llm_sub = llm_nc[~llm_nc["item_id"].isin(negated_reverse_ids)].copy()
hum_sub = hum[~hum["item"].isin(negated_reverse_ids)].copy()


def model_var_filter(d):
    v = d.groupby("model")["model_answer"].var().fillna(0)
    return d[d["model"].isin(v[v > 0].index)]


def per_trait_corrs(d, entity_var, dep_var):
    return {tr: get_corrs_and_ci(d[d["traits"] == tr], entity_var, dep_var)
            for tr in ["O", "C", "E", "A", "N"]}


samples = {
    "LLMs (all)":   ("model", "model_answer", llm_nc, llm_sub),
    "LLMs (var>0)": ("model", "model_answer", model_var_filter(llm_nc), model_var_filter(llm_sub)),
    "Humans":       ("person_id", "response", hum, hum_sub),
}

rows = []
for name, (entity_var, dep_var, df_full, df_sub) in samples.items():
    full = per_trait_corrs(df_full, entity_var, dep_var)
    sub = per_trait_corrs(df_sub, entity_var, dep_var)
    for tr in ["O", "C", "E", "A", "N"]:
        rf, lf, hf, nf = full[tr]
        rs, ls, hs, ns = sub[tr]
        rows.append({
            "sample": name, "trait": tr,
            "r_full": f"{rf:+.2f}", "ci_full": f"[{lf:+.2f}, {hf:+.2f}]", "n_full": nf,
            "r_negfree": f"{rs:+.2f}", "ci_negfree": f"[{ls:+.2f}, {hs:+.2f}]", "n_negfree": ns,
            "delta": f"{rs - rf:+.3f}",
        })

subset_tab = pd.DataFrame(rows)
print(subset_tab.to_string(index=False))
subset_tab.to_csv(OUT / "forward_reverse_negation_free_subset_results.csv", index=False)
