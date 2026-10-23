"""SI: convergent validity of the trait component, descriptive comparison of LLMs
and humans.

For instrument pairs that target related constructs, the correlation of the trait
estimate theta_hat across respondents (models; persons) is compared between LLMs
and humans, alongside the same correlation of the bias estimate b_hat. Only
instruments with both keyings yield theta_hat: IPIP-NEO-300 domains and facets,
BARRATT (BIS) and SSSV (totals and facets), and LOT (DFD is excluded: its keying
coincides with gain/loss domain). The pairs and expected signs are illustrative,
not a theory-derived validation design.

  A. Frey battery (same 1,507 persons did both instruments): human benchmark
  B. Frey battery x IPIP-NEO-300: LLMs only (no person did both)
  C. within the IPIP-NEO-300 (20,993 persons): human benchmark

Per unit and (sub-)scale, with raw answers and m the scale midpoint:
    theta_hat = m + (R_f - R_r)/2,   b_hat = (R_f + R_r)/2 - m
Correlations are also reported disattenuated for item noise, with the same
measurement-error model for both populations (02_capability_and_bias/utils.py):
the noise variance of either contrast is (s_f^2/n_f + s_r^2/n_r)/4 per unit, with
s^2 the unit's variance of ITEM-CENTRED responses (each item's mean across the
population's units removed; item difficulty is shared by all units and is not
noise), reliability = 1 - mean noise variance / observed variance, and
r_dis = r / sqrt(rel_x * rel_y). r_dis is reported only where both reliabilities
are >= .5; below that the ratio is unstable (it can exceed 1).
LOT is coded by position for both (1 = chose the first-listed option, keying =
risky option listed second), so its theta_hat is the share of risky choices.
Human LOT uses first-stage trials only (the 25 gambles the LLMs also answered).

Reads the loaders of 01_profiles_per_keying.py (LLM reference condition, 60
models; position-coded human LOT) and the human IPIP-NEO-300 file.
Writes results/tables/convergent_validity.{csv,tex}.
"""
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from panel import bare_model_name, low_variability_models  # noqa: E402

TABLE_DIR = HERE.parents[2] / "results/tables"
HUMAN_IPIP = HERE.parents[2] / "data/ipipneo300_data/human_data/ipipneo_human_raw.csv"

_spec = importlib.util.spec_from_file_location("profiles", HERE / "01_profiles_per_keying.py")
profiles = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(profiles)

# (sub-)scale -> (experiment, category or None for the whole instrument, scale min, max)
SCALES = {
    "BIS":                ("BARRATT", None, 1, 4),
    "SSSV":               ("SSSV", None, 1, 2),
    "SSSV Exp. seeking":  ("SSSV", "SSexp", 1, 2),
    "LOT":                ("LOT", None, 0, 1),
    "C":                  ("IPIP-NEO-300", "C", 1, 5),
    "E Excitement-Seeking": ("IPIP-NEO-300", "Excitement-Seeking", 1, 5),
    "O Adventurousness":  ("IPIP-NEO-300", "Adventurousness", 1, 5),
    "N Immoderation":     ("IPIP-NEO-300", "Immoderation", 1, 5),
    "C Cautiousness":     ("IPIP-NEO-300", "Cautiousness", 1, 5),
    "C Self-Discipline":  ("IPIP-NEO-300", "Self-Discipline", 1, 5),
}

PAIRS = [  # (set, scale x, scale y, expected sign)
    ("A", "BIS", "SSSV", "+"),
    ("A", "SSSV", "LOT", "+"),
    ("A", "BIS", "LOT", "+"),
    ("B", "SSSV", "E Excitement-Seeking", "+"),
    ("B", "SSSV Exp. seeking", "O Adventurousness", "+"),
    ("B", "BIS", "C", "-"),
    ("B", "BIS", "N Immoderation", "+"),
    ("B", "LOT", "C Cautiousness", "-"),
    ("C", "E Excitement-Seeking", "O Adventurousness", "+"),
    ("C", "C Cautiousness", "N Immoderation", "-"),
    ("C", "C Cautiousness", "E Excitement-Seeking", "-"),
    ("C", "C Self-Discipline", "N Immoderation", "-"),
]

# ============================================================
# Data: one long frame per population (unit, experiment, category, facet, x, reverse_coded)
# ============================================================

with contextlib.redirect_stdout(io.StringIO()):
    llm = profiles.load_combined_reference()
    human_risk = profiles.lot_first_stage(profiles.load_human_risk_items())

human_ipip = pd.read_csv(HUMAN_IPIP, low_memory=False)
human_ipip["response"] = pd.to_numeric(
    human_ipip["response"].replace({"0": None, 0: None, " ": None}), errors="coerce")
human_ipip = human_ipip[human_ipip["response"].between(1, 5)]

# IPIP facet per item, from the item key carried by the human file
facet = human_ipip.drop_duplicates("item").set_index("item")["category"]
llm = llm.rename(columns={"model": "unit", "model_answer": "x", "item_id": "item"})
is_ipip = llm["experiment"] == "IPIP-NEO-300"
llm["facet"] = np.where(is_ipip, pd.to_numeric(llm["item"], errors="coerce").map(facet), None)
assert llm.loc[is_ipip, "facet"].notna().all()

human_ipip_long = pd.DataFrame({
    "unit": human_ipip["person_id"], "experiment": "IPIP-NEO-300",
    "category": human_ipip["traits"], "facet": human_ipip["category"],
    "item": human_ipip["item"], "x": human_ipip["response"],
    "reverse_coded": human_ipip["reverse_coded"],
})
human = pd.concat([
    human_risk.rename(columns={"person_id": "unit", "response": "x", "item_id": "item"})
              .assign(facet=None),
    human_ipip_long,
], ignore_index=True)
human["unit"] = human["experiment"].where(human["experiment"] == "IPIP-NEO-300", "frey") + "_" + human["unit"].astype(str)


def scale_slice(df, name):
    exp, cat, *_ = SCALES[name]
    d = df[df["experiment"] == exp]
    if cat is None:
        return d
    col = "category" if exp != "IPIP-NEO-300" or len(cat) == 1 else "facet"
    return d[d[col] == cat]


def contrasts(df, name):
    """Per unit theta_hat, b_hat and their (common) item-noise variance, the
    latter from item-centred responses (one population at a time)."""
    _, _, lo, hi = SCALES[name]
    m = (lo + hi) / 2
    d = scale_slice(df, name).assign(reverse_coded=lambda d: d["reverse_coded"].astype(bool))
    d = d.assign(resid=d["x"] - d.groupby(["item", "reverse_coded"])["x"].transform("mean"))
    g = (d.groupby(["unit", "reverse_coded"])
          .agg(mean=("x", "mean"), var=("resid", "var"), count=("x", "count")).unstack().dropna())
    rf, rr = g[("mean", False)], g[("mean", True)]
    err = (g[("var", False)] / g[("count", False)] + g[("var", True)] / g[("count", True)]) / 4
    return pd.DataFrame({"theta": m + (rf - rr) / 2, "b": (rf + rr) / 2 - m, "err": err})


MIN_RELIABILITY = 0.5   # disattenuated r shown only if both reliabilities reach this


def reliability(c, comp):
    v = c[comp].var()
    return max(v - c["err"].mean(), 0.0) / v if v > 0 else np.nan


constant = set(low_variability_models()[0]["constant"])
rows = []
for set_, x, y, sign in PAIRS:
    pops = [("LLMs", llm)] + ([] if set_ == "B" else [("Humans", human)])
    for pop, df in pops:
        cx, cy = contrasts(df, x), contrasts(df, y)
        units = cx.index.intersection(cy.index)
        samples = [("all", units)]
        if pop == "LLMs":
            samples.append(("excl. constant", [u for u in units if bare_model_name(u) not in constant]))
        for sample, u in samples:
            row = {"set": set_, "x": x, "y": y, "expected": sign, "population": pop,
                   "sample": sample, "n": len(u)}
            for comp in ["theta", "b"]:
                a, b = cx.loc[u, comp], cy.loc[u, comp]
                r = stats.pearsonr(a, b)[0] if a.std() > 0 and b.std() > 0 else np.nan
                rel_x, rel_y = reliability(cx.loc[u], comp), reliability(cy.loc[u], comp)
                shown = min(rel_x, rel_y) >= MIN_RELIABILITY
                row.update({f"r_{comp}": r,
                            f"r_{comp}_dis": r / np.sqrt(rel_x * rel_y) if shown else np.nan,
                            f"rel_{comp}_x": rel_x, f"rel_{comp}_y": rel_y})
            rows.append(row)
res = pd.DataFrame(rows)
res.to_csv(TABLE_DIR / "convergent_validity.csv", index=False)

# ============================================================
# LaTeX: one row per pair, LLMs (all models) vs humans
# ============================================================

def f(v):
    return "--" if pd.isna(v) else f"{v:.2f}"


def n_persons(ipip: bool) -> str:
    units = human.loc[(human["experiment"] == "IPIP-NEO-300") == ipip, "unit"]
    return f"{units.nunique():,}".replace(",", "{,}")


SET_TITLE = {"A": f"Frey battery (human benchmark, $n$ = {n_persons(False)})",
             "B": "Frey battery $\\times$ IPIP-NEO-300 (LLMs only)",
             "C": f"Within IPIP-NEO-300 (human benchmark, $n$ = {n_persons(True)})"}
lines = []
for set_ in "ABC":
    lines.append(f"\\midrule\n\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{SET_TITLE[set_]}}}}} \\\\")
    for _, p in res[(res["set"] == set_)].groupby(["x", "y"], sort=False):
        L = p[(p["population"] == "LLMs") & (p["sample"] == "all")].iloc[0]
        H = p[p["population"] == "Humans"]
        H = H.iloc[0] if len(H) else pd.Series(dtype=float)
        lines.append(" & ".join([
            f"{L['x']} -- {L['y']}", L["expected"],
            f(L["r_theta"]), f(H.get("r_theta")),
            f(L["r_theta_dis"]), f(H.get("r_theta_dis")),
            f"{f(L['r_b'])} / {f(H.get('r_b'))}",
        ]) + " \\\\")

caption = (
    "Convergent validity of the trait component, descriptive. Pearson correlation of "
    f"$\\hat\\theta$ between two (sub-)scales across respondents ({llm['unit'].nunique()} LLMs; persons), raw "
    "and corrected for item noise with the same error model for both populations "
    "(shown only where item noise accounts for at most half of the spread of both "
    "estimates), and of $\\hat b$ for comparison (LLMs / humans). Expected signs are "
    "illustrative. The Frey $\\times$ IPIP-NEO-300 block has no human benchmark because "
    "no person completed both batteries. "
    "Excluding the constant responders: see the accompanying CSV."
)
header = ("Pair & exp. & $r_{\\hat\\theta}$ LLMs & $r_{\\hat\\theta}$ humans & "
          "$r_{\\hat\\theta}$ corr. LLMs & $r_{\\hat\\theta}$ corr. humans & "
          "$r_{\\hat b}$ LLMs / humans \\\\")
with open(TABLE_DIR / "convergent_validity.tex", "w", encoding="utf-8") as fh:
    fh.write("\\begin{table}\n\\caption{" + caption + "}\n\\label{tab:convergent_validity}\n"
             "\\begin{tabular}{@{}llccccc@{}}\n\\toprule\n" + header + "\n"
             + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

pd.set_option("display.width", 250)
print(res.round(2).to_string(index=False))
