"""SI: clustering check. Is homogeneity higher within a class than
across classes? One-way ICC(1) with F-based 95% CIs, for

  (a) per-model estimates (Big Five b_hat and theta_hat per trait, and the mean
      |normalised b_hat| over the 14 subscales, uncorrected) within model FAMILY, at two
      grain sizes: series (model line, e.g. Qwen2.5 vs Qwen3) and developer;
  (b) the per-subscale points behind the cross-instrument orthogonality
      correlation, within PARENT INSTRUMENT: mean inter-item correlation, p_r, and
      the residual of the mean inter-item correlation on p_r. Only the residual
      bears on whether clustered intervals are needed.

An ICC not reliably above zero indicates no excess homogeneity within the class.
With few, small clusters the ICCs are imprecise: read the CI, not the point estimate.

Reads the reference-condition LLM data via 01_profiles_per_keying.py and
results/tables/orthogonality_subscales_reference.csv (written by
03_orthogonality/02_orthogonality_corr_by_condition.py).
Writes results/tables/clustering_icc.{csv,tex}.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT_DIR = ROOT / "analysis/results/tables"
SUBSCALE_CSV = OUT_DIR / "orthogonality_subscales_reference.csv"

_spec = importlib.util.spec_from_file_location("profiles", HERE / "01_profiles_per_keying.py")
profiles = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(profiles)

# Model -> (series, developer). Keys are the bare names produced by
# 01_profiles_per_keying.normalize_api_model.
FAMILY = {
    "Apertus-70B-Instruct-2509": ("Apertus", "Swiss AI"),
    "Apertus-8B-Instruct-2509": ("Apertus", "Swiss AI"),
    "Falcon-3-10B-Instruct": ("Falcon-3", "TII"),
    "Falcon-3-1B-Instruct": ("Falcon-3", "TII"),
    "Falcon-3-7B-Instruct": ("Falcon-3", "TII"),
    "GLM-5.3-Flash": ("GLM-5", "Zhipu"),
    "Kimi-K2.6": ("Kimi-K2", "Moonshot"),
    "LFM2-1.2B": ("LFM2", "Liquid"),
    "LFM2-2.6B": ("LFM2", "Liquid"),
    "LFM2-8B-A1B": ("LFM2", "Liquid"),
    "Llama-3.1-70B-Instruct": ("Llama-3", "Meta"),
    "Llama-3.1-8B-Instruct": ("Llama-3", "Meta"),
    "Llama-3.2-1B-Instruct": ("Llama-3", "Meta"),
    "Llama-3.2-3B-Instruct": ("Llama-3", "Meta"),
    "Llama-3.3-70B-Instruct": ("Llama-3", "Meta"),
    "MiniMax-M3": ("MiniMax-M", "MiniMax"),
    "Ministral-8B-Instruct-2410": ("Mistral", "Mistral AI"),
    "Mistral-7B-Instruct-v0.3": ("Mistral", "Mistral AI"),
    "Mistral-Small-24B-Instruct-2501": ("Mistral", "Mistral AI"),
    "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": ("Nemotron-3", "NVIDIA"),
    "OLMo-2-7B-Instruct": ("OLMo-2", "AI2"),
    "Phi-3-medium-128k-instruct": ("Phi-3", "Microsoft"),
    "Phi-3-mini-128k-instruct": ("Phi-3", "Microsoft"),
    "Phi-3.5-mini-instruct": ("Phi-3", "Microsoft"),
    "Qwen2.5-1.5B-Instruct": ("Qwen2.5", "Alibaba"),
    "Qwen2.5-14B-Instruct": ("Qwen2.5", "Alibaba"),
    "Qwen2.5-32B-Instruct": ("Qwen2.5", "Alibaba"),
    "Qwen2.5-3B-Instruct": ("Qwen2.5", "Alibaba"),
    "Qwen2.5-7B-Instruct": ("Qwen2.5", "Alibaba"),
    "Qwen3-1.7B": ("Qwen3", "Alibaba"),
    "Qwen3-14B": ("Qwen3", "Alibaba"),
    "Qwen3-30B-A3B-Instruct-2507": ("Qwen3", "Alibaba"),
    "Qwen3-32B": ("Qwen3", "Alibaba"),
    "Qwen3-4B": ("Qwen3", "Alibaba"),
    "Qwen3-8B": ("Qwen3", "Alibaba"),
    "qwen3.5-flash": ("Qwen3.5", "Alibaba"),
    "qwen3.5-plus": ("Qwen3.5", "Alibaba"),
    "SmolLM3-3B": ("SmolLM3", "Hugging Face"),
    "zephyr-7b-beta": ("Zephyr", "Hugging Face"),
    "TildeOpen-30b": ("TildeOpen", "Tilde"),
    "bloomz-3b": ("BLOOMZ", "BigScience"),
    "bloomz-7b1": ("BLOOMZ", "BigScience"),
    "claude-haiku-4-5": ("Claude-4", "Anthropic"),
    "claude-opus-4-6": ("Claude-4", "Anthropic"),
    "claude-sonnet-4-6": ("Claude-4", "Anthropic"),
    "gemini-3.1-flash-lite": ("Gemini-3", "Google"),
    "gemma-2-27b-it": ("Gemma-2", "Google"),
    "gemma-2-2b-it": ("Gemma-2", "Google"),
    "gemma-2-9b-it": ("Gemma-2", "Google"),
    "gemma-3-12b-it": ("Gemma-3", "Google"),
    "gemma-3-1b-it": ("Gemma-3", "Google"),
    "gemma-3-27b-it": ("Gemma-3", "Google"),
    "gemma-3-4b-it": ("Gemma-3", "Google"),
    "gpt-5.4": ("GPT-5", "OpenAI"),
    "gpt-5.4-mini": ("GPT-5", "OpenAI"),
    "gpt-5.4-nano": ("GPT-5", "OpenAI"),
    "gpt-oss-20b": ("gpt-oss", "OpenAI"),
    "granite-3.3-2b-instruct": ("Granite-3", "IBM"),
    "granite-3.3-8b-instruct": ("Granite-3", "IBM"),
    "grok-4.20": ("Grok-4", "xAI"),
}


def icc1(values, groups, alpha=0.05):
    """One-way random-effects ICC(1) for unbalanced groups (n0 = adjusted mean
    group size) with the F-based CI. Singleton groups add between-group but no
    within-group information."""
    d = pd.DataFrame({"y": values, "g": groups}).dropna()
    N, k = len(d), d["g"].nunique()
    sizes = d.groupby("g").size()
    grand = d["y"].mean()
    means = d.groupby("g")["y"].transform("mean")
    msb = (sizes * (d.groupby("g")["y"].mean() - grand) ** 2).sum() / (k - 1)
    msw = ((d["y"] - means) ** 2).sum() / (N - k)
    n0 = (N - (sizes ** 2).sum() / N) / (k - 1)
    F = msb / msw
    icc = (F - 1) / (F + n0 - 1)
    fl = F / stats.f.ppf(1 - alpha / 2, k - 1, N - k)
    fu = F * stats.f.ppf(1 - alpha / 2, N - k, k - 1)
    return {
        "icc": icc, "ci_lo": (fl - 1) / (fl + n0 - 1), "ci_hi": (fu - 1) / (fu + n0 - 1),
        "F": F, "p": stats.f.sf(F, k - 1, N - k), "n": N, "k": k,
        "k_multi": int((sizes > 1).sum()), "n0": n0,
    }


def per_model_estimates():
    """Per model: Big Five signed b_hat and theta_hat (raw 1-5 scale, as in Fig 3),
    and the mean |normalised b_hat| over the 14 subscales (uncorrected)."""
    df = profiles.load_combined_reference()
    big5 = df[df["experiment"] == "IPIP-NEO-300"]
    g = big5.groupby(["model", "category", "reverse_coded"])["model_answer"].mean().unstack()
    Rf, Rr = g[False], g[True]
    b_hat = ((Rf + Rr) / 2 - 3).unstack().add_prefix("b_hat ")
    theta_hat = (3 + (Rf - Rr) / 2).unstack().add_prefix("theta_hat ")
    out = pd.concat([b_hat, theta_hat], axis=1)
    out["mean |b_hat| (14 subscales)"] = profiles.compute_mu_a_risk(df)["mu_a_risk_abs"]
    return out


def main() -> None:
    rows = []

    # (a) model family
    est = per_model_estimates()
    missing = set(est.index) - set(FAMILY)
    if missing:
        raise ValueError(f"Models without a family label: {sorted(missing)}")
    for level, idx in [("series", 0), ("developer", 1)]:
        groups = est.index.map(lambda m: FAMILY[m][idx])
        for col in est.columns:
            rows.append({"cluster": f"family ({level})", "population": "LLMs",
                         "quantity": col, **icc1(est[col].to_numpy(), groups)})

    # (b) parent instrument
    sub = pd.read_csv(SUBSCALE_CSV).dropna(
        subset=["pct_reversed_keyed_items", "mean_interitem_corr"])
    for pop, d in sub.groupby("population"):
        x, y = d["pct_reversed_keyed_items"], d["mean_interitem_corr"]
        slope, intercept, *_ = stats.linregress(x, y)
        resid = y - (intercept + slope * x)
        for name, vals in [("mean inter-item corr", y), ("p_r", x),
                           ("residual (inter-item corr | p_r)", resid)]:
            rows.append({"cluster": "parent instrument", "population": pop,
                         "quantity": name, **icc1(vals.to_numpy(), d["experiment"].to_numpy())})

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "clustering_icc.csv", index=False)

    tex = res.assign(
        ICC=res.apply(lambda r: f"{r.icc:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]", axis=1),
        p=res["p"].map(lambda p: "<.001" if p < .001 else f"{p:.3f}"),
        clusters=res.apply(lambda r: f"{r.k} ({r.k_multi})", axis=1),
    )[["cluster", "population", "quantity", "n", "clusters", "ICC", "p"]]
    latex = tex.to_latex(
        index=False, escape=True, column_format="lllrrll",
        caption="Homogeneity within model family and within parent instrument: "
                "one-way ICC(1) with 95\\% CI. Clusters: total (with more than one member).",
        label="tab:clustering_icc")
    latex = latex.replace("<.001", "$<$.001")  # "<" must be in math mode in LaTeX
    for raw, pretty in [("b\\_hat ", "$\\hat b$ "), ("theta\\_hat ", "$\\hat\\theta$ "),
                        ("mean |$\\hat b$| (14 subscales)", "mean $|\\hat b|$ (14 (sub-)scales)"),
                        ("residual (inter-item corr | p\\_r)", "residual of inter-item corr on $p_r$"),
                        ("p\\_r", "$p_r$"), ("quantity", "Quantity"), ("cluster &", "Cluster &"),
                        ("population", "Population"), ("clusters", "Clusters"), (" n &", " $n$ &"), ("& p \\\\", "& $p$ \\\\")]:
        latex = latex.replace(raw, pretty)
    with open(OUT_DIR / "clustering_icc.tex", "w", encoding="utf-8") as f:
        f.write(latex)

    pd.set_option("display.width", 220)
    print(tex.to_string(index=False))


if __name__ == "__main__":
    main()
