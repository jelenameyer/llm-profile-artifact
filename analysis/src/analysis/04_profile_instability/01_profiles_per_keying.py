"""Fig 4: per-respondent profile small-multiples (mean human + 56 LLMs). For each
respondent, plot the normalised subscale score T across the 14 subscales at three
keying conditions (p_r = 0 / 0.5 / 1); the band between p_r=0 and p_r=1 is the
score range driven by keying alone. Panels ordered by mean |bias|.

Per (respondent, subscale, target p_r): pick the (k_f, k_r) split closest to the
target with k_f+k_r >= MIN_K_PROFILE, bootstrap T over item subsets, normalise to
[0,1]. Two gotchas: behavioral tasks (LOT, DFD) use POSITION-coded human responses
(which option was chosen) so the human keying-shift is comparable to the LLMs; and
BARRATT BISm has 1 reverse item, so p_r=1 collapses to 0.5 and is shown as a faded
proxy via 2*T(0.5)-T(0).

Reads risk_data + ipipneo300_data (LLM/api/human, raw) + raw human LOT/DFD sources.
Writes:
  results/tables/profiles_per_keying.csv
  results/figures/profiles_per_keying.pdf
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[4]
LLM_RAW_CSV = (
    ROOT
    / "analysis/data/intermediate/risk_data/LLM_data_proc_prompts_direct/LLM_no_flip_data_raw.csv"
)
IPIP_RAW_CSV = (
    ROOT
    / "analysis/data/intermediate/ipipneo300_data/llm_data/ipipneo_all_data_raw.csv"
)
# Proprietary-API models (collected separately, no_context / no-reasoning / no logits).
LLM_API_CSV = (
    ROOT
    / "analysis/data/intermediate/risk_data/api_data/LLM_api_no_flip_data_raw.csv"
)
IPIP_API_CSV = (
    ROOT
    / "analysis/data/intermediate/ipipneo300_data/api_data/ipipneo_api_data_raw.csv"
)
# Per-person human item responses (for the human-mean reference panel).
HUMAN_RISK_CSV = (
    ROOT / "analysis/data/intermediate/risk_data/human_data_proc/raw_items_per_person.csv"
)
HUMAN_IPIP_CSV = (
    ROOT / "analysis/data/intermediate/ipipneo300_data/human_data/ipipneo_human_raw.csv"
)
# Raw human source files for the behavioral tasks. Unlike the self-report scales,
# behavioral "reverse-keying" = the risky option's POSITION, so the human profile
# must be position-coded (which option was chosen) to be comparable to the LLM
# panels — the per-person risk file only stores the content score (chose risky).
HUMAN_LOT_SRC = ROOT / "analysis/data/raw/risk_data/orig_human_data/lotteries.csv"
HUMAN_DFD_SRC = ROOT / "analysis/data/raw/risk_data/orig_human_data/dfd_perprob.csv"
# These two are original Frey et al. (2017) files, not re-hosted here but fetched from the
# public Frey OSF project by download_data.py. Required for the human LOT/DFD points.
_missing = [p.name for p in (HUMAN_LOT_SRC, HUMAN_DFD_SRC) if not p.exists()]
if _missing:
    print(
        f"[skip] missing original Frey et al. (2017) data: {', '.join(_missing)} (normally\n"
        "       fetched by download_data.py). Run `python download_data.py` (or\n"
        "       python run_all.py), then re-run. Skipping for now.",
        file=sys.stderr,
    )
    sys.exit(77)
OUT_TABLE = ROOT / "analysis/results/tables/profiles_per_keying.csv"
OUT_FIG = ROOT / "analysis/results/figures/profiles_per_keying.pdf"
OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

# (experiment, category, display_label, scale_min, scale_max). The y-axis
# order in each subplot is set dynamically (sorted by average T_norm at
# p_r=0.5 across all models) — this list is just the canonical inventory.
SUBSCALES = [
    # Big Five — IPIP-NEO-300, 1–5 Likert, ~30 forward & ~30 reverse per trait.
    ("IPIP-NEO-300", "O", "Openness",          1, 5),
    ("IPIP-NEO-300", "C", "Conscientiousness", 1, 5),
    ("IPIP-NEO-300", "E", "Extraversion",      1, 5),
    ("IPIP-NEO-300", "A", "Agreeableness",     1, 5),
    ("IPIP-NEO-300", "N", "Neuroticism",       1, 5),
    # Risk subscales
    ("BARRATT", "BISa",  "BARRATT BISa", 1, 4),
    ("BARRATT", "BISm",  "BARRATT BISm", 1, 4),
    ("BARRATT", "BISn",  "BARRATT BISn", 1, 4),
    ("SSSV",    "SSbor", "SSSV SSbor",   1, 2),
    ("SSSV",    "SSdis", "SSSV SSdis",   1, 2),
    ("SSSV",    "SSexp", "SSSV SSexp",   1, 2),
    ("SSSV",    "SStas", "SSSV SStas",   1, 2),
    ("DFD",     None,    "DFD",          0, 1),
    ("LOT",     None,    "LOT",          0, 1),
]
SUBSCALE_LABELS = [row[2] for row in SUBSCALES]
# Big Five rows lead each panel; everything after them is the risk block. The
# two are drawn as separate segments (see BLOCK_GAP) so the line and band break
# between the personality and risk scale families.
N_BIG5 = sum(1 for row in SUBSCALES if row[0] == "IPIP-NEO-300")
# Long y-axis tick labels: Big Five keep their single-word names; each risk
# subscale gets a "<code>: <description>" gloss. Keyed by the canonical data
# label (SUBSCALES[2]) so the CSV `subscale` column stays unchanged.
SUBSCALE_TICK_LABELS_MAP = {
    "Openness":          "Openness",
    "Conscientiousness": "Conscientiousness",
    "Extraversion":      "Extraversion",
    "Agreeableness":     "Agreeableness",
    "Neuroticism":       "Neuroticism",
    "BARRATT BISa":      "Attentional impulsivity",
    "BARRATT BISm":      "Motor impulsivity",
    "BARRATT BISn":      "Nonplanning behavior impulsivity",
    "SSSV SSbor":        "Boredom susceptibility",
    "SSSV SSdis":        "Disinhibition",
    "SSSV SSexp":        "Experience seeking",
    "SSSV SStas":        "Thrill and adventure seeking",
    "DFD":               "Decisions from description",
    "LOT":               "Lotteries",
}

# Sentinel "model" key + panel title for the human-mean reference profile,
# which is rendered as the first small-multiple (kept out of the LLM roster
# and out of the saved table so downstream scripts see LLMs only).
HUMAN_KEY = "__human_mean__"
HUMAN_DISPLAY = "Humans (mean)"

TARGET_P_R = [0.0, 0.5, 1.0]
MIN_K_PROFILE = 2
N_BOOTSTRAP = 300
RNG_SEED = 13
# If the achievable p_r for a (subscale, target) deviates more than this
# from the target, drop the dot and break the line rather than mis-label.
# BISm at target=1 only reaches p_r=0.5 (1 reverse item), so it gets skipped.
P_R_TOLERANCE = 0.1

# p_r is encoded by neutral line shade + marker shape (so the only colour in
# a panel is the human/LLM fill). Dark→light grey tracks increasing p_r.
LINE_COLORS = {
    0.0: "black",
    0.5: "0.70",
    1.0: "0.35",
}
MARKERS = {
    0.0: "o",   # all forward
    1.0: "s",   # all reverse
}
LINE_LABELS = {
    0.0: r"$p_r=0$ (all forward)",
    0.5: r"$p_r=0.5$ (balanced)",
    1.0: r"$p_r=1$ (all reverse)",
}
# Band fill encodes the entity: purple = humans, teal = open-weight LLMs, distinct
# teal = proprietary-API LLMs. Markers take their line's grey shade.
HUMAN_FILL = "#440154"
LLM_FILL = "#1e847f"
LLM_API_FILL = "#31688e"   # proprietary-API LLMs (slightly distinct teal)
# Shaded band between the two extremes (p_r=0 and p_r=1): the wider it is,
# the more keying convention alone moves the model's subscale score. Drawn in
# the panel's entity fill colour.
BAND_ALPHA = 0.38
# Personality (Big Five) and risk subscales are drawn as two separate line/band
# segments with this vertical gap (in row units) between them, so the two scale
# families read distinctly.
BLOCK_GAP = 0.7
# Font sizes for the per-panel labels.
FS_YTICK = 11    # subscale (y-axis) labels
FS_XTICK = 11    # x-axis tick labels
FS_TITLE = 15    # panel title (model name), legend content
FS_BIAS = 11     # bias estimate, shown one line below the model name (smaller)
ANNOTATION_HALO = [patheffects.withStroke(linewidth=2.0, foreground="white")]


# ============================================================
# Display helpers
# ============================================================


def clean_model_name(name: str) -> str:
    """Shorten verbose model IDs for display: drop trailing dates,
    instruction-tuning suffixes, version tags, context-length tags;
    normalise lowercase model-size 'b' to 'B'.

    The strippers run in a fixed-point loop so that nested suffixes like
    '-Instruct-v0.3' or '-Instruct-2509' fully collapse."""
    n = name
    for _ in range(5):
        prev = n
        n = re.sub(r"-2\d{3}$", "", n)              # date e.g. -2509
        n = re.sub(r"-(?:Instruct|instruct)$", "", n)
        if n.startswith("gemma"):
            n = re.sub(r"-it$", "", n)
        n = re.sub(r"-v\d+(?:\.\d+)?$", "", n)      # version tag
        n = re.sub(r"-128k", "", n)                  # context-length tag
        if n == prev:
            break
    n = re.sub(r"(\d)b(?=\b|-)", r"\1B", n)         # -7b → -7B
    return n


def normalize_api_model(name: str) -> str:
    """Collapse a proprietary-API model key to its bare model name.

    API keys have the form ``provider__model-id__condition`` (e.g.
    ``anthropic__claude-opus-4-6__nr``). Open-weights names contain no
    ``__`` and are returned unchanged. For API keys we keep only the
    model-id and trim provider-specific noise so the same model lines up
    across the IPIP and risk datasets (which were collected in separate
    runs and disagree on a couple of tags, e.g. gemini ``-preview``):
      - drop a trailing ``-preview`` build tag,
      - drop a trailing ISO date ``-YYYY-MM-DD`` (Qwen),
      - drop grok's ``-NNNN-(non-)reasoning`` variant tag.
    """
    if "__" not in name:
        return name
    model_id = name.split("__")[1]
    model_id = re.sub(r"-preview$", "", model_id)
    model_id = re.sub(r"-\d{4}-(?:non-)?reasoning$", "", model_id)  # grok: -0309-non-reasoning
    model_id = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model_id)         # qwen: -2026-02-23
    return model_id


# ============================================================
# Data loading
# ============================================================


def _load_api_nr(path: Path) -> pd.DataFrame:
    """Load a proprietary-API table, keep only the no-reasoning (``__nr``)
    condition, and collapse model keys to bare model names. The API runs
    produced no ``__r`` for the risk scales, so restricting to ``__nr``
    keeps the IPIP and risk rosters aligned and avoids ``_nr``/``_r``
    name collisions when the condition suffix is stripped."""
    df = pd.read_csv(path, low_memory=False)
    df = df[df["model"].str.endswith("__nr")].copy()
    df["model"] = df["model"].map(normalize_api_model)
    return df


def load_llm_reference() -> pd.DataFrame:
    """Risk-subscale LLM responses (reference condition: no_context, no flip),
    open-weights + proprietary-API models combined."""
    df = pd.concat(
        [pd.read_csv(LLM_RAW_CSV, low_memory=False), _load_api_nr(LLM_API_CSV)],
        ignore_index=True,
    )
    df = df[(df["context_mode"] == "no_context") & (df["flipped"] == False)].copy()
    df = df[["model", "experiment", "category", "item_id", "model_answer", "reverse_coded"]]
    df["model_answer"] = pd.to_numeric(df["model_answer"], errors="coerce")
    df["reverse_coded"] = df["reverse_coded"].astype(bool)
    return df.dropna(subset=["model_answer"])


def load_ipip_reference() -> pd.DataFrame:
    """IPIP-NEO-300 LLM responses, normalised to the same columns as the
    risk-data frame: the trait letter (O/C/E/A/N) is moved from the
    `traits` column to `category` so the rest of the pipeline can use a
    single `experiment + category` key uniformly."""
    df = pd.concat(
        [pd.read_csv(IPIP_RAW_CSV, low_memory=False), _load_api_nr(IPIP_API_CSV)],
        ignore_index=True,
    )
    df = df[(df["context_mode"] == "no_context") & (df["flipped"] == False)].copy()
    df = df[["model", "experiment", "traits", "item_id", "model_answer", "reverse_coded"]]
    df = df.rename(columns={"traits": "category"})
    df["model_answer"] = pd.to_numeric(df["model_answer"], errors="coerce")
    df["reverse_coded"] = df["reverse_coded"].astype(bool)
    return df.dropna(subset=["model_answer"])


def load_combined_reference() -> pd.DataFrame:
    """Union of risk + IPIP reference data with consistent column layout.

    Restricted to models present in BOTH datasets so the figure's per-model
    panels carry the full 14-row profile (any model that appears only on
    one side would mislead the dynamic y-axis sort)."""
    risk = load_llm_reference()
    ipip = load_ipip_reference()
    shared = sorted(set(risk["model"]) & set(ipip["model"]))
    return pd.concat([
        risk[risk["model"].isin(shared)],
        ipip[ipip["model"].isin(shared)],
    ], ignore_index=True)


def load_proprietary_model_names() -> set[str]:
    """Bare model names (``normalize_api_model`` form) of the proprietary-API
    models. Used only for display: their panels get the API fill colour and a
    trailing ``*`` in the title."""
    names: set[str] = set()
    for path in (LLM_API_CSV, IPIP_API_CSV):
        col = pd.read_csv(path, low_memory=False, usecols=["model"])["model"]
        names |= set(col[col.str.endswith("__nr")].map(normalize_api_model))
    return names


# ============================================================
# Human per-person item data (for the human-mean reference panel)
# ============================================================


def _normalize_item_id(s: pd.Series) -> pd.Series:
    """Render item ids as canonical strings ('12' not '12.0') so the human
    table and the LLM reverse-coded metadata join on a common key."""
    def coerce(x):
        if pd.isna(x):
            return x
        try:
            f = float(x)
            return str(int(f)) if f.is_integer() else str(f)
        except (TypeError, ValueError):
            return str(x)
    return s.map(coerce)


def _load_risk_item_metadata() -> pd.DataFrame:
    """(experiment, item_id) -> (category, reverse_coded), taken from the LLM
    risk table (the human risk file carries neither the subscale category nor
    the reverse-keying flag). `category` is needed to split BARRATT into
    BISa/BISm/BISn and SSSV into SSbor/SSdis/SSexp/SStas."""
    df = pd.read_csv(LLM_RAW_CSV, low_memory=False)
    df = df[(df["context_mode"] == "no_context") & (df["flipped"] == False)]
    df = df[["experiment", "category", "item_id", "reverse_coded"]].drop_duplicates(
        ["experiment", "item_id"]
    )
    df["item_id"] = _normalize_item_id(df["item_id"])
    df["reverse_coded"] = df["reverse_coded"].astype(bool)
    return df


def _load_human_behavioral_position(meta: pd.DataFrame) -> pd.DataFrame:
    """Position-coded human responses for the behavioral tasks (LOT, DFD): the LLM
    raw response is the chosen option's POSITION, so humans must be position-coded
    too (the per-person risk file's content score "chose risky" can't show a
    keying/position shift). reverse_coded marks trials/items where the risky option
    was shown second.

    LOT: order randomised per trial (`Presentation_XZ`) -> per-trial keying; risky
    gamble is Z for Dec_ID 1-15, X for 16-25. DFD: order fixed per gamble, matches
    the LLM keying (`meta`); response = "chose option A" (position 1).
    """
    # LOT — per-trial position coding.
    lot = pd.read_csv(HUMAN_LOT_SRC, low_memory=False).dropna(
        subset=["Decision_X", "Presentation_XZ", "Dec_ID"]
    )
    dx = lot["Decision_X"].astype(int)
    pxz = lot["Presentation_XZ"].astype(int)
    risky_is_X = lot["Dec_ID"].astype(int) >= 16
    lot_out = pd.DataFrame({
        "person_id": lot["partid"].to_numpy(),
        "experiment": "LOT",
        "category": np.nan,
        "response": np.where(pxz == 1, dx, 1 - dx).astype(float),   # chose first-presented option
        "reverse_coded": np.where(risky_is_X, pxz == 0, pxz == 1),  # risky gamble shown second
    })

    # DFD — per-item position coding (order fixed per gamble == LLM keying).
    dfd_meta = meta[meta["experiment"] == "DFD"][["item_id", "reverse_coded"]]
    dfd = pd.read_csv(HUMAN_DFD_SRC, low_memory=False).dropna(subset=["decision"])
    dfd = dfd.rename(columns={"partid": "person_id", "gamble_lab": "item_id"})
    dfd["item_id"] = _normalize_item_id(dfd["item_id"])
    dfd["response"] = (dfd["decision"].astype(str).str.upper() == "A").astype(float)  # chose option A
    dfd = dfd.drop(columns=["reverse_coded"], errors="ignore").merge(
        dfd_meta, on="item_id", how="left"
    )
    dfd = dfd.dropna(subset=["reverse_coded"])
    dfd_out = pd.DataFrame({
        "person_id": dfd["person_id"].to_numpy(),
        "experiment": "DFD",
        "category": np.nan,
        "response": dfd["response"].to_numpy(),
        "reverse_coded": dfd["reverse_coded"].astype(bool).to_numpy(),
    })
    return pd.concat([lot_out, dfd_out], ignore_index=True)


def load_human_risk_items() -> pd.DataFrame:
    """Per-person human risk responses. Columns: person_id, experiment, category,
    response, reverse_coded.

    Self-report scales (BARRATT, SSSV) use the content score and the per-item
    keying from the LLM metadata (their reverse-keying is item wording, identical
    for humans and LLMs). Behavioral tasks (LOT, DFD) are position-coded by
    `_load_human_behavioral_position` so they are comparable to the LLM panels."""
    meta = _load_risk_item_metadata()
    df = pd.read_csv(HUMAN_RISK_CSV, low_memory=False)
    df["score"] = df["score"].replace({" ": np.nan})
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    df = df.rename(columns={"partid": "person_id", "item": "item_id", "score": "response"})

    # Self-report scales only here; behavioral tasks handled position-coded below.
    exp_alias = {"BARRAT scale": "BARRATT", "SSSV scale": "SSSV"}
    df["experiment"] = df["experiment"].map(exp_alias)
    df = df[df["experiment"].isin(exp_alias.values())].copy()
    df["item_id"] = _normalize_item_id(df["item_id"])

    df = df.drop(columns=["category"], errors="ignore").merge(
        meta, on=["experiment", "item_id"], how="left"
    )
    df = df.dropna(subset=["reverse_coded"])
    df["reverse_coded"] = df["reverse_coded"].astype(bool)
    selfreport = df[["person_id", "experiment", "category", "response", "reverse_coded"]]

    behavioral = _load_human_behavioral_position(meta)
    return pd.concat([selfreport, behavioral], ignore_index=True)


def load_human_ipip_items() -> pd.DataFrame:
    """Per-person human IPIP-NEO-300 responses. The Big-Five letter lives in
    `traits`; move it to `category` to match the risk-frame layout."""
    df = pd.read_csv(
        HUMAN_IPIP_CSV, low_memory=False,
        usecols=["person_id", "traits", "response", "reverse_coded"],
    )
    df = df.rename(columns={"traits": "category"})
    df["experiment"] = "IPIP-NEO-300"
    df["reverse_coded"] = df["reverse_coded"].astype(bool)
    df["response"] = pd.to_numeric(df["response"], errors="coerce")
    df = df.dropna(subset=["response"])
    return df[["person_id", "experiment", "category", "response", "reverse_coded"]]


def load_human_items() -> pd.DataFrame:
    """Union of human IPIP + risk per-person item responses."""
    return pd.concat(
        [load_human_ipip_items(), load_human_risk_items()], ignore_index=True
    )


def _subscale_slice(df: pd.DataFrame, experiment: str, category: str | None) -> pd.DataFrame:
    sub = df[df["experiment"] == experiment]
    if category is not None:
        sub = sub[sub["category"] == category]
    return sub


def _pivot(sub_df: pd.DataFrame, scale_min: int, scale_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    item_meta = sub_df.drop_duplicates("item_id").set_index("item_id")["reverse_coded"]
    forward_ids = item_meta.index[~item_meta.values.astype(bool)].to_numpy()
    reverse_ids = item_meta.index[item_meta.values.astype(bool)].to_numpy()
    wide = sub_df.pivot_table(
        index="model", columns="item_id", values="model_answer", aggfunc="first"
    )
    forward_ids = np.array([i for i in forward_ids if i in wide.columns])
    reverse_ids = np.array([i for i in reverse_ids if i in wide.columns])
    F = wide[forward_ids].to_numpy(dtype=float)
    R_raw = wide[reverse_ids].to_numpy(dtype=float)
    R_recoded = (scale_min + scale_max) - R_raw
    return F, R_recoded, wide.index.to_numpy()


# ============================================================
# Per-model risk-domain acquiescence (mu_a_risk)
# ============================================================


def compute_mu_a_risk(llm_df: pd.DataFrame) -> pd.DataFrame:
    """Per-model normalised bias and two summaries: signed mean (``mu_a_risk``)
    and mean absolute (``mu_a_risk_abs``).

        a_norm[subscale] = ((R̄_f + R̄_r)/2 - midpoint) / half_scale_range

    The absolute summary measures how much keying drives the responses regardless
    of direction; the signed summary cancels for a model that is a yea-sayer on
    some subscales and a nay-sayer on others.
    """
    rows = []
    for exp, cat, _, smin, smax in SUBSCALES:
        sub = _subscale_slice(llm_df, exp, cat)
        midpoint = (smin + smax) / 2
        half_range = (smax - smin) / 2
        g = sub.groupby(["model", "reverse_coded"])["model_answer"].mean().unstack()
        for model in g.index:
            Rf = g.loc[model].get(False, np.nan)
            Rr = g.loc[model].get(True, np.nan)
            a_norm = ((Rf + Rr) / 2 - midpoint) / half_range
            rows.append({"model": model, "a_norm": a_norm})
    per_sub = pd.DataFrame(rows)
    summary = per_sub.groupby("model")["a_norm"].agg(
        mu_a_risk="mean", mu_a_risk_abs=lambda s: np.mean(np.abs(s))
    )
    return summary


def compute_human_mu_a(human_items: pd.DataFrame) -> float:
    """Mean-absolute normalised acquiescence of the average human, computed
    over the 14 subscales exactly as compute_mu_a_risk does per model but
    using the population-mean forward/reverse responses."""
    a_norms = []
    for exp, cat, _, smin, smax in SUBSCALES:
        sub = _subscale_slice(human_items, exp, cat)
        if sub.empty:
            continue
        midpoint = (smin + smax) / 2
        half_range = (smax - smin) / 2
        Rf = sub[~sub["reverse_coded"]].groupby("person_id")["response"].mean().mean()
        Rr = sub[sub["reverse_coded"]].groupby("person_id")["response"].mean().mean()
        a_norms.append(((Rf + Rr) / 2 - midpoint) / half_range)
    return float(np.mean(np.abs(a_norms)))


# ============================================================
# Adaptive split for three target p_r values
# ============================================================


def find_closest_split(F: int, R: int, target: float, min_k: int) -> tuple[int, int, float] | None:
    """Return (k_f, k_r, actual_p_r) closest to target, maximizing k.
    Returns None if no split with k >= min_k is possible."""
    best = None  # (deviation, -k, k_f, k_r, p_actual)
    for k_r in range(0, R + 1):
        for k_f in range(0, F + 1):
            k = k_f + k_r
            if k < min_k:
                continue
            p_actual = k_r / k
            entry = (abs(p_actual - target), -k, k_f, k_r, p_actual)
            if best is None or entry < best:
                best = entry
    return None if best is None else (best[2], best[3], float(best[4]))


def bootstrap_T(
    F: np.ndarray, R_recoded: np.ndarray, k_f: int, k_r: int, rng: np.random.Generator
) -> np.ndarray:
    """Mean T per model for one random subset draw."""
    parts = []
    if k_f:
        f_idx = rng.choice(F.shape[1], size=k_f, replace=False)
        parts.append(F[:, f_idx])
    if k_r:
        r_idx = rng.choice(R_recoded.shape[1], size=k_r, replace=False)
        parts.append(R_recoded[:, r_idx])
    stacked = np.concatenate(parts, axis=1)
    return np.nanmean(stacked, axis=1)


# ============================================================
# Main computation
# ============================================================


def compute_subscale_counts(llm_df: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """(subscale label) -> (#forward items, #reverse items) from the LLM item
    pool. Reused to drive the human baseline through the *same*
    find_closest_split conditions, so the human dots land at the identical
    achievable p_r as the LLM panels (incl. BISm's p_r=1 -> 0.5 collapse)."""
    counts = {}
    for exp, cat, label, smin, smax in SUBSCALES:
        sub = _subscale_slice(llm_df, exp, cat)
        F, R_recoded, _ = _pivot(sub, smin, smax)
        counts[label] = (F.shape[1], R_recoded.shape[1])
    return counts


def compute_profiles(llm_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for exp, cat, label, smin, smax in SUBSCALES:
        sub = _subscale_slice(llm_df, exp, cat)
        F, R_recoded, models = _pivot(sub, smin, smax)
        F_count, R_count = F.shape[1], R_recoded.shape[1]

        for target in TARGET_P_R:
            split = find_closest_split(F_count, R_count, target, MIN_K_PROFILE)
            if split is None:
                print(f"  [{label}] target p_r={target}: no valid split — skipped")
                continue
            k_f, k_r, p_actual = split

            if k_f and not k_r:
                draws = np.tile(F.mean(axis=1)[None, :], (N_BOOTSTRAP, 1)) \
                    if k_f == F_count \
                    else np.stack([bootstrap_T(F, R_recoded, k_f, k_r, rng)
                                    for _ in range(N_BOOTSTRAP)])
            elif k_r and not k_f:
                draws = np.tile(R_recoded.mean(axis=1)[None, :], (N_BOOTSTRAP, 1)) \
                    if k_r == R_count \
                    else np.stack([bootstrap_T(F, R_recoded, k_f, k_r, rng)
                                    for _ in range(N_BOOTSTRAP)])
            else:
                draws = np.stack([bootstrap_T(F, R_recoded, k_f, k_r, rng)
                                  for _ in range(N_BOOTSTRAP)])

            T_mean = draws.mean(axis=0)
            T_norm = (T_mean - smin) / (smax - smin)
            for i, m in enumerate(models):
                rows.append({
                    "model": m,
                    "subscale": label,
                    "target_p_r": target,
                    "actual_p_r": p_actual,
                    "k_f": k_f,
                    "k_r": k_r,
                    "k": k_f + k_r,
                    "scale_min": smin,
                    "scale_max": smax,
                    "T_mean": T_mean[i],
                    "T_norm": T_norm[i],
                })
    return pd.DataFrame(rows)


def compute_human_profiles(
    human_items: pd.DataFrame, subscale_counts: dict[str, tuple[int, int]]
) -> pd.DataFrame:
    """Mean-human profile, same long format as compute_profiles and run through the
    identical find_closest_split conditions (shared item counts) so it is exactly
    comparable to the LLM panels. Per person the score is the linear blend
    T(p_r) = (1 - p_r)*T_fwd + p_r*T_rev at the achievable p_r -- the closed form of
    the LLM bootstrap mean (E[T] is linear in p_r), without the Monte-Carlo noise.
    """
    rows = []
    for exp, cat, label, smin, smax in SUBSCALES:
        sub = _subscale_slice(human_items, exp, cat)
        if sub.empty:
            continue
        F_count, R_count = subscale_counts.get(label, (0, 0))
        fwd = sub[~sub["reverse_coded"]].groupby("person_id")["response"].mean()
        rev = (smin + smax) - sub[sub["reverse_coded"]].groupby("person_id")["response"].mean()
        common = fwd.index.intersection(rev.index)
        if len(common) == 0:
            continue
        T_fwd = fwd.loc[common]
        T_rev = rev.loc[common]
        for target in TARGET_P_R:
            split = find_closest_split(F_count, R_count, target, MIN_K_PROFILE)
            if split is None:
                continue
            k_f, k_r, p_actual = split
            T_person = (1 - p_actual) * T_fwd + p_actual * T_rev
            T_norm_person = (T_person - smin) / (smax - smin)
            rows.append({
                "model": HUMAN_KEY,
                "subscale": label,
                "target_p_r": target,
                "actual_p_r": p_actual,
                "k_f": k_f,
                "k_r": k_r,
                "k": k_f + k_r,
                "scale_min": smin,
                "scale_max": smax,
                "T_mean": float(T_person.mean()),
                "T_norm": float(T_norm_person.mean()),
            })
    return pd.DataFrame(rows)


# ============================================================
# Figure
# ============================================================


def plot_profile(ax: plt.Axes, panel_df: pd.DataFrame, model: str,
                 mu_a_abs: float, label_order: list[str],
                 tick_order: list[str], first_in_grid: bool,
                 fill_color: str, display_name: str | None = None,
                 proprietary: bool = False) -> None:
    """Draw one model's 3-line profile on the given axis.

    `label_order` is the y-axis ordering of subscales (top→bottom = first→last),
    set once at figure-build time from the cross-model average T at p_r=0.
    `tick_order` is the matching list of compact y-tick labels.
    `fill_color` tints the keying-shift band (entity colour: purple for the
    human panel, teal for LLMs); markers take their line's grey shade.
    """
    n_rows = len(label_order)
    # Two y-blocks (personality on top, risk below) separated by BLOCK_GAP so the
    # line and band break between the scale families.
    ys = np.arange(n_rows, dtype=float)
    ys[N_BIG5:] += BLOCK_GAP
    blocks = [slice(0, N_BIG5), slice(N_BIG5, n_rows)]

    # Resolve the x-position (normalised T) of every subscale for each line,
    # NaN where the achievable p_r strays too far from target (broken line).
    xs_by_target = {}
    for target in TARGET_P_R:
        rows = panel_df[panel_df["target_p_r"] == target].set_index("subscale")
        xs = []
        for label in label_order:
            if label in rows.index and abs(rows.loc[label, "actual_p_r"] - target) <= P_R_TOLERANCE:
                xs.append(rows.loc[label, "T_norm"])
            else:
                xs.append(np.nan)
        xs_by_target[target] = np.array(xs, dtype=float)

    # Where the all-reverse extreme is unreachable (BISm has only 1 reverse
    # item, so target p_r=1 collapses to p_r=0.5), fill it with the linear
    # extrapolation T(1) = 2·T(0.5) − T(0) — exact in expectation since E[T]
    # is linear in p_r. These get a faded "proxy" dot, flagged in the caption.
    proxy_mask = (np.isnan(xs_by_target[1.0])
                  & ~np.isnan(xs_by_target[0.5])
                  & ~np.isnan(xs_by_target[0.0]))
    xs_by_target[1.0][proxy_mask] = (
        2.0 * xs_by_target[0.5][proxy_mask] - xs_by_target[0.0][proxy_mask]
    )

    # Shade the band spanned by the two extremes (p_r=0 ↔ p_r=1): its width
    # at each row is the score range attributable to keying convention alone.
    xs_lo, xs_hi = xs_by_target[0.0], xs_by_target[1.0]
    band_mask = ~np.isnan(xs_lo) & ~np.isnan(xs_hi)
    for blk in blocks:
        ax.fill_betweenx(ys[blk], xs_lo[blk], xs_hi[blk], where=band_mask[blk],
                         interpolate=False, color=fill_color, alpha=BAND_ALPHA,
                         linewidth=0, zorder=1)

    # Only the two extremes are drawn: the p_r=0.5 midpoint line is omitted (the
    # band already conveys the balanced position). Each line is split per block
    # so it breaks at the personality/risk boundary.
    for target in (0.0, 1.0):
        xs = xs_by_target[target]
        for blk in blocks:
            ax.plot(xs[blk], ys[blk], color=LINE_COLORS[target],
                    linewidth=1.1, alpha=0.9, zorder=3)
        measured = ~np.isnan(xs)
        if target == 1.0:
            measured = measured & ~proxy_mask
        ax.scatter(xs[measured], ys[measured], marker=MARKERS[target],
                   facecolor=LINE_COLORS[target], edgecolor="none", s=16,
                   linewidth=0, zorder=4)

    # Faded marker for the extrapolated p_r=1 proxy point(s).
    if proxy_mask.any():
        ax.scatter(xs_by_target[1.0][proxy_mask], ys[proxy_mask],
                   marker=MARKERS[1.0], facecolor=LINE_COLORS[1.0], alpha=0.4,
                   edgecolor="none", s=16, linewidth=0, zorder=4)


    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(ys[-1] + 0.5, ys[0] - 0.5)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", ".5", "1"], fontsize=FS_XTICK)
    ax.set_yticks(ys)
    ax.set_yticklabels(tick_order if first_in_grid else [], fontsize=FS_YTICK)
    ax.tick_params(axis="both", length=2, pad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    name = display_name if display_name is not None else clean_model_name(model)
    if proprietary:
        name = f"{name}*"   # trailing * flags proprietary-API models
    # Two-row header: model name on top (FS_TITLE), bias estimate below it in a
    # smaller font. The title pad leaves room for the bias line, which is drawn
    # just above the axes top.
    ax.set_title(name, fontsize=FS_TITLE, pad=FS_BIAS + 6)
    ax.text(0.5, 1.0, rf"$|\mu_b|={mu_a_abs:.2f}$", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=FS_BIAS)


def make_figure(profiles: pd.DataFrame, mu_a: pd.DataFrame,
                human_profiles: pd.DataFrame, human_mu_a_abs: float,
                proprietary: set[str]) -> None:
    # Y-axis subscale ordering: fixed manual order from SUBSCALES — Big
    # Five (OCEAN) on top, then BARRATT → SSSV → DFD → LOT. Lets the
    # reader scan the personality block and risk block separately.
    label_order = list(SUBSCALE_LABELS)
    # Long descriptive y-axis labels (SUBSCALE_TICK_LABELS_MAP); label_order
    # stays the canonical data key used to match rows in panel_df.
    tick_order = [SUBSCALE_TICK_LABELS_MAP[lab] for lab in label_order]

    # Model ordering: ascending by mu_a_risk_abs so reader scans from
    # "least bias-affected → fully bias-determined" profiles. The human-mean
    # reference profile is pinned to the first panel.
    ordered = [HUMAN_KEY] + mu_a.sort_values("mu_a_risk_abs").index.tolist()
    panel_profiles = pd.concat([human_profiles, profiles], ignore_index=True)
    n_cols = 9
    n_rows = int(np.ceil(len(ordered) / n_cols))

    # Panel height scales with the number of subscale rows so the y-axis
    # ticks don't crowd at higher rowcounts.
    n_subscales = len(label_order)
    panel_height = (0.20 * n_subscales + 0.45) * 0.87
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols + 1.4, panel_height * n_rows + 1.4),
        gridspec_kw=dict(wspace=0.18, hspace=0.28, left=0.21, right=0.98,
                         top=0.96, bottom=0.06),
    )
    axes_flat = axes.flatten()

    for idx, ax in enumerate(axes_flat):
        if idx >= len(ordered):
            ax.set_visible(False)
            continue
        model = ordered[idx]
        panel = panel_profiles[panel_profiles["model"] == model]
        is_human = model == HUMAN_KEY
        is_prop = (not is_human) and (model in proprietary)
        plot_profile(
            ax, panel, model,
            human_mu_a_abs if is_human else float(mu_a.loc[model, "mu_a_risk_abs"]),
            label_order=label_order, tick_order=tick_order,
            first_in_grid=(idx % n_cols == 0),
            fill_color=HUMAN_FILL if is_human else (LLM_API_FILL if is_prop else LLM_FILL),
            display_name=HUMAN_DISPLAY if is_human else None,
            proprietary=is_prop,
        )

    # Legend + x-axis caption: only the two extremes (the p_r=0.5 midpoint line
    # is removed), then the entity fill colours (human / open-weight / proprietary).
    handles = []
    for t in (0.0, 1.0):
        handles.append(Line2D([0], [0], color=LINE_COLORS[t], marker=MARKERS[t],
                              markersize=10, markerfacecolor=LINE_COLORS[t],
                              markeredgecolor=LINE_COLORS[t], linewidth=1.6,
                              label=LINE_LABELS[t]))
    handles.append(Patch(facecolor=HUMAN_FILL, alpha=0.85, linewidth=0,
                         label="humans (fill)"))
    handles.append(Patch(facecolor=LLM_FILL, alpha=0.85, linewidth=0,
                         label="open-weight LLMs (fill)"))
    handles.append(Patch(facecolor=LLM_API_FILL, alpha=0.85, linewidth=0,
                         label="proprietary LLMs* (fill)"))
    caption = "normalised subscale score  T\n(0 = scale min, 1 = scale max)"

    # The last grid row is usually only partly filled (57 panels rarely divide
    # evenly by n_cols), leaving empty cells. Park the legend + caption in that
    # block so they sit inline with the grid instead of floating far below it —
    # the tight bbox then crops the otherwise-empty bottom margin.
    empty_idx = list(range(len(ordered), n_rows * n_cols))
    if empty_idx:
        boxes = [axes_flat[i].get_position() for i in empty_idx]
        rx0, rx1 = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
        ry1 = max(b.y1 for b in boxes)        # top edge of the empty (last) row
        cx = (rx0 + rx1) / 2
        # Raise the caption off the top of the empty row toward the bottom of the
        # row above it. caption_raise in [0, 1]: 0 = top of the empty row,
        # 1 = right under the second-to-last row (higher value = closer to it).
        caption_raise = 0.5
        above = [axes_flat[i - n_cols] for i in empty_idx if i - n_cols >= 0]
        y_above = min(ax.get_position().y0 for ax in above) if above else ry1
        caption_y = ry1 + caption_raise * (y_above - ry1)
        fig.text(cx, caption_y, caption, ha="center", va="top", fontsize=FS_TITLE)
        fig.legend(handles=handles, loc="upper center", ncol=2,
                   bbox_to_anchor=(cx, ry1 - 0.040), frameon=False, fontsize=FS_TITLE)
    else:
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
                   bbox_to_anchor=(0.5, 0.01), fontsize=FS_TITLE)
        fig.supxlabel(caption.replace("\n", "  "), fontsize=FS_TITLE, y=0.02)

    fig.savefig(OUT_FIG, bbox_inches="tight")
    print(f"-> {OUT_FIG.relative_to(ROOT)}")


# ============================================================
# Entry point
# ============================================================


def main() -> None:
    print("Loading combined LLM data (risk + IPIP-NEO-300)...")
    llm_df = load_combined_reference()
    n_models = llm_df["model"].nunique()
    print(f"  {n_models} models (intersect of risk and IPIP rosters)")

    print("Computing per-model |mu_a| across all 14 scales (signed and absolute)...")
    mu_a = compute_mu_a_risk(llm_df)
    print(f"  signed   mu_a     range: {mu_a['mu_a_risk'].min():+.2f} … {mu_a['mu_a_risk'].max():+.2f}")
    print(f"  absolute |mu_a|   range: {mu_a['mu_a_risk_abs'].min():.2f} … {mu_a['mu_a_risk_abs'].max():.2f}")

    print("Computing T at p_r ∈ {0, 0.5, 1} per (model, subscale)...")
    profiles = compute_profiles(llm_df)
    profiles = profiles.merge(mu_a, on="model", how="left")
    profiles.to_csv(OUT_TABLE, index=False)
    print(f"  -> {OUT_TABLE.relative_to(ROOT)}  ({len(profiles):,} rows)")

    print("Loading human per-person item data for the mean-human panel...")
    human_items = load_human_items()
    print(f"  {human_items['person_id'].nunique():,} persons")
    subscale_counts = compute_subscale_counts(llm_df)
    human_profiles = compute_human_profiles(human_items, subscale_counts)
    human_mu_a_abs = compute_human_mu_a(human_items)
    print(f"  human |mu_a| = {human_mu_a_abs:.2f}")

    print("Rendering small-multiples figure...")
    proprietary = load_proprietary_model_names()
    print(f"  {len(proprietary)} proprietary-API models flagged (distinct fill + '*')")
    make_figure(profiles, mu_a, human_profiles, human_mu_a_abs, proprietary)


if __name__ == "__main__":
    main()
