"""Shared definitions of the LLM panel, imported by several analysis scripts
(not an analysis script itself: run_all.py only runs files named ``NN_*.py``).

  MODEL_SIZE / MODEL_SIZE_ACTIVE   parameter counts (B) of the open-weight models
  MODEL_METADATA                   model_metadata.csv (dates, sizes, capability scores)
  bare_model_name()                one name per model across the IPIP and risk tables
  low_variability_models()         the constant / low-variability exclusion sets

Exclusion rule. A model's within-model SD is the SD of its raw (not recoded)
answers over the 300 IPIP-NEO-300 items in the reference condition (no context,
no flip; non-reasoning for API models). Three nested sets:
  constant   SD = 0
  below_p1   SD below the 1st percentile of the human within-person SD
  below_p5   SD below the 5th percentile (sensitivity check)
Human SDs are computed the same way (raw answers, all answered items, N = 20,993).
The anchor is strict: a model in below_p1 varies less than 99% of people, which
does not by itself show that it is disengaged.
"""
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "ipipneo300_data"

# ============================================================
# Model metadata (model_metadata.csv, next to this file): release date, parameter
# counts, released dtype and published capability scores for all 60 models, each
# value with its source URL (access dates in the columns access_date and
# eci_access_date). Release dates are those of the weights version administered. Keys (panel_key) are the model strings of the
# analysis frames: bare names for local-inference models, API keys without the
# condition suffix for the large open-weight and proprietary models.
# ============================================================

MODEL_METADATA = pd.read_csv(Path(__file__).resolve().parent / "model_metadata.csv")


def _analysis_key(panel_key: str) -> str:
    # the large open-weight models enter the analysis frames as their __nr runs
    return f"{panel_key}__nr" if panel_key.startswith("vllm__") else panel_key


def _as_number(x: float):
    return int(x) if float(x).is_integer() else float(x)


_open = MODEL_METADATA[MODEL_METADATA["params_total_B"].notna()]

# Parameter counts (B) of the 50 open-weight models, as used in the analyses
# (nominal sizes; model-card and safetensors counts are separate columns of the
# CSV). Membership also defines "open-weight" vs "proprietary" in the analyses.
MODEL_SIZE = {_analysis_key(k): _as_number(v)
              for k, v in zip(_open["panel_key"], _open["params_total_B"])}

# Active parameters per token (B): equal to the total for dense models.
MODEL_SIZE_ACTIVE = {_analysis_key(k): _as_number(v)
                     for k, v in zip(_open["panel_key"], _open["params_active_B"])}

# The four >=300B open-weight MoE models administered under the proprietary
# protocol (reference condition only).
LARGE_OPEN_MODELS = {_analysis_key(k) for k in
                     MODEL_METADATA.loc[MODEL_METADATA["group"] == "large open", "panel_key"]}


def bare_model_name(name: str) -> str:
    """Collapse a model string to its bare model name so one model lines up across
    the IPIP and risk tables (same rule as normalize_api_model in
    04_profile_instability/01_profiles_per_keying.py). API keys are
    ``provider__model-id__cond``, vLLM keys ``vllm__org__model-id__cond``; the two
    API runs disagree on a few tags (gemini ``-preview``, dates, grok variant)."""
    if "__" not in name:
        return name
    parts = name.split("__")
    model_id = parts[2] if parts[0] == "vllm" else parts[1]
    model_id = re.sub(r"-preview$", "", model_id)
    model_id = re.sub(r"-\d{4}-(?:non-)?reasoning$", "", model_id)
    model_id = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model_id)
    return model_id


# ============================================================
# Constant / low-variability responders
# ============================================================

def _human_within_person_sd() -> pd.Series:
    df = pd.read_csv(DATA_DIR / "human_data/ipipneo_human_raw.csv", low_memory=False,
                     usecols=["person_id", "response"])
    df["response"] = pd.to_numeric(
        df["response"].replace({"0": None, 0: None, " ": None}), errors="coerce")
    df = df[df["response"].between(1, 5)]
    return df.groupby("person_id")["response"].std()


def _model_within_sd() -> pd.Series:
    llm = pd.read_csv(DATA_DIR / "llm_data/ipipneo_no_flip_data_raw.csv", low_memory=False)
    llm = llm[llm["context_mode"] == "no_context"]
    api = pd.read_csv(DATA_DIR / "api_data/ipipneo_api_data_raw.csv", low_memory=False)
    api = api[api["model"].str.endswith("__nr")]
    df = pd.concat([llm, api], ignore_index=True)
    df["model"] = df["model"].map(bare_model_name)
    df["model_answer"] = pd.to_numeric(df["model_answer"], errors="coerce")
    return df.groupby("model")["model_answer"].std()


def low_variability_models():
    """Returns (sets, table). sets: {"constant", "below_p1", "below_p5"} -> set of
    bare model names (nested). table: per-model SD, its percentile among humans,
    and set membership, sorted by SD; its attrs hold the two human thresholds."""
    human_sd = _human_within_person_sd()
    p1, p5 = human_sd.quantile(0.01), human_sd.quantile(0.05)
    sd = _model_within_sd()
    table = pd.DataFrame({
        "sd": sd,
        "human_percentile": sd.map(lambda v: 100 * (human_sd < v).mean()),
        "constant": sd == 0,
        "below_p1": sd < p1,
        "below_p5": sd < p5,
    }).sort_values("sd")
    table.attrs.update(human_p1=p1, human_p5=p5, n_humans=len(human_sd))
    sets = {k: set(table.index[table[k]]) for k in ["constant", "below_p1", "below_p5"]}
    return sets, table
