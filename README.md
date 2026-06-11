# Reliability and validity of human personality measures applied to LLMs

Code to reproduce the data generation as well as all analyses, figures, and tables in
**"Apparent Psychological Profiles of Large Language Models are Largely a Measurement Artifact"**
(Jelena Meyer, David Garcia, Dirk U. Wulff, 2026).

- Paper: DOI to be added
- Data: <https://osf.io/nckds/overview>

---

## What you can reproduce

| You want to...                                 | Do this                         | Needs                |
| ---------------------------------------------- | ------------------------------- | -------------------- |
| **Reproduce every figure & table** (default)   | `download_data.py`, then run the analysis scripts | Python, ~590 MB data |
| Re-run preprocessing from raw responses        | `download_data.py --raw`, then the preprocessing scripts | Python, ~10 GB data  |
| **Re-generate the LLM data from scratch**      | see [Data generation](#data-generation) | GPUs + paid API keys |

The default path downloads the cleaned (`intermediate`) data from OSF and runs the
analysis: no API keys, no GPUs, no model downloads required.

---

## Quickstart (reproduce the analysis)

```bash
# 1. Environment (tested with Python 3.11)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download the cleaned data from OSF
python download_data.py
```

Then run the analysis scripts. Outputs go to [`analysis/results/`](analysis/results/). The
analysis is one folder per Results section under
[`analysis/src/analysis/`](analysis/src/analysis/) — run the scripts in numeric order
within each. For example, to reproduce Table 1:

```bash
cd analysis/src/analysis/01_bias_not_trait
python 02_forward_reverse_corr_table.py
```

To re-run preprocessing from the raw responses, fetch the raw tier
(`python download_data.py --raw`) and run the scripts in
[`analysis/src/preprocessing/`](analysis/src/preprocessing/) first.

---

## Repository layout

```
.
├── download_data.py          # fetch data from OSF into analysis/data/
├── requirements.txt          # analysis dependencies
├── requirements-generation.txt  # extra dependencies to re-generate LLM data
│
├── data_generation/          # query LLMs + collect human data (see below)
│
├── analysis/
│   ├── src/
│   │   ├── preprocessing/     # 01–07: raw responses -> clean person-per-item data
│   │   └── analysis/          # one folder per Results section, in paper order
│   │       ├── 01_bias_not_trait/         # ρ sign-flip diagnostic, Fig 1, Table 1, negation checks
│   │       ├── 02_capability_and_bias/    # trait/bias decomposition, Fig 2, size/provider
│   │       ├── 03_orthogonality/          # reliability (Cronbach α) vs orthogonality, Fig 3
│   │       └── 04_profile_instability/    # profile shift across keyings, Fig 4
│   ├── data/                 # populated by download_data.py (git-ignored)
│   └── results/
│       ├── figures/          # regenerated PDFs
│       └── tables/           # regenerated CSV / LaTeX tables
```

---

## Data

All data is hosted on OSF as plain folder trees and fetched with
[`download_data.py`](download_data.py), which mirrors the chosen folders into the repo
(preserving structure) where the code reads from.

- **OSF project:** <https://osf.io/nckds/>

| Tier           | Mirrored into                          | Size    | Needed for                       |
| -------------- | -------------------------------------- | ------- | -------------------------------- |
| `intermediate` | `analysis/data/intermediate/`          | ~590 MB | all figures/tables (default)     |
| `raw`          | `analysis/data/raw/`                   | ~10 GB  | re-running preprocessing (01–07) |
| `survey_data`  | `data_generation/hybrid/survey_data/`  | ~200 MB | re-generating data (hybrid path) |

```bash
python download_data.py                 # intermediate only
python download_data.py --raw           # intermediate + raw
python download_data.py --all           # everything
```

**Human data.** To reproduce preprocessing from scratch you also need the original human datasets 
[`Frey et al. (2017)`](https://osf.io/rce7g/overview) and
[`Johnson et al. (2014)`](https://osf.io/tbmh5/overview). Place them in
`analysis/data/raw/ipipneo300_data/human/` and `analysis/data/raw/risk_data/human/`.

---

## Data generation

> You do **not** need to run this to reproduce the paper — the generated outputs are
> published on OSF and fetched by `download_data.py`. This folder documents (and, with the
> right hardware and API-keys, reproduces) the raw LLM responses. LLM outputs are not fully
> deterministic, so re-generated data can differ slightly from the published data.

There are two generation paths:

- **hybrid** — open-source LLMs run once per human respondent, on prompts whose context is
  derived from that participant's data (hence the large per-participant prompt files). Due to their size, these
  prompt files are **not** in the repo; fetch them from OSF first (see the command below).
- **direct** — the model is prompted on the instrument items directly, with no per-participant
  human context. Covers both open-source models (local, via `transformers`) and proprietary
  models (native provider APIs).

```bash
pip install -r requirements.txt -r requirements-generation.txt
# Set API keys for the providers you query (only the ones you use), e.g.:
#   export ANTHROPIC_API_KEY=...  OPENAI_API_KEY=...  GOOGLE_API_KEY=...
#   export DASHSCOPE_API_KEY=...  XAI_API_KEY=...   # Qwen via DashScope; xAI/Grok
#   export HF_TOKEN=...                             # gated open-source models
```

```bash
# Open-source, hybrid-with-human path
# Prerequisite: download the per-participant prompts from OSF (~200 MB) into
# data_generation/hybrid/survey_data/ — the scripts read from there.
python download_data.py --survey-data
cd data_generation/hybrid
python Calling_LLM_Models.py --model all --task-dir tasks
python run_many_models.py        # GPU-memory-safe wrapper; use if you hit OOM

# Open-source + proprietary, direct path
cd ../direct
python model_manager.py                    # open-source models (local, needs CUDA GPU)
python model_manager_native_apis.py        # proprietary API models (these calls cost money)
```

Outputs are written as response CSV with logits and metadata, then fed into the
preprocessing scripts in [`analysis/src/preprocessing/`](analysis/src/preprocessing/).

Notes:
- Open-source VRAM scales with model size (1B–70B). Most models use `transformers==5.0.0.dev0`;
  Phi models need a separate env with `transformers==4.42.3`.
- For the full CLI options of any generation script, run it with `--help`.
- The full model panel (HF names, parameter counts, release dates) is in the paper's
  supplementary table.
