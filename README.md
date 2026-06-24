# Apparent Psychological Profiles of Large Language Models are Largely a Measurement Artifact

Code to reproduce the data generation as well as all analyses, figures, and tables in
**"Apparent Psychological Profiles of Large Language Models are Largely a Measurement Artifact"**
(Jelena Meyer, David Garcia, Dirk U. Wulff, 2026).

- Paper: DOI to be added
- Preprint: <https://arxiv.org/abs/2606.20205>
- Data: <https://osf.io/nckds/overview>

---

## What you can reproduce

| You want to...                                 | Do this                         | Needs                |
| ---------------------------------------------- | ------------------------------- | -------------------- |
| **Reproduce every figure & table** (default)   | `python run_all.py` (one command) | Python, ~590 MB data |
| Re-run preprocessing from raw responses        | [request the source data](#requesting-the-source-data), then the preprocessing scripts | Python, ~10 GB data  |
| **Re-generate the LLM data from scratch**      | see [Data generation](#data-generation) | GPUs + paid API keys |

The default path downloads the cleaned (`data`) tier from OSF and runs the
analysis: no API keys, no GPUs, no model downloads required.

---

## Quickstart (reproduce the analysis)

```bash
# 1. Environment — needs Python 3.11 (tested with 3.11.9). Check your version first:
python3 --version          # should report 3.11.x; if not, see the note below
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download the data and reproduce every figure and table, in one command:
python run_all.py
```

> If `python3` is not 3.11 on your machine, point the venv at a 3.11 interpreter explicitly,
> then re-run the two commands above (`python3 -m venv .venv && source .venv/bin/activate`).
> - **pyenv:** `pyenv install -s 3.11.9 && pyenv local 3.11.9` (`-s` skips the prompt if 3.11.9
>   is already installed); confirm with `python3 --version`.
> - **conda:** `conda create -n llm-artifact python=3.11 && conda activate llm-artifact`
>   (with conda you can skip the `venv` step — the conda env already isolates packages).

[`run_all.py`](run_all.py) downloads the cleaned data from OSF (via
[`download_data.py`](download_data.py)) and then runs every analysis script in paper order,
writing all outputs to [`analysis/results/`](analysis/results/). If the data is already
downloaded, use `python run_all.py --skip-download`.

Two outputs — the risk forward–reverse SI table and the profile-instability figure (Fig 4)
— additionally need two original Frey et al. (2017) files. These are **fetched automatically**
by `download_data.py` straight from the public Frey OSF project (not re-hosted here), so no
manual step is needed. If they are ever unavailable (e.g. an offline `--skip-download` run),
`run_all.py` simply **skips** those two scripts and produces everything else.

To run a **single** figure or table instead, the analysis is organized as one folder per
Results section under [`analysis/src/analysis/`](analysis/src/analysis/); run a script from
within its folder. For example, to reproduce Table 1:

```bash
# 2b. (optional) download only, then run one script
python download_data.py
cd analysis/src/analysis/01_bias_not_trait
python 02_forward_reverse_corr_table.py
```

To re-run preprocessing from the raw responses, first
[request the source data](#requesting-the-source-data) (it is not hosted on OSF),
place it under `analysis/source/`, and run the scripts in
[`analysis/src/preprocessing/`](analysis/src/preprocessing/).

---

## Repository layout

```
.
├── run_all.py                # download data + reproduce every figure & table
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
│   ├── data/                 # cleaned data, populated by download_data.py (git-ignored)
│   ├── source/               # raw responses (on request, not on OSF); git-ignored
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
| `data`         | `analysis/data/`                       | ~590 MB | all figures/tables (default)     |
| `survey_data`  | `data_generation/hybrid/survey_data/`  | ~200 MB | re-generating data (hybrid path) |
| `source`       | `analysis/source/`                     | ~10 GB  | re-running preprocessing (01–07) — **not on OSF, [available on request](#requesting-the-source-data)** |

```bash
python download_data.py                 # data tier only
python download_data.py --survey-data   # data + generation prompts
python download_data.py --all           # every OSF-hosted tier (data + survey_data)
python download_data.py --source        # prints how to request the source data
```

### Requesting the source data

To keep the public archive manageable, the **source data** (the raw LLM responses, ~10 GB)
is **not hosted on OSF**. It is available from the authors on request — email
**wulff@mpib-berlin.mpg.de**. You only need it to re-run the preprocessing scripts
(01–07) from scratch; the default analysis path uses the cleaned `data` tier and
needs nothing else. Once received, place it under `analysis/source/` (preserving the
folder structure) before running the preprocessing scripts.

**Human data.** The original human datasets are not re-hosted here; they come from their
original sources, [`Frey et al. (2017)`](https://osf.io/rce7g/overview) and
[`Johnson et al. (2014)`](https://osf.io/tbmh5/overview).

- Two **analysis** scripts read two original Frey files directly (`lotteries.csv`,
  `dfd_perprob.csv`, for the per-trial LOT/DFD keying the cleaned data does not carry).
  `download_data.py` pulls these two files **automatically** from the public Frey OSF project,
  so the default analysis runs with no manual step.
- To re-run **preprocessing** from scratch you need the *full* original datasets. Download
  them from the two sources above and place them in
  `analysis/source/ipipneo300_data/human/` and `analysis/source/risk_data/human/`.

---

## Data generation

> You do **not** need to run this to reproduce the paper — the cleaned data needed for every
> figure and table is published on OSF and fetched by `download_data.py`. This folder
> documents (and, with the right hardware and API-keys, reproduces) the raw LLM responses
> themselves, which are [available on request](#requesting-the-source-data) rather than on
> OSF. LLM outputs are not fully deterministic, so re-generated data can differ slightly from
> the data underlying the paper.

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
