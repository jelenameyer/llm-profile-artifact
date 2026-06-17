#!/usr/bin/env python3
"""Download the project data from OSF into the locations the code expects.

The data is stored on OSF as plain folder trees (not zip archives), so individual
files stay small and there is no per-file size limit to worry about. This script
mirrors the chosen folders from OSF into the local repo, preserving structure:

    OSF:  osfstorage/data/...           ->  analysis/data/...
    OSF:  osfstorage/survey_data/...    ->  data_generation/hybrid/survey_data/...

Tiers
-----
  * data          (~590 MB)  cleaned, person-per-item data. All you need to
                             reproduce every figure and table. Downloaded by default.
  * survey_data   (~200 MB)  per-participant prompts for the hybrid generation path.
                             Only needed to re-generate data. Use ``--survey-data``.

Every run except ``--tier survey_data`` also fetches two small original Frey et al. (2017)
source files (lotteries.csv, dfd_perprob.csv) that two analysis scripts read directly. These
are pulled from the *public* Frey OSF project, not re-hosted here.

Usage
-----
    python download_data.py                  # data tier only (default)
    python download_data.py --all            # every OSF-hosted tier (data + survey_data)
    python download_data.py --survey-data    # data + generation prompts
    python download_data.py --list           # show tiers and exit

For a still-private OSF project, set an access token first:
    export OSF_TOKEN=...        # https://osf.io/settings/tokens

Requires ``osfclient`` (in requirements.txt):  pip install osfclient
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURATIONS
# ---------------------------------------------------------------------------
OSF_PROJECT_ID = "nckds"

REPO_ROOT = Path(__file__).resolve().parent

# tier name -> (OSF top-level folder, local base dir the folder is mirrored into).
TIERS = {
    "data":        ("data",        REPO_ROOT / "analysis"),
    "survey_data": ("survey_data", REPO_ROOT / "data_generation" / "hybrid"),
}

# Two original Frey et al. (2017) files that two analysis scripts read directly (for
# the per-trial LOT/DFD keying the cleaned data does not carry). We do not re-host these;
# they are fetched straight from the public Frey OSF project. Maps {path on that project:
# local filename}. Paths verified against the rce7g "main" sample.
FREY_OSF_PROJECT_ID = "rce7g"
FREY_FILES = {
    "/data/main/lotteries/lotteries.csv": "lotteries.csv",
    "/data/main/dfd/dfd_perprob.csv":     "dfd_perprob.csv",
}
FREY_DEST_DIR = REPO_ROOT / "analysis" / "source" / "risk_data" / "orig_human_data"


def _osf_storage(project_id: str = OSF_PROJECT_ID, use_token: bool = True):
    try:
        from osfclient import OSF
    except ImportError:
        sys.exit("osfclient is not installed. Run:  pip install osfclient")
    token = os.environ.get("OSF_TOKEN") if use_token else None  # token only for our private project
    osf = OSF(token=token) if token else OSF()
    return osf.project(project_id).storage("osfstorage")


def download_tier(tier: str, storage, force: bool = False) -> None:
    osf_folder, local_base = TIERS[tier]
    prefix = f"/{osf_folder}/"
    n = 0
    for f in storage.files:
        if not f.path.startswith(prefix):
            continue
        n += 1
        target = local_base / f.path.lstrip("/")
        if target.exists() and not force:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {f.path} -> {target.relative_to(REPO_ROOT)}")
        with open(target, "wb") as fh:
            f.write_to(fh)
    if n == 0:
        print(f"[warn] no files found under osfstorage/{osf_folder}/ -- is it uploaded?")
    else:
        print(f"[done] {tier}: {n} files under {local_base.relative_to(REPO_ROOT)}/{osf_folder}/")


def download_frey_files(force: bool = False) -> None:
    """Fetch the two Frey et al. (2017) source files from the public Frey OSF project."""
    storage = _osf_storage(FREY_OSF_PROJECT_ID, use_token=False)
    remaining = dict(FREY_FILES)
    for f in storage.files:
        if not remaining:
            break
        dest_name = remaining.pop(f.path, None)
        if dest_name is None:
            continue
        target = FREY_DEST_DIR / dest_name
        if target.exists() and not force:
            print(f"  [Frey OSF] already have {target.relative_to(REPO_ROOT)}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [Frey OSF] {f.path} -> {target.relative_to(REPO_ROOT)}")
        with open(target, "wb") as fh:
            f.write_to(fh)
    for path in remaining:
        print(f"[warn] not found on Frey OSF project {FREY_OSF_PROJECT_ID}: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey-data", action="store_true", help="also download generation prompts")
    ap.add_argument("--all", action="store_true", help="download every OSF-hosted tier")
    ap.add_argument("--tier", choices=list(TIERS), help="download only this tier")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    ap.add_argument("--list", action="store_true", help="list tiers and exit")
    args = ap.parse_args()

    if args.list:
        print(f"OSF project: {OSF_PROJECT_ID}")
        for name, (folder, base) in TIERS.items():
            print(f"  - {name:13s} osfstorage/{folder}/ -> {(base / folder).relative_to(REPO_ROOT)}/")
        return

    if args.tier:
        tiers = [args.tier]
    elif args.all:
        tiers = list(TIERS)
    else:
        tiers = ["data"]
        if args.survey_data:
            tiers.append("survey_data")

    storage = _osf_storage()
    for tier in tiers:
        print(f"  downloading '{tier}' on OSF")
        download_tier(tier, storage, force=args.force)

    # Two analysis scripts read original Frey et al. (2017) files directly; pull them
    # straight from the public Frey OSF project (not re-hosted here). Skipped on a
    # survey-data-only run, which is unrelated to the analysis.
    if args.tier != "survey_data":
        print(f"  downloading Frey et al. (2017) source files from OSF ({FREY_OSF_PROJECT_ID})")
        download_frey_files(force=args.force)


if __name__ == "__main__":
    main()
