#!/usr/bin/env python3
"""Download the project data from OSF into the locations the code expects.

The data is stored on OSF as plain folder trees (not zip archives), so individual
files stay small and there is no per-file size limit to worry about. This script
mirrors the chosen folders from OSF into the local repo, preserving structure:

    OSF:  osfstorage/intermediate/...   ->  analysis/data/intermediate/...
    OSF:  osfstorage/raw/...            ->  analysis/data/raw/...
    OSF:  osfstorage/survey_data/...    ->  data_generation/hybrid/survey_data/...

Tiers
-----
  * intermediate  (~590 MB)  cleaned, person-per-item data. All you need to
                             reproduce every figure and table. Downloaded by default.
  * raw           (~10 GB)   raw LLM/human responses. Only needed to re-run the
                             preprocessing scripts (01-07). Use ``--raw``.
  * survey_data   (~200 MB)  per-participant prompts for the hybrid generation path.
                             Only needed to re-generate data. Use ``--survey-data``.

Usage
-----
    python download_data.py                  # intermediate only (default)
    python download_data.py --raw            # intermediate + raw
    python download_data.py --all            # everything
    python download_data.py --tier raw       # raw only
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

# tier name -> (OSF top-level folder, local base dir the folder is mirrored into)
TIERS = {
    "intermediate": ("intermediate", REPO_ROOT / "analysis" / "data"),
    "raw":          ("raw",          REPO_ROOT / "analysis" / "data"),
    "survey_data":  ("survey_data",  REPO_ROOT / "data_generation" / "hybrid"),
}


def _osf_storage():
    try:
        from osfclient import OSF
    except ImportError:
        sys.exit("osfclient is not installed. Run:  pip install osfclient")
    token = os.environ.get("OSF_TOKEN")  # only needed while the project is private
    osf = OSF(token=token) if token else OSF()
    return osf.project(OSF_PROJECT_ID).storage("osfstorage")


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", action="store_true", help="also download the ~10 GB raw data")
    ap.add_argument("--survey-data", action="store_true", help="also download generation prompts")
    ap.add_argument("--all", action="store_true", help="download every tier")
    ap.add_argument("--tier", choices=list(TIERS), help="download only this tier")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    ap.add_argument("--list", action="store_true", help="list tiers and exit")
    args = ap.parse_args()

    if args.list:
        print(f"OSF project: {OSF_PROJECT_ID}")
        for name, (folder, base) in TIERS.items():
            print(f"  - {name:13s} osfstorage/{folder}/ -> {base.relative_to(REPO_ROOT)}/{folder}/")
        return

    if args.tier:
        tiers = [args.tier]
    elif args.all:
        tiers = list(TIERS)
    else:
        tiers = ["intermediate"]
        if args.raw:
            tiers.append("raw")
        if args.survey_data:
            tiers.append("survey_data")

    storage = _osf_storage()
    for tier in tiers:
        print(f"  downloading '{tier}' on OSF")
        download_tier(tier, storage, force=args.force)


if __name__ == "__main__":
    main()
