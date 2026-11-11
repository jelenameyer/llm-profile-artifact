#!/usr/bin/env python3
"""Download the project data from OSF into the locations the code expects.

The data is stored on OSF as plain folder trees (not zip archives), so individual
files stay small and there is no per-file size limit to worry about. This script
mirrors the chosen folders from OSF into the local repo, preserving structure:

    OSF:  osfstorage/data/...           ->  analysis/data/...
    OSF:  osfstorage/survey_data/...    ->  data_generation/hybrid/survey_data/...
    OSF:  osfstorage/source/...         ->  analysis/source/...

Tiers
-----
  * data          (~590 MB)  cleaned, person-per-item data. All you need to
                             reproduce every figure and table. Downloaded by default.
  * survey_data   (~200 MB)  per-participant prompts for the hybrid generation path.
                             Only needed to re-generate data. Use ``--survey-data``.
  * source        (~10 GB)   raw LLM responses. Only needed to re-run the preprocessing
                             scripts. Use ``--source``.

The human reference data are not re-hosted here; the files the code needs are fetched
straight from their public OSF projects (Frey et al., 2017; Johnson, 2005/2014):
  * every run except ``--tier survey_data``: two Frey files that some analysis scripts
    read directly (lotteries.csv, dfd_perprob.csv);
  * runs that include the source tier: the remaining human files the preprocessing
    scripts need (~30 MB).

Usage
-----
    python download_data.py                  # data tier only (default)
    python download_data.py --all            # every tier (data + survey_data + source)
    python download_data.py --survey-data    # data + generation prompts
    python download_data.py --source         # data + raw LLM responses (~10 GB)
    python download_data.py --list           # show tiers and exit

For a still-private OSF project, set an access token first:
    export OSF_TOKEN=...        # https://osf.io/settings/tokens

Requires ``osfclient`` (in requirements.txt):  pip install osfclient
"""
from __future__ import annotations

import argparse
import os
import sys
import time
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
    "source":      ("source",      REPO_ROOT / "analysis"),
}

# Original human data files, fetched from their public OSF projects (not re-hosted here):
# (OSF project, path on that project, local folder, needed for). "analysis" files are read
# directly by some analysis scripts (per-trial LOT/DFD keying the cleaned data does not
# carry); "preprocessing" files are only needed to rebuild the cleaned data from source.
FREY = "rce7g"         # Frey et al. (2017), risk battery
JOHNSON_DATA = "sxeq5"  # Johnson (2005), IPIP-NEO-300 responses
JOHNSON_KEY = "ycvdk"   # IPIP-NEO-300 scoring key
RISK_HUMAN = REPO_ROOT / "analysis" / "source" / "risk_data" / "orig_human_data"
IPIP_HUMAN = REPO_ROOT / "analysis" / "source" / "ipipneo300_data" / "human_data"
HUMAN_FILES = [
    (FREY, "/data/main/lotteries/lotteries.csv", RISK_HUMAN, "analysis"),
    (FREY, "/data/main/dfd/dfd_perprob.csv",     RISK_HUMAN, "analysis"),
    (FREY, "/data/main/quest/quest_raw.csv",     RISK_HUMAN, "preprocessing"),
    (FREY, "/data/main/quest/quest_proc.csv",    RISK_HUMAN, "preprocessing"),
    (FREY, "/data/main/bart/bart_pumps.csv",     RISK_HUMAN, "preprocessing"),
    (FREY, "/data/main/cct/cct.csv",             RISK_HUMAN, "preprocessing"),
    (FREY, "/data/main/dfe/dfe_perprob.csv",     RISK_HUMAN, "preprocessing"),
    (FREY, "/data/main/mpl/mpl.csv",             RISK_HUMAN, "preprocessing"),
    (JOHNSON_DATA, "/ipip20993.dat",             IPIP_HUMAN, "preprocessing"),
    (JOHNSON_KEY,  "/IPIP-NEO-ItemKey.xls",      IPIP_HUMAN, "preprocessing"),
]


def _osf_storage(project_id: str = OSF_PROJECT_ID, use_token: bool = True):
    try:
        from osfclient import OSF
    except ImportError:
        sys.exit("osfclient is not installed. Run:  pip install osfclient")
    token = os.environ.get("OSF_TOKEN") if use_token else None  # token only for our private project
    osf = OSF(token=token) if token else OSF()
    return osf.project(project_id).storage("osfstorage")


# OSF rate-limits API requests (HTTP 429), so list folders 100 entries per request
# (the default is 10), walk only the folder that is needed, and retry with backoff.
PAGE_SIZE = 100
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_TRIES = 10


def _get(session, url, **kwargs):
    for attempt in range(MAX_TRIES):
        r = session.get(url, **kwargs)
        if r.status_code not in RETRY_STATUS:
            return r
        try:
            wait = float(r.headers.get("Retry-After", ""))
        except ValueError:
            wait = min(5 * 2 ** attempt, 300)
        print(f"  [osf] HTTP {r.status_code}, retrying in {wait:.0f}s", flush=True)
        r.close()
        time.sleep(wait)
    raise RuntimeError(f"OSF request failed after {MAX_TRIES} tries ({r.status_code}): {url}")


def _list(session, url):
    """All entries (files and folders) directly inside one OSF folder."""
    entries, params = [], {"page[size]": PAGE_SIZE}
    while url:
        r = _get(session, url, params=params)
        if r.status_code != 200:
            raise RuntimeError(f"OSF listing failed ({r.status_code}): {url}")
        page = r.json()
        entries.extend(page["data"])
        url, params = page["links"].get("next"), None  # the next link carries the params
    return entries


def _children_url(entry):
    return entry["relationships"]["files"]["links"]["related"]["href"]


def _find_folder(storage, path: str):
    """Files URL of the folder at ``path`` (e.g. "data/main"), descending by name."""
    url = storage._files_url
    for name in path.strip("/").split("/"):
        match = [e for e in _list(storage.session, url)
                 if e["attributes"]["kind"] == "folder" and e["attributes"]["name"] == name]
        if not match:
            return None
        url = _children_url(match[0])
    return url


def _walk(session, url):
    """Yield every file entry below the folder at ``url``, recursively."""
    for e in _list(session, url):
        if e["attributes"]["kind"] == "folder":
            yield from _walk(session, _children_url(e))
        else:
            yield e


def _download(session, entry, target: Path) -> None:
    """Download via a temporary file, so an interrupted run leaves no partial file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    with _get(session, entry["links"]["download"], stream=True) as r:
        if r.status_code != 200:
            raise RuntimeError(f"download failed ({r.status_code}): {entry['attributes']['materialized_path']}")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    tmp.replace(target)


def download_tier(tier: str, storage, force: bool = False) -> None:
    osf_folder, local_base = TIERS[tier]
    url = _find_folder(storage, osf_folder)
    if url is None:
        print(f"[warn] no folder osfstorage/{osf_folder}/ on OSF -- is it uploaded?")
        return
    n = 0
    for e in _walk(storage.session, url):
        n += 1
        path = e["attributes"]["materialized_path"]
        target = local_base / path.lstrip("/")
        if target.exists() and not force:
            continue
        print(f"  {path} -> {target.relative_to(REPO_ROOT)}", flush=True)
        _download(storage.session, e, target)
    if n == 0:
        print(f"[warn] no files found under osfstorage/{osf_folder}/ -- is it uploaded?")
    else:
        print(f"[done] {tier}: {n} files under {local_base.relative_to(REPO_ROOT)}/{osf_folder}/")


def download_human_files(purposes: set[str], force: bool = False) -> None:
    """Fetch the original human data files from their public OSF projects."""
    storages = {}
    for project, path, dest_dir, purpose in HUMAN_FILES:
        if purpose not in purposes:
            continue
        folder, name = path.rsplit("/", 1)
        target = dest_dir / name
        if target.exists() and not force:
            print(f"  [osf.io/{project}] already have {target.relative_to(REPO_ROOT)}")
            continue
        if project not in storages:
            storages[project] = _osf_storage(project, use_token=False)
        storage = storages[project]
        url = _find_folder(storage, folder) if folder else storage._files_url
        entry = None if url is None else next(
            (e for e in _list(storage.session, url) if e["attributes"]["name"] == name), None)
        if entry is None:
            print(f"[warn] not found on OSF project {project}: {path}")
            continue
        print(f"  [osf.io/{project}] {path} -> {target.relative_to(REPO_ROOT)}", flush=True)
        _download(storage.session, entry, target)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey-data", action="store_true", help="also download generation prompts")
    ap.add_argument("--source", action="store_true", help="also download the raw LLM responses (~10 GB)")
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
        if args.source:
            tiers.append("source")

    storage = _osf_storage()
    for tier in tiers:
        print(f"  downloading '{tier}' on OSF")
        download_tier(tier, storage, force=args.force)

    # Original human files, from their own public OSF projects. A survey-data-only run
    # is unrelated to the analysis and fetches none.
    purposes = set()
    if args.tier != "survey_data":
        purposes.add("analysis")
    if "source" in tiers:
        purposes.add("preprocessing")
    if purposes:
        print("  downloading original human data files from their OSF projects")
        download_human_files(purposes, force=args.force)


if __name__ == "__main__":
    main()
