#!/usr/bin/env python3
"""Reproduce every figure and table from the cleaned (intermediate) data.

Runs the full default reproduction path end to end:

  1. download the intermediate data from OSF  (download_data.py)
  2. run every analysis script, in paper order, writing all figures and
     tables to analysis/results/

Each analysis script is run from its own folder (the scripts use paths
relative to their location and import a sibling utils.py).

Usage
-----
    python run_all.py                  # download data, then run all analyses
    python run_all.py --skip-download  # data already present; just run analyses
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = REPO_ROOT / "analysis" / "src" / "analysis"


def analysis_scripts() -> list[Path]:
    """Every numbered analysis script, in folder-then-filename order.
    Picks up files named like ``01_*.py`` and skips helpers such as utils.py.
    """
    scripts: list[Path] = []
    for folder in sorted(p for p in ANALYSIS_DIR.iterdir() if p.is_dir()):
        scripts += sorted(folder.glob("[0-9]*.py"))
    return scripts


# Scripts exit with this code to signal "skipped" (e.g. optional Frey et al. source
# data not present) rather than a real failure; run_all reports these separately.
SKIP_EXIT_CODE = 77


def run(cmd: list[str], cwd: Path) -> int:
    """Run a command, streaming its output; return its exit code."""
    rel = cwd.relative_to(REPO_ROOT)
    print(f"\n{'=' * 70}\n>>> {' '.join(cmd)}   (in {rel})\n{'=' * 70}", flush=True)
    return subprocess.run(cmd, cwd=cwd).returncode


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--skip-download", action="store_true",
                    help="data is already in analysis/data/; skip download_data.py")
    args = ap.parse_args()

    scripts = analysis_scripts()

    if not args.skip_download:
        if run([sys.executable, "download_data.py"], cwd=REPO_ROOT) != 0:
            sys.exit("\n[abort] data download failed -- fix the above and re-run.")

    failed: list[Path] = []
    skipped: list[Path] = []
    for script in scripts:
        code = run([sys.executable, script.name], cwd=script.parent)
        if code == SKIP_EXIT_CODE:
            skipped.append(script)
        elif code != 0:
            failed.append(script)

    ok = len(scripts) - len(failed) - len(skipped)
    print(f"\n{'=' * 70}")
    print(f"[done] {ok}/{len(scripts)} scripts produced output; "
          f"{len(skipped)} skipped, {len(failed)} failed.")
    if skipped:
        print("skipped (need original Frey et al. data, see README -> Human data):")
        for s in skipped:
            print(f"  - {s.relative_to(REPO_ROOT)}")
    if failed:
        print("failed:")
        for s in failed:
            print(f"  - {s.relative_to(REPO_ROOT)}")
        sys.exit(1)
    print(f"Outputs in {(REPO_ROOT / 'analysis' / 'results').relative_to(REPO_ROOT)}/.")


if __name__ == "__main__":
    main()
