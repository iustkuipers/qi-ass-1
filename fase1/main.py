"""Run the whole Part 1 pipeline, in order.

    python main.py                 every step, 01 through 08
    python main.py --from 05       from step 5 onwards (after changing mu)
    python main.py --only 06       just one step
    python main.py --list          what the steps are, without running them

Each step is a script in `scripts/` and each depends on the files the previous
one wrote, so they cannot be run out of order on a clean checkout. Every step
prints a report and also saves it under `output/`.

The steps are separate scripts rather than functions in here on purpose: the
assignment asks us to be able to show and explain each stage on its own, and a
step that takes a minute to run should not have to be re-run because a later
one changed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

FASE1 = Path(__file__).resolve().parent
SCRIPTS = FASE1 / "scripts"

STEPS: list[tuple[str, str, str]] = [
    ("01", "01_load_and_check.py",
     "load CRSP monthly and daily, check them, write clean copies"),
    ("02", "02_characteristics.py",
     "book equity, market equity and B/M with the Fama-French timing"),
    ("03", "03_betas.py",
     "pre-ranking betas, 5x5 size-beta sorts, post-ranking betas (Table I/II)"),
    ("04", "04_fama_macbeth.py",
     "cross-sectional regressions (Table III) and the CAPM test"),
    ("05", "05_expected_returns.py",
     "mu = rf + MRP * beta, plus the two robustness variants"),
    ("06", "06_portfolio.py",
     "single-index covariance and the long-only tangency portfolio"),
    ("07", "07_allocation.py",
     "y* at A = 4, whole shares, the completed hand-in sheet"),
    ("08", "08_summary_numbers.py",
     "collect every headline number into output/summary_numbers.txt"),
]


def run(step: str, script: str) -> float:
    """Run one step, letting its output through to the terminal."""
    started = time.perf_counter()
    result = subprocess.run([sys.executable, str(SCRIPTS / script)], cwd=FASE1)
    if result.returncode != 0:
        raise SystemExit(f"\nstep {step} ({script}) failed with exit code "
                         f"{result.returncode}; nothing after it was run.")
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", metavar="STEP",
                        help="first step to run, e.g. 05")
    parser.add_argument("--only", metavar="STEP", help="run a single step")
    parser.add_argument("--list", action="store_true",
                        help="show the steps and exit")
    args = parser.parse_args()

    if args.list:
        for step, script, what in STEPS:
            print(f"  {step}  {script:<26} {what}")
        return

    if args.only:
        chosen = [s for s in STEPS if s[0] == args.only.zfill(2)]
    elif args.start:
        start = args.start.zfill(2)
        chosen = [s for s in STEPS if s[0] >= start]
    else:
        chosen = STEPS

    if not chosen:
        raise SystemExit(f"no such step; run --list to see them")

    timings = []
    for step, script, what in chosen:
        print()
        print("#" * 78)
        print(f"# STEP {step}  {what}")
        print("#" * 78)
        timings.append((step, script, run(step, script)))

    print()
    print("=" * 78)
    print("PIPELINE COMPLETE")
    print("=" * 78)
    for step, script, seconds in timings:
        print(f"  {step}  {script:<26} {seconds:>7.1f}s")
    print(f"  {'':4}{'total':<26} {sum(t for _, _, t in timings):>7.1f}s")
    print()
    print(f"  reports     : {FASE1 / 'output'}")
    print(f"  deliverables: {FASE1 / 'data' / 'output'}")
    print(f"  tests       : python -m pytest tests -q")


if __name__ == "__main__":
    main()
