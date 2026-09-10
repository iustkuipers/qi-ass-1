"""Step 3 - pre-ranking betas, 5x5 size-beta portfolios, post-ranking betas.

Run from the fase1 folder:   python scripts/03_betas.py

Produces
    data/processed/assignments.csv        stock -> 5x5 cell, per FF year, with beta
    data/processed/portfolio_returns.csv  the 25 post-ranking monthly series
    data/processed/post_ranking_betas.csv the 25 betas
    output/03_betas.txt                   the report printed below
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PROCESSED, OUTPUT, SHARE_CLASS_DROPPED  # noqa: E402
from modules import module3_betas as m3  # noqa: E402


def _fmt_grid(grid: pd.DataFrame, fmt: str = "{:>8.3f}") -> list[str]:
    """Render a 5x5 grid as fixed-width text, with row and column means."""
    g = grid.copy()
    g["All"] = grid.mean(axis=1)
    g.loc["All"] = g.mean(axis=0)
    head = "      " + "".join(f"{c:>9}" for c in g.columns)
    lines = [head]
    for idx, row in g.iterrows():
        lines.append(f"{idx:>6}" + "".join(fmt.format(v) for v in row))
    return lines


def main() -> None:
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say("=" * 78)
    say("STEP 3 - PRE-RANKING BETAS, 5x5 PORTFOLIOS, POST-RANKING BETAS")
    say("=" * 78)

    monthly = pd.read_parquet(PROCESSED / "monthly.parquet")
    chars = pd.read_csv(PROCESSED / "characteristics.csv")

    # --- factors ---------------------------------------------------------
    factors, finfo = m3.load_ff_factors()
    say()
    say("FAMA-FRENCH FACTORS")
    say(f"  rows dropped for being after 31-12-2025 : {finfo['n_dropped_future']}")
    say(f"  months kept              : {finfo['n_months']} "
        f"({finfo['first']:%Y-%m} to {finfo['last']:%Y-%m})")
    say(f"  sd of mktrf              : {finfo['mktrf_sd']:.4f}  -> decimals, "
        f"not percent")
    say(f"  mean market excess return: {factors['mktrf'].mean():.4%} per month")
    say(f"  mean risk-free rate      : {factors['rf'].mean():.4%} per month")

    # --- panel -----------------------------------------------------------
    # One share class per company: the dropped classes never enter the sorts,
    # so they cannot occupy a slot in a portfolio they will not be held in.
    panel = m3.excess_return_panel(monthly, factors)
    panel = panel.drop(columns=[c for c in SHARE_CLASS_DROPPED if c in panel.columns])
    market = factors.set_index("ym")["mktrf"].reindex(panel.index)

    say()
    say("EXCESS RETURN PANEL")
    say(f"  months x stocks          : {panel.shape[0]} x {panel.shape[1]}")
    say(f"  dropped share classes    : {len(SHARE_CLASS_DROPPED)}")
    say(f"  period                   : {panel.index.min()} to {panel.index.max()}")

    # --- one beta, two ways ---------------------------------------------
    # FF regress raw returns on the raw market return; we use excess on
    # excess. This shows the choice does not matter, instead of asserting it.
    raw_panel = (monthly.assign(ym=monthly["date"].dt.to_period("M"))
                        .pivot(index="ym", columns="permno", values="ret")
                        .sort_index())
    raw_panel = raw_panel.drop(columns=[c for c in SHARE_CLASS_DROPPED
                                        if c in raw_panel.columns])
    raw_market = factors.set_index("ym")["mkt"].reindex(raw_panel.index)
    last_june = pd.Period("2025-06", freq="M")
    b_excess = m3.pre_ranking_betas(panel, market, last_june)
    b_raw = m3.pre_ranking_betas(raw_panel, raw_market, last_june)
    both = pd.concat([b_excess.rename("excess"), b_raw.rename("raw")], axis=1).dropna()
    say()
    say("BETA DEFINITION CHECK (pre-ranking betas at June 2025)")
    say(f"  stocks compared          : {len(both)}")
    say(f"  correlation excess vs raw: {both['excess'].corr(both['raw']):.6f}")
    say(f"  mean absolute difference : {(both['excess'] - both['raw']).abs().mean():.6f}")
    say("  -> excess-on-excess and FF's raw-on-raw give the same beta; we use")
    say("     excess for consistency with the step 4 regressions.")

    # --- the June sorts --------------------------------------------------
    eligible = set(panel.columns)
    assignments = m3.build_assignments(panel, market, chars, eligible=eligible)
    say()
    say("JUNE SORTS (5 size groups x 5 pre-ranking beta groups)")
    say(f"  FF years sorted          : {assignments['ff_year'].min()} to "
        f"{assignments['ff_year'].max()}")
    say(f"  stock-years assigned     : {len(assignments):,}")
    say()
    say(f"      {'year':>6} {'stocks':>7} {'per cell':>9} {'pre-beta min':>13} "
        f"{'median':>8} {'max':>8}")
    for year, g in assignments.groupby("ff_year"):
        say(f"      {year:>6} {len(g):>7} {len(g) / 25:>9.1f} "
            f"{g['pre_beta'].min():>13.2f} {g['pre_beta'].median():>8.2f} "
            f"{g['pre_beta'].max():>8.2f}")

    # --- post-ranking ----------------------------------------------------
    port = m3.portfolio_returns(assignments, panel)
    post = m3.post_ranking_betas(port, market)
    say()
    say("POST-RANKING PORTFOLIO RETURNS")
    say(f"  portfolios               : {len(post)} of 25")
    say(f"  months                   : {port['ym'].min()} to {port['ym'].max()} "
        f"({post['n_months'].max()} in the longest series)")
    say(f"  stocks per portfolio     : median {int(port['n_stocks'].median())}, "
        f"min {int(port['n_stocks'].min())}, max {int(port['n_stocks'].max())}")

    # --- Table I ---------------------------------------------------------
    say()
    say("=" * 78)
    say("TABLE I - portfolios formed on size, then on pre-ranking beta")
    say("  (Size 1 = smallest, Beta 1 = lowest pre-ranking beta)")
    say("=" * 78)

    say()
    say("Panel A: average monthly excess return, in percent")
    grid = m3.table_grid(post.assign(v=post["mean_exret"] * 100), "v")
    for ln in _fmt_grid(grid):
        say(ln)

    say()
    say("Panel B: post-ranking beta (sum of the current and lagged slopes)")
    grid = m3.table_grid(post, "post_beta")
    for ln in _fmt_grid(grid):
        say(ln)

    say()
    say("Panel C: average ln(market equity) in June")
    avg_size = (assignments.assign(ln_me=np.log(assignments["me"]))
                .groupby(["size_group", "beta_group"])["ln_me"].mean().reset_index())
    grid = m3.table_grid(avg_size, "ln_me")
    for ln in _fmt_grid(grid):
        say(ln)

    say()
    say("Panel D: average pre-ranking beta at formation")
    avg_pre = (assignments.groupby(["size_group", "beta_group"])["pre_beta"]
               .mean().reset_index())
    grid = m3.table_grid(avg_pre, "pre_beta")
    for ln in _fmt_grid(grid):
        say(ln)

    # --- does the sort work? --------------------------------------------
    say()
    say("DOES THE DOUBLE SORT DO ITS JOB?")
    size_spread = (grid.mean(axis=1).max() - grid.mean(axis=1).min())
    beta_grid = m3.table_grid(post, "post_beta")
    say(f"  post-ranking beta, lowest to highest beta group (averaged over size):")
    col_means = beta_grid.mean(axis=0)
    say("      " + "  ".join(f"{c}: {v:.2f}" for c, v in col_means.items()))
    say(f"  spread in post-ranking beta across beta groups : "
        f"{col_means.max() - col_means.min():.2f}")
    row_means = beta_grid.mean(axis=1)
    say(f"  spread in post-ranking beta across size groups : "
        f"{row_means.max() - row_means.min():.2f}")
    size_means = m3.table_grid(avg_size, "ln_me").mean(axis=1)
    say(f"  spread in ln(ME) across size groups            : "
        f"{size_means.max() - size_means.min():.2f}")
    say(f"  spread in ln(ME) across beta groups            : "
        f"{m3.table_grid(avg_size, 'ln_me').mean(axis=0).max() - m3.table_grid(avg_size, 'ln_me').mean(axis=0).min():.2f}")

    # --- Table II --------------------------------------------------------
    say()
    say("=" * 78)
    say("TABLE II - is there a relation between beta and average return?")
    say("=" * 78)
    say()
    say("  averaged over size groups, by post-ranking beta group:")
    say(f"      {'group':>8} {'post-beta':>10} {'mean exret %':>13} "
        f"{'sd %':>8} {'n months':>9}")
    by_beta = post.groupby("beta_group").agg(
        post_beta=("post_beta", "mean"), mean_exret=("mean_exret", "mean"),
        sd=("sd_exret", "mean"), n=("n_months", "max"))
    for g, r in by_beta.iterrows():
        say(f"      {g:>8} {r['post_beta']:>10.2f} {r['mean_exret'] * 100:>13.3f} "
            f"{r['sd'] * 100:>8.2f} {int(r['n']):>9}")

    say()
    say("  averaged over beta groups, by size group:")
    say(f"      {'group':>8} {'post-beta':>10} {'mean exret %':>13} "
        f"{'ln(ME)':>8}")
    by_size = post.groupby("size_group").agg(
        post_beta=("post_beta", "mean"), mean_exret=("mean_exret", "mean"))
    sz = avg_size.groupby("size_group")["ln_me"].mean()
    for g, r in by_size.iterrows():
        say(f"      {g:>8} {r['post_beta']:>10.2f} {r['mean_exret'] * 100:>13.3f} "
            f"{sz[g]:>8.2f}")

    # --- stock-level betas ----------------------------------------------
    assignments = m3.assign_post_ranking_betas(assignments, post)
    say()
    say("STOCK-LEVEL POST-RANKING BETAS (each stock gets its portfolio's beta)")
    say(f"  stock-years with a beta  : {int(assignments['post_beta'].notna().sum()):,}")
    say(f"  beta range               : {assignments['post_beta'].min():.2f} to "
        f"{assignments['post_beta'].max():.2f}")
    for year in [assignments["ff_year"].max()]:
        g = assignments[assignments["ff_year"] == year]
        say(f"  FF year {year}: {len(g)} stocks, beta {g['post_beta'].min():.2f} "
            f"to {g['post_beta'].max():.2f}")

    # --- write -----------------------------------------------------------
    assignments.to_csv(PROCESSED / "assignments.csv", index=False)
    port.to_csv(PROCESSED / "portfolio_returns.csv", index=False)
    post.to_csv(PROCESSED / "post_ranking_betas.csv", index=False)
    say()
    say("WRITTEN")
    for f in ["assignments.csv", "portfolio_returns.csv", "post_ranking_betas.csv"]:
        say(f"  {PROCESSED / f}")

    report = OUTPUT / "03_betas.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport: {report}")


if __name__ == "__main__":
    main()
