"""Step 5 - expected returns.

Run from the fase1 folder:   python scripts/05_expected_returns.py

Produces
    data/processed/expected_returns.csv   mu for every investable stock
    output/05_expected_returns.txt        the report printed below
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (EXCLUDE_NO_COMPUSTAT, MONTHS_PER_YEAR, OUTPUT, PROCESSED,  # noqa: E402
                    RF_ANNUAL, SHARE_CLASS_DROPPED, TOO_SHORT_HISTORY)
from modules import module3_betas as m3  # noqa: E402
from modules import module5_expected_returns as m5  # noqa: E402

FF_YEAR = 2025          # the portfolio is formed at the end of December 2025


def main() -> None:
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say("=" * 78)
    say("STEP 5 - EXPECTED RETURNS")
    say("=" * 78)

    assignments = pd.read_csv(PROCESSED / "assignments.csv")
    chars = pd.read_csv(PROCESSED / "characteristics.csv")
    universe = pd.read_csv(PROCESSED / "universe.csv")
    fm = pd.read_csv(PROCESSED / "fm_summary.csv")

    tickers = universe.drop_duplicates("permno").set_index("permno")["ticker"]

    # --- the market risk premium -----------------------------------------
    say()
    say("MARKET RISK PREMIUM")
    long_path = m5.LONG_FACTOR_FILE
    if long_path.exists():
        long_factors = m5.load_long_factors(long_path)
        mrp = m5.market_risk_premium(long_factors)
        source = f"{long_path.name} (long history)"
        placeholder = False
    else:
        short_factors, _ = m3.load_ff_factors()
        mrp = m5.market_risk_premium(short_factors.rename(columns={"date": "date"}))
        source = "famafrench.csv, 2006-2025 -- PLACEHOLDER"
        placeholder = True

    say(f"  source                   : {source}")
    say(f"  period                   : {mrp['first']:%Y-%m} to {mrp['last']:%Y-%m} "
        f"({mrp['n_months']} months)")
    say(f"  mean monthly mktrf       : {mrp['monthly']:.4%}  (t = {mrp['t']:.2f})")
    say(f"  annualised (x12)         : {mrp['annual']:.2%}")
    say(f"  annualised sd            : {mrp['annual_sd']:.2%}")
    if placeholder:
        say()
        say("  !! famafrench_long.csv is not in data/raw yet, so this is the")
        say("     in-sample 2006-2025 premium, which is exactly the number the")
        say("     group decided NOT to use. Drop the long file in and re-run.")
        say("     Note this does not change the tangency weights at all: with")
        say("     mu - rf = MRP * beta, the MRP is a positive scalar on the")
        say("     whole vector and cancels out of the portfolio direction. It")
        say("     changes the reported expected return and Sharpe ratio, and it")
        say("     will matter in step 7 for the risk-free split.")

    # --- who is investable ------------------------------------------------
    latest = assignments[assignments["ff_year"] == FF_YEAR].set_index("permno")
    all_firms = universe.loc[~universe["is_duplicate_row"], "permno"]

    excluded = {}
    for permno in all_firms:
        if permno in SHARE_CLASS_DROPPED:
            excluded[permno] = "second share class"
        elif permno in TOO_SHORT_HISTORY:
            excluded[permno] = "under 24 months of history"
        elif permno in EXCLUDE_NO_COMPUSTAT:
            excluded[permno] = "no Compustat data (group decision)"
        elif permno not in latest.index:
            excluded[permno] = "no 2025 post-ranking beta"

    investable = [p for p in all_firms if p not in excluded]

    say()
    say("INVESTABLE UNIVERSE")
    say(f"  firms in the hand-in sheet : {len(all_firms)}")
    for reason in ["second share class", "under 24 months of history",
                   "no Compustat data (group decision)", "no 2025 post-ranking beta"]:
        names = [p for p, r in excluded.items() if r == reason]
        if names:
            say(f"  minus {reason:<34}: {len(names):>3}  "
                f"({', '.join(str(tickers.get(p, p)) for p in names)})")
    say(f"  investable                 : {len(investable)}")

    lapsed = [p for p in EXCLUDE_NO_COMPUSTAT if p in latest.index]
    if lapsed:
        detail = ", ".join(
            f"{tickers.get(p)} {latest.loc[p, 'post_beta']:.3f}" for p in lapsed)
        say()
        say(f"  Note: ETN and DAY do have a 2025 post-ranking beta ({detail}).")
        say("  They were excluded because they have no B/M, but the expected")
        say("  return we settled on uses only beta, so that reason has lapsed.")
        say("  Clearing EXCLUDE_NO_COMPUSTAT in config.py puts them back.")

    # --- the main expected returns ---------------------------------------
    betas = latest.loc[investable, "post_beta"]
    mu = m5.mu_capm(betas, mrp["monthly"], RF_ANNUAL)
    mu["ticker"] = [tickers.get(p) for p in mu.index]

    say()
    say("MAIN SPECIFICATION:  E[r] - rf = MRP * beta,  intercept 0")
    say(f"  stocks                   : {len(mu)}")
    say(f"  distinct betas           : {betas.nunique()} "
        f"(one per 5x5 cell, as FF assign them)")
    say(f"  beta range               : {betas.min():.3f} to {betas.max():.3f}")
    say(f"  expected excess return   : {mu['mu_excess'].min():.2%} to "
        f"{mu['mu_excess'].max():.2%}")
    say(f"  cross-sectional mean     : {mu['mu_excess'].mean():.2%}")
    say()
    say("  Because every stock in a 5x5 cell gets the same beta, there are only")
    say("  25 distinct expected returns. Within a cell the optimiser therefore")
    say("  chooses purely on covariance - it takes the best-diversifying")
    say("  combination of stocks that share an expected return.")

    say()
    say("  expected excess return by 5x5 cell:")
    cell = (latest.loc[investable]
            .assign(mu=mu["mu_excess"])
            .groupby(["size_group", "beta_group"])
            .agg(n=("post_beta", "size"), beta=("post_beta", "first"),
                 mu=("mu", "first")).reset_index())
    say(f"      {'size':>5} {'beta grp':>9} {'n':>4} {'beta':>7} {'mu-rf':>8}")
    for _, r in cell.iterrows():
        say(f"      {int(r['size_group']):>5} {int(r['beta_group']):>9} "
            f"{int(r['n']):>4} {r['beta']:>7.3f} {r['mu']:>8.2%}")

    # --- robustness variants ---------------------------------------------
    slopes = (fm[(fm["spec"] == "(6) beta + ln(ME) + ln(BE/ME)")
                 & (fm["sample"] == "main (no financials, no negative BE)")]
              .set_index("variable")["mean"])
    ch25 = chars[chars["ff_year"] == FF_YEAR].set_index("permno")
    ch25 = ch25.reindex(investable)
    ch25["beta"] = betas
    anchor = mrp["monthly"] * MONTHS_PER_YEAR * betas.mean()

    say()
    say("ROBUSTNESS VARIANTS (reported only, not the portfolio we submit)")
    say(f"  both anchored so the average expected excess return equals "
        f"{anchor:.2%},")
    say(f"  which is MRP x average beta. Only the tilts differ.")

    variants = {}
    spec6 = {"beta": slopes["beta"], "ln_me": slopes["ln_me"], "ln_bm": slopes["ln_bm"]}
    variants["(a) spec (6), level anchored"] = m5.mu_from_slopes(ch25, spec6, anchor)
    no_size = {"beta": slopes["beta"], "ln_bm": slopes["ln_bm"]}
    variants["(b) beta + B/M, no size"] = m5.mu_from_slopes(ch25, no_size, anchor)

    out = mu.copy()
    say()
    say(f"      {'variant':<32} {'min':>9} {'mean':>9} {'max':>9} {'sd':>9} {'n':>5}")
    say(f"      {'main: MRP x beta':<32} {mu['mu_excess'].min():>9.2%} "
        f"{mu['mu_excess'].mean():>9.2%} {mu['mu_excess'].max():>9.2%} "
        f"{mu['mu_excess'].std():>9.2%} {mu['mu_excess'].notna().sum():>5}")
    for name, series in variants.items():
        col = "mu_" + ("a" if name.startswith("(a)") else "b")
        out[col] = series
        say(f"      {name:<32} {series.min():>9.2%} {series.mean():>9.2%} "
            f"{series.max():>9.2%} {series.std():>9.2%} {series.notna().sum():>5}")

    say()
    say("  correlation between the three sets of expected returns:")
    corr = out[["mu_excess", "mu_a", "mu_b"]].corr()
    say(f"      {'':>10} {'main':>8} {'(a)':>8} {'(b)':>8}")
    for label, row in zip(["main", "(a)", "(b)"], corr.to_numpy()):
        say(f"      {label:>10} " + "".join(f"{v:>8.3f}" for v in row))
    say()
    say("  Variants (a) and (b) leave some stocks without an expected return,")
    say("  because they need B/M and 9 investable firms do not have one.")

    out["ln_me"] = ch25["ln_me"]
    out["ln_bm"] = ch25["ln_bm"]
    out["size_group"] = latest.loc[investable, "size_group"]
    out["beta_group"] = latest.loc[investable, "beta_group"]
    out.index.name = "permno"
    out.to_csv(PROCESSED / "expected_returns.csv")

    pd.Series({"mrp_monthly": mrp["monthly"], "mrp_annual": mrp["annual"],
               "placeholder": placeholder, "source": source,
               "anchor_excess": anchor}).to_csv(PROCESSED / "mrp.csv")

    say()
    say("WRITTEN")
    say(f"  {PROCESSED / 'expected_returns.csv'}")
    say(f"  {PROCESSED / 'mrp.csv'}")

    report = OUTPUT / "05_expected_returns.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport: {report}")


if __name__ == "__main__":
    main()
