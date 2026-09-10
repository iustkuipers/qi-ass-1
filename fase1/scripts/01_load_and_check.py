"""Step 1 - load the raw data, check it, and write a clean copy.

Run from the fase1 folder:   python scripts/01_load_and_check.py

Produces
    data/processed/universe.csv     the 505 sheet rows, 504 firms, in sheet order
    data/processed/monthly.parquet  CRSP monthly panel, returns as floats
    data/processed/daily.parquet    CRSP daily panel, returns as floats
    output/01_data_check.txt        the report printed below
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import N_FIRMS_EXPECTED, OUTPUT, PROCESSED, TRADE_DATE  # noqa: E402
from modules import module1_data as m1  # noqa: E402


def main() -> None:
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say("=" * 72)
    say("STEP 1 - DATA LOADING AND CHECKS")
    say("=" * 72)

    # --- universe --------------------------------------------------------
    universe = m1.load_universe()
    firms = universe.loc[~universe["is_duplicate_row"]]

    say()
    say("UNIVERSE (from Hand_in_sheet.xlsx)")
    say(f"  sheet rows read          : {len(universe)}")
    say(f"  unique firms (PERMNO)    : {firms['permno'].nunique()}")
    say(f"  expected                 : {N_FIRMS_EXPECTED}")
    repeats = universe.loc[universe["is_duplicate_row"]]
    for _, r in repeats.iterrows():
        first_row = universe.loc[universe["permno"] == r["permno"], "sheet_row"].iloc[0]
        say(f"  repeated line            : {r['ticker']} (PERMNO {r['permno']}) "
            f"on sheet rows {first_row} and {r['sheet_row']}")

    dual = m1.dual_class_permcos(universe)
    say(f"  dual share classes       : {dual['permco'].nunique()} companies, "
        f"{len(dual)} tickers")
    # Three of these carry the same ticker in the sheet and are only told
    # apart by CUSIP, so the CUSIP is printed too.
    for permco, grp in dual.groupby("permco"):
        say(f"      PERMCO {permco:>5} {grp['company'].iloc[0][:28]:28s} " + ", ".join(
            f"{t} ({p}, cusip {c})"
            for t, p, c in zip(grp["ticker"], grp["permno"], grp["cusip"])))

    # --- monthly panel ---------------------------------------------------
    monthly, minfo = m1.load_monthly()
    say()
    say("CRSP MONTHLY PANEL")
    say(f"  file                     : {m1.find_raw_file(m1.MONTHLY_KEY_COLUMN).name}")
    say(f"  rows                     : {len(monthly):,}")
    say(f"  columns                  : {', '.join(monthly.columns)}")
    say(f"  date range               : {monthly['date'].min():%Y-%m-%d} to "
        f"{monthly['date'].max():%Y-%m-%d}")
    say(f"  distinct PERMNOs         : {monthly['permno'].nunique()}")
    say(f"  returns as text codes    : {minfo['n_letter_codes']}")
    say(f"  returns as -66/-77/-88/-99: {minfo['n_missing_codes']}")
    say(f"  missing returns in total : {minfo['n_missing_total']} "
        f"({minfo['n_missing_total'] / minfo['n_rows']:.2%})")
    say(f"  rows after {TRADE_DATE:%Y-%m-%d}   : {m1.no_lookahead(monthly)}")

    dup_m = m1.duplicate_rows(monthly)
    say(f"  duplicate (PERMNO, date) : {len(dup_m)} rows, on "
        f"{dup_m['permno'].nunique()} PERMNOs")
    monthly, dinfo_m = m1.drop_exact_duplicates(monthly)
    say(f"  -> exact copies, dropped : {dinfo_m['n_dropped']} "
        f"({dinfo_m['n_before']:,} -> {dinfo_m['n_after']:,} rows)")

    cov_m = m1.coverage(monthly, universe)
    say(f"  universe firms present   : {cov_m['n_firms_in_panel']} of "
        f"{cov_m['n_firms_universe']}")
    if cov_m["missing_permnos"]:
        say(f"  MISSING from panel       : {cov_m['missing_permnos']}")
    if cov_m["extra_permnos"]:
        say(f"  in panel but not in sheet: {len(cov_m['extra_permnos'])} PERMNOs")

    obs_m = m1.observations_per_firm(monthly)
    say(f"  monthly obs per firm     : min {obs_m.min()}, median "
        f"{int(obs_m.median())}, max {obs_m.max()}")
    say(f"  firms with < 24 months   : {int((obs_m < 24).sum())}")
    say(f"  firms with < 60 months   : {int((obs_m < 60).sum())}")

    gaps_m = m1.internal_gaps(monthly)
    say(f"  firms with internal gaps : {len(gaps_m)}")
    if len(gaps_m):
        for _, r in gaps_m.head(5).iterrows():
            say(f"      PERMNO {r['permno']}: {r['n_gaps']} gap(s), largest "
                f"{r['largest_gap']} month(s)")

    # --- daily panel -----------------------------------------------------
    daily, dinfo = m1.load_daily()
    say()
    say("CRSP DAILY PANEL")
    say(f"  file                     : {m1.find_raw_file(m1.DAILY_KEY_COLUMN).name}")
    say(f"  rows                     : {len(daily):,}")
    say(f"  columns                  : {', '.join(daily.columns)}")
    say(f"  date range               : {daily['date'].min():%Y-%m-%d} to "
        f"{daily['date'].max():%Y-%m-%d}")
    say(f"  distinct PERMNOs         : {daily['permno'].nunique()}")
    say(f"  returns as text codes    : {dinfo['n_letter_codes']}")
    say(f"  returns as -66/-77/-88/-99: {dinfo['n_missing_codes']}")
    say(f"  missing returns in total : {dinfo['n_missing_total']} "
        f"({dinfo['n_missing_total'] / dinfo['n_rows']:.2%})")
    say(f"  rows after {TRADE_DATE:%Y-%m-%d}   : {m1.no_lookahead(daily)}")

    dup_d = m1.duplicate_rows(daily)
    say(f"  duplicate (PERMNO, date) : {len(dup_d)} rows, on "
        f"{dup_d['permno'].nunique()} PERMNOs")
    daily, dinfo_d = m1.drop_exact_duplicates(daily)
    say(f"  -> exact copies, dropped : {dinfo_d['n_dropped']} "
        f"({dinfo_d['n_before']:,} -> {dinfo_d['n_after']:,} rows)")

    cov_d = m1.coverage(daily, universe)
    say(f"  universe firms present   : {cov_d['n_firms_in_panel']} of "
        f"{cov_d['n_firms_universe']}")
    if cov_d["missing_permnos"]:
        say(f"  MISSING from panel       : {cov_d['missing_permnos']}")

    obs_d = m1.observations_per_firm(daily)
    say(f"  daily obs per firm       : min {obs_d.min()}, median "
        f"{int(obs_d.median())}, max {obs_d.max()}")

    gaps_d = m1.internal_gaps(daily)
    say(f"  firms with internal gaps : {len(gaps_d)}")
    if len(gaps_d):
        for _, r in gaps_d.head(5).iterrows():
            say(f"      PERMNO {r['permno']}: {r['n_gaps']} gap(s), largest "
                f"{r['largest_gap']} trading day(s)")

    # --- short histories -------------------------------------------------
    # FF form pre-ranking betas on 24 to 60 months of past returns. A firm
    # with a shorter history cannot get one, so it matters now who they are.
    say()
    say("SHORT HISTORIES (matters for the pre-ranking betas in step 3)")
    names = universe.drop_duplicates("permno").set_index("permno")[["ticker", "company"]]
    short = pd.concat(
        [obs_m.rename("n_months"), monthly.groupby("permno")["date"].min().rename("first")],
        axis=1,
    ).join(names).sort_values("n_months")
    for _, r in short.loc[short["n_months"] < 60].iterrows():
        flag = "  <- under 24 months" if r["n_months"] < 24 else ""
        say(f"      {r['ticker']:6s} {r['n_months']:>4} months, from "
            f"{r['first']:%Y-%m}  {r['company'][:34]}{flag}")

    # --- prices we must trade at ----------------------------------------
    say()
    say("TRADING PRICES (sheet vs CRSP, 31-12-2025)")
    mism = m1.price_cross_check(universe, monthly)
    say(f"  firms where the sheet price differs by > 0.01 : {len(mism)}")
    if len(mism):
        for _, r in mism.head(10).iterrows():
            say(f"      {r['ticker']:6s} sheet {r['price']:>10.2f}  "
                f"crsp {r['crsp_price']!s:>10}  diff {r['abs_diff']!s}")

    # --- write clean copies ---------------------------------------------
    universe.to_csv(PROCESSED / "universe.csv", index=False)
    monthly.to_parquet(PROCESSED / "monthly.parquet", index=False)
    daily.to_parquet(PROCESSED / "daily.parquet", index=False)

    say()
    say("WRITTEN")
    say(f"  {PROCESSED / 'universe.csv'}")
    say(f"  {PROCESSED / 'monthly.parquet'}")
    say(f"  {PROCESSED / 'daily.parquet'}")

    report = OUTPUT / "01_data_check.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport: {report}")


if __name__ == "__main__":
    main()
