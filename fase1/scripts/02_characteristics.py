"""Step 2 - book equity, market equity and B/M with the Fama-French timing.

Run from the fase1 folder:   python scripts/02_characteristics.py

Produces
    data/processed/characteristics.csv   one row per (PERMNO, FF year)
    output/02_characteristics.txt        the report printed below
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import OUTPUT, PROCESSED  # noqa: E402
from modules import module2_characteristics as m2  # noqa: E402


def main() -> None:
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say("=" * 72)
    say("STEP 2 - BOOK EQUITY, MARKET EQUITY AND B/M")
    say("=" * 72)

    universe = pd.read_csv(PROCESSED / "universe.csv")
    monthly = pd.read_parquet(PROCESSED / "monthly.parquet")
    firms = universe.loc[~universe["is_duplicate_row"]]

    # --- compustat -------------------------------------------------------
    comp, cinfo = m2.load_compustat()
    say()
    say("COMPUSTAT (CRSP/Compustat Merged, Fundamentals Annual)")
    say(f"  rows as delivered        : {cinfo['n_raw']:,}")
    say(f"  after LINKPRIM in (P, C) : {cinfo['n_after_linkprim']:,}")
    say(f"  rows dated after 31-12-2025 (dropped) : {cinfo['n_dropped_future']}")
    say(f"  rows used                : {cinfo['n_final']:,}")
    say(f"  non-USD rows             : {cinfo['n_bad_currency']}")
    say(f"  latest datadate kept     : {cinfo['datadate_max']:%Y-%m-%d}")
    say(f"  one row per (PERMNO, fiscal year) : yes (checked, would raise otherwise)")

    have = set(comp["permno"])
    missing = firms.loc[~firms["permno"].isin(have)]
    say(f"  universe firms with accounting data : {len(set(firms['permno']) & have)}"
        f" of {len(firms)}")
    if len(missing):
        say("  NO Compustat row at all:")
        for _, r in missing.iterrows():
            say(f"      {r['ticker']:6s} {r['permno']}  {r['company']}")

    # --- book equity -----------------------------------------------------
    be = m2.book_equity(comp)
    say()
    say("BOOK EQUITY (Davis, Fama & French)")
    say(f"  rows with a BE value     : {int(be['be'].notna().sum()):,} of {len(be):,}")
    say(f"  stockholders' equity from SEQ        : {int(be['seq'].notna().sum()):,}")
    say(f"  ... from CEQ + PSTK (SEQ missing)    : "
        f"{int((be['seq'].isna() & be['ceq'].notna()).sum()):,}")
    say(f"  ... from AT - LT (SEQ and CEQ missing): "
        f"{int((be['seq'].isna() & be['ceq'].isna() & be['at'].notna()).sum()):,}")
    say(f"  deferred taxes from TXDITC           : {int(be['txditc'].notna().sum()):,}")
    say(f"  ... TXDITC missing, TXDB/ITCB used   : "
        f"{int((be['txditc'].isna() & (be['txdb'].notna() | be['itcb'].notna())).sum()):,}")
    say(f"  ... neither, treated as zero         : "
        f"{int((be['txditc'].isna() & be['txdb'].isna() & be['itcb'].isna()).sum()):,}")
    say(f"  preferred stock from PSTKRV          : {int(be['pstkrv'].notna().sum()):,}")
    say(f"  negative or zero BE                  : {int(be['neg_be'].sum())}")
    say(f"  financial firms (SICH 6000-6999)     : {int(be['is_financial'].sum()):,} "
        f"rows, {be.loc[be['is_financial'], 'permno'].nunique()} firms")

    # Negative book equity is real, not an error: heavy buybacks or big
    # write-offs can push it below zero. B/M is meaningless then, so FF drop
    # those firm-years. The ones with a fiscal year ending in 2024 are the
    # ones that will be missing a B/M in the portfolio we actually build.
    neg = be.loc[be["neg_be"], ["permno", "tic", "conm", "fyear", "datadate", "be"]]
    if len(neg):
        per_firm = neg.groupby("tic").size().sort_values(ascending=False)
        say(f"  firms ever showing negative BE       : {len(per_firm)}")
        say("  most persistent: " + ", ".join(
            f"{t} ({n}y)" for t, n in per_firm.head(8).items()))
        neg24 = neg[neg["datadate"].dt.year == 2024]
        say(f"  negative BE in the fiscal year that feeds the Dec-2025 "
            f"portfolio: {len(neg24)}")
        say("      " + ", ".join(sorted(neg24["tic"].astype(str))))

    # --- characteristics -------------------------------------------------
    ch = m2.build_characteristics(comp, monthly)
    ch = ch[ch["permno"].isin(set(firms["permno"]))]
    say()
    say("CHARACTERISTICS (one row per PERMNO per FF year, July t - June t+1)")
    say(f"  rows                     : {len(ch):,}")
    say(f"  FF years                 : {ch['ff_year'].min()} to {ch['ff_year'].max()}")

    say()
    say("  firms with a usable B/M, per FF year")
    say(f"      {'year':>6} {'in panel':>9} {'has ME':>7} {'has BE':>7} "
        f"{'has B/M':>8} {'neg BE':>7} {'financial':>10}")
    for year, g in ch.groupby("ff_year"):
        say(f"      {year:>6} {len(g):>9} {int(g['me_june'].notna().sum()):>7} "
            f"{int(g['be'].notna().sum()):>7} {int(g['has_bm'].sum()):>8} "
            f"{int(g['neg_be'].sum()):>7} {int(g['is_financial'].sum()):>10}")

    # --- examples we can check by hand ----------------------------------
    say()
    say("EXAMPLES (FF year 2025: size from June-2025, BE from FY2024, ME from Dec-2024)")
    ex = ch[ch["ff_year"] == 2025].merge(
        firms[["permno", "ticker"]], on="permno", how="left")
    show = ["AAPL", "XOM", "JPM", "F", "KO", "NVDA", "BRK.B", "T"]
    picked = ex[ex["ticker"].isin(show)]
    if picked.empty:
        picked = ex.nlargest(6, "me_june")
    say(f"      {'ticker':7} {'ME Jun25':>12} {'ME Dec24':>12} {'BE FY24':>12} "
        f"{'B/M':>7} {'ln(ME)':>8} {'fisc.':>8} {'fin':>4}")
    for _, r in picked.sort_values("me_june", ascending=False).iterrows():
        bm = f"{r['bm']:.3f}" if pd.notna(r["bm"]) else "  n/a"
        dd = f"{r['datadate']:%Y-%m}" if pd.notna(r["datadate"]) else "     n/a"
        say(f"      {str(r['ticker']):7} {r['me_june']:>12,.0f} "
            f"{r['me_dec_firm']:>12,.0f} {r['be']:>12,.0f} {bm:>7} "
            f"{r['ln_me']:>8.2f} {dd:>8} {str(bool(r['is_financial'])):>4}")

    say()
    say("  B/M distribution in FF year 2025 (positive BE only)")
    q = ex["bm"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    for k in ["count", "min", "5%", "25%", "50%", "75%", "95%", "max"]:
        say(f"      {k:>6}: {q[k]:>10.3f}")

    # --- write -----------------------------------------------------------
    ch.to_csv(PROCESSED / "characteristics.csv", index=False)
    say()
    say("WRITTEN")
    say(f"  {PROCESSED / 'characteristics.csv'}")

    report = OUTPUT / "02_characteristics.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport: {report}")


if __name__ == "__main__":
    main()
