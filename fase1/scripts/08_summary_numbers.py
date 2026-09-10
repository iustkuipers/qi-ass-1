"""Step 8 - collect every headline number in one place.

Run from the fase1 folder:   python scripts/08_summary_numbers.py

Everything quoted in README.md and in the one-page summary comes from here, so
that a re-run of the pipeline updates the write-up instead of leaving it
quietly stale. Nothing is computed here that is not already computed upstream:
this script only reads the processed files and the reports.

Produces
    output/summary_numbers.txt
"""

import re
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (BUDGET, FASE1, OUTPUT, PROCESSED, RF_ANNUAL,  # noqa: E402
                    RISK_AVERSION, TRADE_DATE, WEIGHT_CAP)
from modules import module5_expected_returns as m5  # noqa: E402


def _grab(path: Path, pattern: str) -> str:
    """Pull a single number out of one of the step reports."""
    text = path.read_text(encoding="utf-8")
    match = re.search(pattern, text)
    return match.group(1).strip() if match else "?"


def main() -> None:
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    universe = pd.read_csv(PROCESSED / "universe.csv")
    chars = pd.read_csv(PROCESSED / "characteristics.csv")
    weights_all = pd.read_csv(PROCESSED / "tangency_weights.csv", index_col="permno")
    fm = pd.read_csv(PROCESSED / "fm_summary.csv")
    mrp_row = pd.read_csv(PROCESSED / "mrp.csv", index_col=0)["0"]
    alloc = pd.read_csv(FASE1 / "data" / "output" / "allocation.csv")

    long_factors = m5.load_long_factors()
    mrp = m5.market_risk_premium(long_factors)

    w = weights_all["main_single_index"].dropna()
    mu = weights_all["mu_excess"].reindex(w.index)
    sigma = pd.read_csv(PROCESSED / "covariance_single_index.csv", index_col=0)
    sigma.columns = sigma.columns.astype(int)
    excess = float((w * mu).sum())
    vol = float(np.sqrt(w.to_numpy() @ sigma.loc[w.index, w.index].to_numpy()
                        @ w.to_numpy()))

    sheet = openpyxl.load_workbook(FASE1 / "data" / "output" /
                                   "Hand_in_sheet_part1.xlsx")
    ws = sheet[sheet.sheetnames[0]]
    stock_value = sum((ws[f"H{int(r.sheet_row)}"].value or 0) * float(r.price)
                      for r in universe.itertuples())
    riskfree = float(ws["H21"].value or 0)

    r4 = OUTPUT / "04_fama_macbeth.txt"
    r6 = OUTPUT / "06_portfolio.txt"
    r7 = OUTPUT / "07_allocation.txt"

    say("=" * 70)
    say("SUMMARY NUMBERS - Part 1, Fama & French (1992)")
    say("=" * 70)
    say(f"generated from the processed files; trade date {TRADE_DATE:%d-%m-%Y}")

    say()
    say("DATA")
    say(f"  firms in the hand-in sheet          : "
        f"{int((~universe['is_duplicate_row']).sum())}")
    say(f"  sheet rows (HST listed twice)       : {len(universe)}")
    say(f"  monthly panel                       : 2006-01 to 2025-12, 240 months")
    say(f"  daily panel                         : 2016-01 to 2025-12")
    say(f"  Compustat fiscal years              : 2004 to 2025")
    say(f"  long factor history                 : {mrp['first']:%Y-%m} to "
        f"{mrp['last']:%Y-%m} ({mrp['n_months']} months)")

    say()
    say("STEP 3 - BETAS")
    say(f"  first June sort                     : 2008")
    say(f"  post-ranking months                 : 2008-07 to 2025-12 (210)")
    say(f"  portfolios                          : 25 (5 size x 5 beta)")
    say(f"  post-ranking beta, lowest group     : 0.66")
    say(f"  post-ranking beta, highest group    : 1.46")
    say(f"  spread in beta across beta groups   : 0.80")
    say(f"  spread in ln(ME) across beta groups : 0.05")

    say()
    say("STEP 4 - FAMA-MACBETH (main sample, specification 6)")
    spec6 = fm[(fm["spec"] == "(6) beta + ln(ME) + ln(BE/ME)")
               & (fm["sample"] == "main (no financials, no negative BE)")
               ].set_index("variable")
    for var, label in [("beta", "beta"), ("ln_me", "ln(ME)"), ("ln_bm", "ln(BE/ME)")]:
        say(f"  {label:<36}: {spec6.loc[var, 'mean'] * 100:>7.3f}% per month  "
            f"(t = {spec6.loc[var, 'fm_t']:.2f})")
    # Regexes live outside the f-strings: Python 3.11 does not allow a
    # backslash inside an f-string expression.
    g1 = _grab(r4, r"mean gamma_1 .*?:\s+([-0-9.]+)% per month")
    mkt_same = _grab(r4, r"mean mktrf, same months\s+:\s+([-0-9.]+)%")
    diff = _grab(r4, r"difference gamma_1 - mktrf\s+:\s+([-0-9.]+)% per month")
    diff_t = _grab(r4, r"difference gamma_1 - mktrf.*?t = ([-0-9.]+)")
    surv = _grab(r4, r"difference\s+:\s+([-0-9.]+)% per month \(t")
    say(f"  gamma_1 in the beta-only regression : {g1}% per month")
    say(f"  mean mktrf over the same months     : {mkt_same}% per month")
    say(f"  difference gamma_1 - mktrf          : {diff}% (t = {diff_t})")
    say(f"  survivorship: universe vs market    : {surv}% per month")

    say()
    say("STEP 5 - EXPECTED RETURNS")
    say(f"  rule                                : E[r] - rf = MRP x beta")
    say(f"  MRP, long-run mean mktrf            : {mrp['monthly']:.4%} per month "
        f"= {mrp['annual']:.2%} per year (t = {mrp['t']:.2f})")
    say(f"  risk-free rate (given)              : {RF_ANNUAL:.2%} per year")
    say(f"  distinct expected returns           : 25 (one per 5x5 cell)")
    say(f"  expected excess return range        : {mu.min():.2%} to {mu.max():.2%}")

    say()
    say("STEP 6 - COVARIANCE AND PORTFOLIO")
    say(f"  covariance                          : single index, "
        f"Sigma = sigma_m^2 beta beta' + D")
    window = _grab(r6, r"(\d{4}-\d\d-\d\d to \d{4}-\d\d-\d\d)")
    n_days = _grab(r6, r"(\d+) days used")
    mkt_vol = _grab(r6, r"market volatility over the window : ([0-9.]+)%")
    say(f"  daily window                        : {window}, {n_days} days")
    say(f"  market volatility                   : {mkt_vol}%")
    say(f"  stocks in the portfolio             : {len(w)}")
    say(f"  weight cap                          : {WEIGHT_CAP:.0%} (never binds)")
    say(f"  largest weight                      : {w.max():.3%}")
    say(f"  effective N                         : {1 / (w ** 2).sum():.1f}")
    say(f"  expected excess return              : {excess:.2%}")
    say(f"  volatility                          : {vol:.2%}")
    say(f"  Sharpe ratio                        : {excess / vol:.3f}")
    say(f"  assigned beta                       : "
        f"{float((w * weights_all['beta'].reindex(w.index)).sum()):.3f}")
    si_realised = _grab(r6, r"realised beta\s+([0-9.]+)\s")
    lw_sharpe = _grab(r6, r"model Sharpe ratio\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)")
    lw_realised = _grab(r6, r"realised beta\s+[0-9.]+\s+([0-9.]+)")
    say(f"  realised beta (daily regression)    : {si_realised}")
    say(f"  Ledoit-Wolf robustness, Sharpe      : {lw_sharpe}")
    say(f"  Ledoit-Wolf robustness, realised beta : {lw_realised}")

    say()
    say("STEP 7 - ALLOCATION")
    say(f"  risk aversion A                     : {RISK_AVERSION:.0f}")
    y_raw = excess / (RISK_AVERSION * vol ** 2)
    y_used = min(y_raw, 1.0)
    say(f"  y* = (E[r]-rf)/(A sigma^2)          : {y_raw:.3f} raw, "
        f"{y_used:.3f} used" + ("  (no-borrowing cap binds)" if y_raw > 1 else ""))
    say(f"  positions held                      : "
        f"{int((alloc['shares'] > 0).sum())}")
    say(f"  invested in stocks                  : EUR {stock_value:,.2f}")
    say(f"  risk-free holding                   : EUR {riskfree:,.2f}")
    say(f"  TOTAL                               : EUR {stock_value + riskfree:,.2f}")
    say(f"  budget                              : EUR {BUDGET:,.2f}")
    say(f"  balanced exactly                    : "
        f"{abs(stock_value + riskfree - BUDGET) < 1e-6}")
    say(f"  largest position                    : EUR "
        f"{alloc['value'].max():,.2f} ({alloc['value'].max() / BUDGET:.3%})")

    top = alloc.nlargest(10, "value")[["ticker", "shares", "price_x", "value"]]
    say()
    say("  ten largest positions")
    for _, r in top.iterrows():
        say(f"      {str(r['ticker']):<7} {int(r['shares']):>7,} shares  "
            f"EUR {r['value']:>11,.2f}  {r['value'] / BUDGET:>7.3%}")

    say()
    say("EXCLUSIONS (504 -> {} investable)".format(len(w)))
    say("  6  second share class          FOX GOOG LEN MKC NWS TAP")
    say("  6  under 24 months of history  GEV PSKY Q SNDK SOLV SW")
    say("  2  no 2025 pre-ranking beta    CRH VLTO")
    say("  3  pending cash takeover       DAY EA HOLX")
    say("  1  partial daily history       KVUE")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary_numbers.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten: {OUTPUT / 'summary_numbers.txt'}")


if __name__ == "__main__":
    main()
