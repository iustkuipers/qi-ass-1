"""Module 2 - firm characteristics: book equity, market equity and B/M.

This follows the construction in Fama & French (1992, JF), with the book
equity definition of Davis, Fama & French (1997) that FF themselves later
adopted.

The timing is the part that is easy to get wrong, so it is stated once here
and enforced in `build_characteristics`:

    for the twelve months from July of year t to June of year t+1
        size    = ME at the end of June of year t
        B/M     = BE of the fiscal year ending in calendar year t-1
                  divided by ME at the end of December of year t-1

The six-month gap between the fiscal year end and the sort is what keeps the
exercise honest: accounting numbers for a fiscal year ending in, say, October
of t-1 are public well before July of t. Nothing here uses information that
was not available at the time it is used.

Units: Compustat reports in millions of dollars, CRSP market cap in thousands.
Everything below is converted to millions before it meets anything else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import RAW, TRADE_DATE

COMPUSTAT_FILE = RAW / "compustat.csv"

# Only the columns we actually use. The WRDS Funda export has ~950 of them,
# and reading all of them is both slow and an invitation to use something by
# accident.
COMPUSTAT_COLUMNS = [
    "GVKEY", "LINKPRIM", "LINKTYPE", "LPERMNO", "LPERMCO",
    "datadate", "fyear", "tic", "conm", "curcd", "curncd",
    "seq", "ceq", "at", "lt",          # stockholders' equity and fallbacks
    "pstkrv", "pstkl", "pstk",         # preferred stock, in FF's order
    "txditc", "txdb", "itcb",          # deferred taxes and investment tax credit
    "ib", "sich",                      # earnings, historical SIC
]

# Fama & French (1992) drop financial firms: their leverage means something
# different from an industrial firm's, so B/M is not comparable.
FINANCIAL_SIC_RANGE = (6000, 6999)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_compustat(path: Path = COMPUSTAT_FILE,
                   trade_date: pd.Timestamp = TRADE_DATE) -> tuple[pd.DataFrame, dict]:
    """Read the CRSP/Compustat Merged annual file and make it one row per
    (PERMNO, fiscal year).

    Three filters, each for a reason:
      - `LINKPRIM` in {P, C}: the primary link only. A company can be linked
        to several securities; without this filter one firm's accounting data
        would be attached to two or three PERMNOs at once.
      - `datadate <= trade_date`: the file as delivered runs into 2026. Any
        row dated after 31-12-2025 is information we are not allowed to have.
      - currency must be USD, since we never convert.
    """
    df = pd.read_csv(path, usecols=COMPUSTAT_COLUMNS, dtype={"GVKEY": str})
    n_raw = len(df)

    df["datadate"] = pd.to_datetime(df["datadate"])

    n_bad_currency = int((df["curcd"] != "USD").sum() + (df["curncd"] != "USD").sum())

    df = df[df["LINKPRIM"].isin(["P", "C"])]
    n_after_link = len(df)

    n_future = int((df["datadate"] > trade_date).sum())
    df = df[df["datadate"] <= trade_date]

    df = df.dropna(subset=["LPERMNO", "fyear"])
    df["permno"] = df["LPERMNO"].astype(int)
    df["fyear"] = df["fyear"].astype(int)

    # After the primary-link filter there should be exactly one row per firm
    # per fiscal year. We check rather than assume.
    dup = df.duplicated(subset=["permno", "fyear"], keep=False)
    if dup.any():
        raise ValueError(
            f"{int(dup.sum())} rows share a (PERMNO, fiscal year) after the "
            f"LINKPRIM filter; the accounting data is ambiguous for those firms."
        )

    info = {
        "n_raw": n_raw,
        "n_after_linkprim": n_after_link,
        "n_dropped_future": n_future,
        "n_final": len(df),
        "n_bad_currency": n_bad_currency,
        "n_permnos": int(df["permno"].nunique()),
        "datadate_max": df["datadate"].max(),
    }
    return df.reset_index(drop=True), info


# ---------------------------------------------------------------------------
# book equity
# ---------------------------------------------------------------------------


def _first_available(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Value of the first column in `columns` that is not missing, per row."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for col in columns:
        out = out.fillna(df[col])
    return out


def book_equity(df: pd.DataFrame) -> pd.DataFrame:
    """Book equity following Davis, Fama & French (1997).

        BE = stockholders' equity + deferred taxes - preferred stock

    Each of the three pieces has a fallback ladder, because Compustat does not
    report the same item for every firm:

      stockholders' equity : SEQ, else CEQ + PSTK, else AT - LT
      deferred taxes       : TXDITC, else TXDB + ITCB, else 0
      preferred stock      : PSTKRV, else PSTKL, else PSTK, else 0

    Preferred stock is subtracted because BE is meant to be the book value of
    *common* equity. The redemption value (PSTKRV) is FF's first choice, then
    liquidating value, then carrying value.

    Deferred taxes are added back: under the tax rules of the sample they are
    a source of funds belonging to the common shareholder, not a liability to
    an outside claimant. FF (1992) use the balance-sheet number, which is what
    TXDB + ITCB reconstructs when the combined item TXDITC is missing.
    """
    out = df.copy()

    out["se"] = _first_available(out, ["seq"])
    se_alt1 = out["ceq"] + out["pstk"].fillna(0.0)
    out["se"] = out["se"].fillna(se_alt1.where(out["ceq"].notna()))
    out["se"] = out["se"].fillna(out["at"] - out["lt"])

    # TXDB + ITCB only counts as available if at least one of the two is.
    txdb_itcb = out["txdb"].fillna(0.0) + out["itcb"].fillna(0.0)
    has_txdb_itcb = out["txdb"].notna() | out["itcb"].notna()
    out["deferred_tax"] = out["txditc"].fillna(txdb_itcb.where(has_txdb_itcb))
    out["deferred_tax"] = out["deferred_tax"].fillna(0.0)

    out["preferred"] = _first_available(out, ["pstkrv", "pstkl", "pstk"]).fillna(0.0)

    out["be"] = out["se"] + out["deferred_tax"] - out["preferred"]
    out.loc[out["se"].isna(), "be"] = np.nan     # no equity number at all -> no BE

    out["neg_be"] = out["be"] <= 0
    out["is_financial"] = out["sich"].between(*FINANCIAL_SIC_RANGE)
    return out


# ---------------------------------------------------------------------------
# market equity
# ---------------------------------------------------------------------------


def market_equity(monthly: pd.DataFrame, month: int) -> pd.DataFrame:
    """Market equity, in millions, at the end of every `month` in the panel.

    Two versions come out of this:

      me        the market equity of the security itself. This is the size
                variable: it is a property of the stock we can buy.
      me_firm   the market equity of the whole company, summing every share
                class with the same PERMCO. This is the denominator of B/M,
                because the numerator (book equity) is a company-level number.
                Dividing company book equity by one class's market equity
                would overstate B/M for Alphabet, Fox, News Corp and the rest.
    """
    sub = monthly[monthly["date"].dt.month == month].copy()
    sub["me"] = sub["mktcap"] / 1000.0            # CRSP thousands -> millions
    sub["year"] = sub["date"].dt.year

    # `min_count=1` matters: a plain sum of an all-missing group returns 0.0,
    # not NaN, and a market equity of zero turns B/M into infinity. Super
    # Micro (PERMNO 91907) has no December price in 2018 and 2019 while it
    # was off Nasdaq, which is exactly this case.
    firm_me = (sub.groupby(["permco", "year"])["me"].sum(min_count=1)
                  .rename("me_firm").reset_index())
    sub = sub.merge(firm_me, on=["permco", "year"], how="left")
    return sub[["permno", "permco", "year", "date", "me", "me_firm"]]


# ---------------------------------------------------------------------------
# putting it together with the FF timing
# ---------------------------------------------------------------------------


def build_characteristics(compustat: pd.DataFrame,
                          monthly: pd.DataFrame) -> pd.DataFrame:
    """One row per (PERMNO, FF year t), where year t covers July t - June t+1."""
    acc = book_equity(compustat)

    # A fiscal year is assigned to the calendar year in which it ends. FF sort
    # in June of t on the fiscal year that ended in t-1, so this calendar year
    # is what the merge keys on.
    acc["fy_end_year"] = acc["datadate"].dt.year

    # If a firm changed its fiscal year end it can have two fiscal years
    # ending in one calendar year. Keep the later one: it is the most recent
    # information available in June of the next year.
    acc = (acc.sort_values(["permno", "fy_end_year", "datadate"])
              .drop_duplicates(subset=["permno", "fy_end_year"], keep="last"))

    # FF require two years of Compustat history before a firm enters, to keep
    # backfilled data out of the sample.
    acc["years_on_compustat"] = acc.groupby("permno").cumcount()

    me_june = market_equity(monthly, month=6).rename(
        columns={"me": "me_june", "me_firm": "me_june_firm", "date": "june_date"})
    me_dec = market_equity(monthly, month=12).rename(
        columns={"me": "me_dec", "me_firm": "me_dec_firm", "date": "dec_date"})

    # The FF year t: size from June t, accounting from the fiscal year ending
    # in t-1, market equity in the B/M denominator from December of t-1.
    june = me_june.rename(columns={"year": "ff_year"})
    dec = me_dec.copy()
    dec["ff_year"] = dec["year"] + 1                     # December t-1 -> year t
    acc = acc.copy()
    acc["ff_year"] = acc["fy_end_year"] + 1              # fiscal year t-1 -> year t

    out = june.merge(
        dec[["permno", "ff_year", "me_dec", "me_dec_firm", "dec_date"]],
        on=["permno", "ff_year"], how="left")
    out = out.merge(
        acc[["permno", "ff_year", "datadate", "fyear", "be", "neg_be",
             "is_financial", "sich", "years_on_compustat", "conm"]],
        on=["permno", "ff_year"], how="left")

    # B/M uses company-level market equity, and only exists for positive BE.
    out["bm"] = out["be"] / out["me_dec_firm"].where(out["me_dec_firm"] > 0)
    out.loc[out["be"].isna() | (out["be"] <= 0), "bm"] = np.nan
    out["bm"] = out["bm"].replace([np.inf, -np.inf], np.nan)

    out["ln_me"] = np.log(out["me_june"].where(out["me_june"] > 0))
    out["ln_bm"] = np.log(out["bm"].where(out["bm"] > 0))

    out["has_bm"] = out["bm"].notna() & out["me_june"].notna()
    # A firm-year with no accounting row is neither financial nor negative-BE;
    # it simply has no B/M and drops out of the regressions that need one.
    # A firm-year with no accounting row is neither financial nor negative-BE;
    # it simply has no B/M and drops out of the regressions that need one.
    # `eq(True)` rather than `fillna(False)` because the merge leaves an
    # object column of {True, False, NaN}, which fillna would downcast.
    out["is_financial"] = out["is_financial"].eq(True)
    out["neg_be"] = out["neg_be"].eq(True)

    cols = ["permno", "permco", "ff_year", "june_date", "me_june", "me_june_firm",
            "dec_date", "me_dec", "me_dec_firm", "datadate", "fyear", "be", "bm",
            "ln_me", "ln_bm", "neg_be", "is_financial", "sich",
            "years_on_compustat", "has_bm", "conm"]
    return out[cols].sort_values(["ff_year", "permno"]).reset_index(drop=True)
