"""Module 1 - load the raw data and check it before anything else uses it.

The rest of the project assumes three things that are established here:

1. the investable universe is exactly the 504 firms in the hand-in sheet, in
   the order of that sheet (the hand-in has to come back in that order);
2. every return is a float, with CRSP's missing-value codes turned into NaN
   rather than into a -66% return;
3. there is at most one row per (PERMNO, date) in each panel.

Nothing here fills in, interpolates or invents data. Where something is
missing it is reported and left missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CRSP_MISSING_RETURN_CODES,
    HAND_IN_SHEET,
    RAW,
    SHEET_FIRST_FIRM_ROW,
    SHEET_LAST_FIRM_ROW,
    TRADE_DATE,
)

# WRDS gives every download a random file name, so we identify the two panels
# by a column only that panel has. Re-downloading the data then needs no code
# change.
MONTHLY_KEY_COLUMN = "MthRet"
DAILY_KEY_COLUMN = "DlyRet"


def find_raw_file(key_column: str, folder: Path = RAW) -> Path:
    """Return the single CSV in `folder` whose header contains `key_column`."""
    matches = []
    for path in sorted(folder.glob("*.csv")):
        if key_column in pd.read_csv(path, nrows=0).columns:
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one CSV in {folder} with a '{key_column}' "
            f"column, found {[m.name for m in matches]}"
        )
    return matches[0]


# ---------------------------------------------------------------------------
# returns
# ---------------------------------------------------------------------------


def coerce_returns(raw: pd.Series) -> tuple[pd.Series, dict]:
    """Turn a raw CRSP return column into floats.

    Two separate problems are handled:
      - WRDS exports can contain letter codes (e.g. 'B', 'C') for a missing
        return, which make pandas read the whole column as text;
      - CRSP also uses the numeric codes -66/-77/-88/-99 for the same thing,
        and those look like perfectly valid (disastrous) returns.

    Both become NaN. The counts are returned so that the report can show how
    much was affected instead of hiding it.
    """
    numeric = pd.to_numeric(raw, errors="coerce")
    n_letter_codes = int((numeric.isna() & raw.notna()).sum())

    is_code = numeric.isin(CRSP_MISSING_RETURN_CODES)
    n_missing_codes = int(is_code.sum())
    numeric = numeric.mask(is_code)

    info = {
        "n_letter_codes": n_letter_codes,
        "n_missing_codes": n_missing_codes,
        "n_missing_total": int(numeric.isna().sum()),
        "n_rows": int(len(numeric)),
    }
    return numeric, info


# ---------------------------------------------------------------------------
# the universe
# ---------------------------------------------------------------------------


def load_universe(path: Path = HAND_IN_SHEET) -> pd.DataFrame:
    """Read the firm block of the hand-in sheet.

    Rows 22-526 hold 505 lines but only 504 firms: HST (PERMNO 46703) is
    listed twice. We keep every sheet row - shares have to be written back
    into all of them - but mark the repeat so the firm is not counted, or
    optimised over, twice.
    """
    n_rows = SHEET_LAST_FIRM_ROW - SHEET_FIRST_FIRM_ROW + 1
    df = pd.read_excel(
        path,
        header=None,
        skiprows=SHEET_FIRST_FIRM_ROW - 1,
        nrows=n_rows,
        usecols="A:F",
        names=["permno", "ticker", "permco", "cusip", "company", "price"],
    )
    df["sheet_row"] = range(SHEET_FIRST_FIRM_ROW, SHEET_LAST_FIRM_ROW + 1)
    df["permno"] = df["permno"].astype(int)
    df["permco"] = df["permco"].astype(int)
    df["price"] = df["price"].astype(float)

    # The first occurrence of a PERMNO is the one we treat as "the firm".
    df["is_duplicate_row"] = df.duplicated(subset="permno", keep="first")
    return df


def dual_class_permcos(universe: pd.DataFrame) -> pd.DataFrame:
    """Firms (PERMCOs) that appear with more than one distinct PERMNO.

    GOOG/GOOGL, FOX/FOXA and NWS/NWSA are two share classes of one company.
    They are separate assets in the sheet, but their returns are almost
    perfectly correlated, which pushes the covariance matrix towards
    singularity. Step 6 keeps one class per company; here we only find them.
    """
    firms = universe.loc[~universe["is_duplicate_row"]]
    counts = firms.groupby("permco")["permno"].nunique()
    multi = counts[counts > 1].index
    return firms[firms["permco"].isin(multi)].sort_values(["permco", "ticker"])


# ---------------------------------------------------------------------------
# the two CRSP panels
# ---------------------------------------------------------------------------


def load_monthly(path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """CRSP monthly stock file, 2006-2025."""
    path = path or find_raw_file(MONTHLY_KEY_COLUMN)
    df = pd.read_csv(path, dtype={"MthRet": str, "SICCD": str})
    df = df.rename(
        columns={
            "PERMNO": "permno",
            "PERMCO": "permco",
            "Ticker": "ticker",
            "SICCD": "siccd",
            "MthCalDt": "date",
            "MthPrc": "price",
            "MthCap": "mktcap",
            "MthRet": "ret",
            "ShrOut": "shrout",
            "HdrCUSIP": "cusip",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df["ret"], info = coerce_returns(df["ret"])
    # CRSP reports a bid/ask average as a negative price when there was no
    # trade. The sign says where the number came from, not what it is worth.
    df["price"] = df["price"].abs()
    return df, info


def load_daily(path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """CRSP daily stock file, 2016-2025."""
    path = path or find_raw_file(DAILY_KEY_COLUMN)
    df = pd.read_csv(path, dtype={"DlyRet": str})
    df = df.rename(
        columns={
            "PERMNO": "permno",
            "PERMCO": "permco",
            "Ticker": "ticker",
            "DlyCalDt": "date",
            "DlyRet": "ret",
            "HdrCUSIP": "cusip",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df["ret"], info = coerce_returns(df["ret"])
    return df, info


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def duplicate_rows(df: pd.DataFrame, keys=("permno", "date")) -> pd.DataFrame:
    """All rows belonging to a (PERMNO, date) pair that occurs more than once."""
    keys = list(keys)
    return df.loc[df.duplicated(subset=keys, keep=False)].sort_values(keys)


def drop_exact_duplicates(df: pd.DataFrame,
                          keys=("permno", "date")) -> tuple[pd.DataFrame, dict]:
    """Drop repeated rows, but only after checking they carry no information.

    The WRDS extracts contain repeated (PERMNO, date) rows. Before removing
    them we verify that the repeats are identical in *every* column - if two
    rows shared a key but differed in, say, the return, silently keeping one
    would be a data choice, not a clean-up. If that ever happens the function
    refuses rather than guessing.
    """
    keys = list(keys)
    n_before = len(df)
    deduped = df.drop_duplicates()          # exact copies only

    still_dup = deduped.duplicated(subset=keys, keep=False)
    if still_dup.any():
        offenders = deduped.loc[still_dup, keys].drop_duplicates()
        raise ValueError(
            f"{len(offenders)} (PERMNO, date) pairs appear more than once with "
            f"different values; these are not exact duplicates and cannot be "
            f"dropped automatically. First few:\n{offenders.head()}"
        )

    info = {"n_before": n_before, "n_after": len(deduped),
            "n_dropped": n_before - len(deduped)}
    return deduped.reset_index(drop=True), info


def coverage(panel: pd.DataFrame, universe: pd.DataFrame) -> dict:
    """How well a panel covers the 504 firms we have to invest in."""
    firms = universe.loc[~universe["is_duplicate_row"], "permno"]
    in_panel = set(panel["permno"].unique())
    return {
        "n_firms_universe": int(firms.nunique()),
        "n_firms_in_panel": int(len(set(firms) & in_panel)),
        "missing_permnos": sorted(set(firms) - in_panel),
        "extra_permnos": sorted(in_panel - set(firms)),
    }


def observations_per_firm(panel: pd.DataFrame) -> pd.Series:
    """Number of non-missing returns per PERMNO."""
    return panel.dropna(subset=["ret"]).groupby("permno").size()


def internal_gaps(panel: pd.DataFrame) -> pd.DataFrame:
    """Firms whose observed dates skip periods *inside* their own sample.

    A short history is fine: a stock listed in 2021 simply starts late. A hole
    in the middle is not, because compounding straight across it silently
    attributes a multi-period return to a single period. Gaps are counted in
    trading periods of the panel itself, so no calendar assumptions are made.
    """
    all_dates = pd.Index(sorted(panel["date"].unique()))
    rank = pd.Series(np.arange(len(all_dates)), index=all_dates)

    rows = []
    for permno, grp in panel.groupby("permno"):
        pos = rank.loc[sorted(grp["date"].unique())].to_numpy()
        if len(pos) < 2:
            continue
        gaps = np.diff(pos) - 1
        if gaps.max() > 0:
            rows.append(
                {
                    "permno": permno,
                    "n_obs": len(pos),
                    "n_gaps": int((gaps > 0).sum()),
                    "largest_gap": int(gaps.max()),
                    "first_date": all_dates[pos[0]],
                    "last_date": all_dates[pos[-1]],
                }
            )
    cols = ["permno", "n_obs", "n_gaps", "largest_gap", "first_date", "last_date"]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("largest_gap", ascending=False) if len(out) else out


def price_cross_check(universe: pd.DataFrame, monthly: pd.DataFrame,
                      tol: float = 0.01) -> pd.DataFrame:
    """Compare the sheet's 31-12-2025 prices with CRSP's own December price.

    This is the check that ties our data to the prices we are forced to trade
    at. If the two disagree, the shares we buy do not cost what the sheet says
    they cost, and the budget will not add up to EUR 10m.
    """
    dec = monthly.loc[monthly["date"] == TRADE_DATE, ["permno", "price"]]
    dec = dec.rename(columns={"price": "crsp_price"})
    firms = universe.loc[~universe["is_duplicate_row"],
                         ["permno", "ticker", "price"]]
    merged = firms.merge(dec, on="permno", how="left")
    merged["abs_diff"] = (merged["price"] - merged["crsp_price"]).abs()
    bad = merged["crsp_price"].isna() | (merged["abs_diff"] > tol)
    return merged.loc[bad].sort_values("abs_diff", ascending=False)


def no_lookahead(panel: pd.DataFrame, trade_date: pd.Timestamp = TRADE_DATE) -> int:
    """Number of rows dated after the trading date. Must be zero."""
    return int((panel["date"] > trade_date).sum())
