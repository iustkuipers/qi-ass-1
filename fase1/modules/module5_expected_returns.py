"""Module 5 - expected returns from the Fama-MacBeth evidence.

The group's rule after step 4 was: use only characteristics that are
statistically significant *and* not an artefact of how our universe was
selected. That leaves beta and nothing else.

  - size is the strongest slope in the data, and also the one most damaged by
    survivorship: a firm that was small in 2008 and is still in the S&P 500 in
    2025 was selected for having survived;
  - B/M has t = -1.95, fails to reach significance in either subperiod, and
    carries the wrong sign relative to the paper;
  - the D(BE<0) dummy only exists to keep negative-BE firms in a regression
    that uses B/M, so it goes with it;
  - the intercept is insignificant in the beta-only specification (t = 0.83),
    and the CAPM says it should be zero, so it is set to zero.

What remains is the CAPM itself:

    E[r_i] - rf = MRP * beta_i

with beta the post-ranking beta of the 5x5 portfolio the stock sits in for FF
year 2025, and MRP the *long-run* market risk premium rather than the one
realised over our own sample. The realised 2008-2025 premium of 0.97% per
month is a bull market plus survivorship; using it would build the very bias
we are trying to avoid into the expected returns.

One consequence worth stating plainly: because the tangency portfolio depends
on mu only through the direction of (mu - rf), and here that direction is
MRP * beta, **the value chosen for MRP does not change the tangency weights at
all**. It scales the expected return and the Sharpe ratio of the portfolio,
and it matters for how much is held at the risk-free rate, but not for which
stocks are held or in what proportion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import MONTHS_PER_YEAR, RAW, RF_ANNUAL, TRADE_DATE

def _find_long_factor_file() -> Path:
    """The long factor history, under either spelling of the file name."""
    for name in ("famafrench_long.csv", "famafrenchlong.csv"):
        candidate = RAW / name
        if candidate.exists():
            return candidate
    return RAW / "famafrench_long.csv"       # the name the report will ask for


LONG_FACTOR_FILE = _find_long_factor_file()


def load_long_factors(path: Path = LONG_FACTOR_FILE,
                      trade_date: pd.Timestamp = TRADE_DATE) -> pd.DataFrame:
    """The long Fama-French factor history, truncated at the trading date.

    Accepts either a WRDS-style file (`dateff`, decimals) or a Ken French
    library file (`Date` as YYYYMM, percent), because the two are easy to mix
    up and the difference is a factor of 100.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    date_col = next((c for c in ["dateff", "date", "month"] if c in df.columns), None)
    if date_col is None:
        raise ValueError(f"no date column found in {path.name}: {list(df.columns)}")

    raw_dates = df[date_col].astype(str).str.strip()
    if raw_dates.str.fullmatch(r"\d{6}").all():
        df["date"] = pd.to_datetime(raw_dates, format="%Y%m") + pd.offsets.MonthEnd(0)
    else:
        df["date"] = pd.to_datetime(raw_dates)

    mkt_col = next((c for c in ["mktrf", "mkt-rf", "mkt_rf"] if c in df.columns), None)
    if mkt_col is None:
        raise ValueError(f"no market excess return column in {path.name}")
    df = df.rename(columns={mkt_col: "mktrf"})

    df = df[df["date"] <= trade_date].copy()
    df["mktrf"] = pd.to_numeric(df["mktrf"], errors="coerce")
    df = df.dropna(subset=["mktrf"])

    # Ken French publishes percent, WRDS publishes decimals. A monthly market
    # excess return with a standard deviation above 0.5 can only be percent.
    if df["mktrf"].std() > 0.5:
        df["mktrf"] = df["mktrf"] / 100.0
        df.attrs["converted_from_percent"] = True
    else:
        df.attrs["converted_from_percent"] = False

    return df[["date", "mktrf"]].sort_values("date").reset_index(drop=True)


def market_risk_premium(factors: pd.DataFrame) -> dict:
    """Average monthly market excess return, with its standard error."""
    x = factors["mktrf"]
    monthly = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    return {
        "monthly": monthly,
        "annual": monthly * MONTHS_PER_YEAR,
        "se_monthly": se,
        "t": monthly / se,
        "n_months": len(x),
        "first": factors["date"].min(),
        "last": factors["date"].max(),
        "annual_sd": x.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR),
    }


# ---------------------------------------------------------------------------
# the expected returns themselves
# ---------------------------------------------------------------------------


def mu_capm(betas: pd.Series, mrp_monthly: float,
            rf_annual: float = RF_ANNUAL) -> pd.DataFrame:
    """E[r] - rf = MRP * beta, annualised.

    Returns both the excess and the total expected return. The excess return
    is what the optimiser uses; the total is only for reporting.
    """
    excess = betas * mrp_monthly * MONTHS_PER_YEAR
    return pd.DataFrame({
        "beta": betas,
        "mu_excess": excess,
        "mu_total": excess + rf_annual,
    })


def mu_from_slopes(characteristics: pd.DataFrame, slopes: dict[str, float],
                   anchor_excess: float | None = None) -> pd.Series:
    """Fitted expected excess return from a Fama-MacBeth specification.

    `slopes` maps a column of `characteristics` to its average monthly slope.
    The intercept is deliberately not taken from the regression: it carries
    the survivorship level. Instead, when `anchor_excess` is given, a constant
    is added so that the cross-sectional average expected excess return equals
    that value. All the relative tilts are kept; only the level is replaced.
    """
    contrib = pd.Series(0.0, index=characteristics.index)
    for col, slope in slopes.items():
        contrib = contrib + slope * characteristics[col].astype(float)

    annual = contrib * MONTHS_PER_YEAR
    if anchor_excess is not None:
        annual = annual - annual.mean() + anchor_excess
    return annual


def describe_tilt(weights: pd.Series, characteristics: pd.DataFrame) -> dict:
    """Weighted average characteristics of a portfolio, for comparing variants.

    A portfolio's tilt is easier to judge from the characteristics it holds
    than from 490 individual weights.
    """
    w = weights[weights.abs() > 0]
    chars = characteristics.reindex(w.index)
    out = {"n_positions": int((w > 1e-8).sum()), "gross": float(w.abs().sum())}
    for col in ["beta", "ln_me", "ln_bm"]:
        if col in chars.columns:
            valid = chars[col].notna()
            if valid.any():
                ww = w[valid] / w[valid].sum()
                out[f"w_{col}"] = float((ww * chars.loc[valid, col]).sum())
                out[f"ew_{col}"] = float(chars.loc[valid, col].mean())
    return out
