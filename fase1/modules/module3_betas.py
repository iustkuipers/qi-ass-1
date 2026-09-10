"""Module 3 - pre-ranking betas, 5x5 size-beta portfolios, post-ranking betas.

This is the machinery behind Tables I and II of Fama & French (1992). The
problem it solves is that a beta estimated on a single stock is far too noisy
to sort on, and beta and size are strongly correlated in the cross-section, so
a plain beta sort is largely a size sort in disguise.

FF's answer, reproduced here:

  1. In June of year t, estimate a *pre-ranking* beta for every stock from the
     preceding 60 months of returns (at least 24 required).
  2. Sort stocks into size groups, and within each size group into
     pre-ranking beta groups. FF use 10x10 on NYSE breakpoints; our universe
     is 504 large caps, so NYSE breakpoints are meaningless here and we use
     5x5 groups on our own universe. This is our one deliberate deviation.
  3. Track the equal-weighted return of each of the 25 portfolios from July t
     to June t+1, and chain those years into one long series.
  4. Estimate a *post-ranking* beta on that full series, and give every stock
     in a portfolio the beta of its portfolio.

Step 3 makes beta and size vary independently: the beta sort inside a size
group spreads beta while holding size roughly fixed.

Both the pre- and post-ranking betas are "sum of slopes" betas: the stock is
regressed on the current *and* the previous month's market return and the two
slopes are added. FF do this to undo the downward bias in beta caused by
non-synchronous trading.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import RAW, TRADE_DATE

FF_FACTOR_FILE = RAW / "famafrench.csv"

MIN_MONTHS_PRE_RANKING = 24
MAX_MONTHS_PRE_RANKING = 60
N_SIZE_GROUPS = 5
N_BETA_GROUPS = 5


# ---------------------------------------------------------------------------
# factors
# ---------------------------------------------------------------------------


def load_ff_factors(path: Path = FF_FACTOR_FILE,
                    trade_date: pd.Timestamp = TRADE_DATE) -> tuple[pd.DataFrame, dict]:
    """Monthly Fama-French factors, truncated at the trading date.

    The file as delivered runs past 31-12-2025. Everything after the trading
    date is information we are not allowed to have, so it is cut here rather
    than being relied on not to matter.

    The factors are checked to be in decimals rather than percent: a monthly
    market excess return has a standard deviation near 0.04 in decimals and
    near 4 in percent, so the two are impossible to confuse.
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["dateff"])
    n_future = int((df["date"] > trade_date).sum())
    df = df[df["date"] <= trade_date].copy()

    sd = df["mktrf"].std()
    if sd > 0.5:
        raise ValueError(
            f"mktrf has a standard deviation of {sd:.2f}, which looks like "
            f"percent rather than decimals; divide by 100 before using it."
        )

    # Match on year-month: CRSP dates the month by its last calendar day,
    # the factor file by the last trading day.
    df["ym"] = df["date"].dt.to_period("M")
    df["mkt"] = df["mktrf"] + df["rf"]           # total market return
    info = {
        "n_dropped_future": n_future,
        "n_months": len(df),
        "first": df["date"].min(),
        "last": df["date"].max(),
        "mktrf_sd": sd,
        "in_decimals": True,
    }
    return df[["ym", "date", "mktrf", "rf", "smb", "hml", "umd", "mkt"]], info


# ---------------------------------------------------------------------------
# the sum-of-slopes beta
# ---------------------------------------------------------------------------


def sum_slope_beta(y: np.ndarray, x_now: np.ndarray, x_lag: np.ndarray) -> float:
    """Beta as the sum of the slopes on the current and lagged market return.

    Rows where anything is missing are dropped first. Returns NaN if fewer
    than `MIN_MONTHS_PRE_RANKING` rows survive or the regressors are
    degenerate.
    """
    ok = np.isfinite(y) & np.isfinite(x_now) & np.isfinite(x_lag)
    if ok.sum() < MIN_MONTHS_PRE_RANKING:
        return np.nan

    X = np.column_stack([np.ones(ok.sum()), x_now[ok], x_lag[ok]])
    try:
        coef, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    return float(coef[1] + coef[2])


# ---------------------------------------------------------------------------
# pre-ranking betas
# ---------------------------------------------------------------------------


def excess_return_panel(monthly: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Wide panel of monthly *excess* returns: rows are months, columns PERMNOs.

    FF (1992) regress raw returns on the raw market return. We use excess
    returns on the excess market return so that the betas are consistent with
    the excess-return regressions in step 4. With a monthly risk-free rate
    that is small and slow-moving the two give practically the same beta, and
    the script reports both so that claim can be checked rather than believed.
    """
    df = monthly[["permno", "date", "ret"]].copy()
    df["ym"] = df["date"].dt.to_period("M")
    df = df.merge(factors[["ym", "rf"]], on="ym", how="left")
    df["exret"] = df["ret"] - df["rf"]
    return df.pivot(index="ym", columns="permno", values="exret").sort_index()


def pre_ranking_betas(panel: pd.DataFrame, market: pd.Series,
                      formation_ym: pd.Period) -> pd.Series:
    """Pre-ranking beta for every stock, estimated at the end of `formation_ym`.

    The window is the `MAX_MONTHS_PRE_RANKING` months ending in that month,
    and a stock needs `MIN_MONTHS_PRE_RANKING` usable observations to get a
    beta at all - which is exactly why the six firms that listed in 2024-2025
    cannot receive one.
    """
    months = panel.index[panel.index <= formation_ym][-MAX_MONTHS_PRE_RANKING:]
    if len(months) < MIN_MONTHS_PRE_RANKING:
        return pd.Series(dtype=float)

    window = panel.loc[months]
    mkt_now = market.reindex(months).to_numpy()
    # The lag reaches one month back beyond the window, which is available
    # whenever the window does not start at the very first month of the panel.
    lag_index = [panel.index.get_loc(m) - 1 for m in months]
    mkt_lag = np.array([market.iloc[i] if i >= 0 else np.nan for i in lag_index])

    betas = {
        permno: sum_slope_beta(window[permno].to_numpy(), mkt_now, mkt_lag)
        for permno in window.columns
    }
    return pd.Series(betas, name="pre_beta").dropna()


# ---------------------------------------------------------------------------
# 5x5 portfolios
# ---------------------------------------------------------------------------


def _quantile_group(values: pd.Series, n_groups: int) -> pd.Series:
    """Rank into `n_groups` equal-count groups, numbered 1..n_groups."""
    return pd.qcut(values.rank(method="first"), n_groups,
                   labels=range(1, n_groups + 1)).astype(int)


def form_portfolios(size: pd.Series, pre_beta: pd.Series,
                    n_size: int = N_SIZE_GROUPS,
                    n_beta: int = N_BETA_GROUPS) -> pd.DataFrame:
    """Sort on size first, then on pre-ranking beta *within* each size group.

    Sorting on size first is what makes the exercise informative: it holds
    size roughly fixed inside a column, so the beta sort is not just picking
    up the size effect again.
    """
    both = pd.concat([size.rename("me"), pre_beta.rename("pre_beta")], axis=1).dropna()
    if len(both) < n_size * n_beta:
        return pd.DataFrame(columns=["permno", "size_group", "beta_group",
                                     "me", "pre_beta"])

    both["size_group"] = _quantile_group(both["me"], n_size)
    both["beta_group"] = (both.groupby("size_group", group_keys=False)["pre_beta"]
                              .apply(lambda s: _quantile_group(s, n_beta)))
    both = both.reset_index().rename(columns={"index": "permno"})
    return both[["permno", "size_group", "beta_group", "me", "pre_beta"]]


def build_assignments(panel: pd.DataFrame, market: pd.Series,
                      characteristics: pd.DataFrame,
                      eligible: set[int] | None = None) -> pd.DataFrame:
    """Run the June sort for every year for which it is possible.

    Returns one row per (FF year, PERMNO) with its 5x5 cell, its pre-ranking
    beta and its June market equity.
    """
    out = []
    for ff_year, chars in characteristics.groupby("ff_year"):
        formation = pd.Period(f"{ff_year}-06", freq="M")
        if formation not in panel.index:
            continue

        size = chars.set_index("permno")["me_june"].dropna()
        if eligible is not None:
            size = size[size.index.isin(eligible)]

        betas = pre_ranking_betas(panel, market, formation)
        betas = betas[betas.index.isin(size.index)]
        if betas.empty:
            continue

        assigned = form_portfolios(size.reindex(betas.index), betas)
        if assigned.empty:
            continue
        assigned["ff_year"] = ff_year
        out.append(assigned)

    if not out:
        return pd.DataFrame(columns=["ff_year", "permno", "size_group",
                                     "beta_group", "me", "pre_beta"])
    return pd.concat(out, ignore_index=True)[
        ["ff_year", "permno", "size_group", "beta_group", "me", "pre_beta"]]


# ---------------------------------------------------------------------------
# post-ranking portfolio returns and betas
# ---------------------------------------------------------------------------


def holding_months(ff_year: int, panel_index: pd.PeriodIndex) -> pd.PeriodIndex:
    """July of year t through June of year t+1, clipped to the data we have."""
    start = pd.Period(f"{ff_year}-07", freq="M")
    end = pd.Period(f"{ff_year + 1}-06", freq="M")
    return panel_index[(panel_index >= start) & (panel_index <= end)]


def portfolio_returns(assignments: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Equal-weighted monthly excess return of each of the 25 portfolios.

    Weights are equal among the stocks that actually traded that month, so a
    stock disappearing mid-year does not silently keep a weight.
    """
    rows = []
    for ff_year, group in assignments.groupby("ff_year"):
        months = holding_months(ff_year, panel.index)
        if len(months) == 0:
            continue
        window = panel.loc[months]
        for (sg, bg), cell in group.groupby(["size_group", "beta_group"]):
            members = [p for p in cell["permno"] if p in window.columns]
            if not members:
                continue
            ret = window[members].mean(axis=1, skipna=True)
            for ym, r in ret.items():
                rows.append({"ym": ym, "size_group": sg, "beta_group": bg,
                             "exret": r, "n_stocks": int(window.loc[ym, members]
                                                         .notna().sum())})
    out = pd.DataFrame(rows)
    return out.sort_values(["size_group", "beta_group", "ym"]).reset_index(drop=True)


def post_ranking_betas(port_returns: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    """Full-period sum-of-slopes beta for each of the 25 portfolios.

    One beta per portfolio, estimated once on the whole post-ranking series -
    which is the point of the exercise: a portfolio of ~20 stocks has a beta
    that can actually be estimated, unlike a single stock's.
    """
    mkt = market.rename("mkt")
    lag = market.shift(1).rename("mkt_lag")
    rows = []
    for (sg, bg), g in port_returns.groupby(["size_group", "beta_group"]):
        df = (g.set_index("ym")[["exret"]]
                .join(mkt).join(lag).dropna())
        if len(df) < MIN_MONTHS_PRE_RANKING:
            continue
        X = sm.add_constant(df[["mkt", "mkt_lag"]])
        fit = sm.OLS(df["exret"], X).fit()
        beta = fit.params["mkt"] + fit.params["mkt_lag"]
        rows.append({
            "size_group": sg, "beta_group": bg,
            "post_beta": beta,
            "beta_now": fit.params["mkt"],
            "beta_lag": fit.params["mkt_lag"],
            "se_now": fit.bse["mkt"],
            "n_months": len(df),
            "mean_exret": df["exret"].mean(),
            "sd_exret": df["exret"].std(),
        })
    return pd.DataFrame(rows)


def assign_post_ranking_betas(assignments: pd.DataFrame,
                              post: pd.DataFrame) -> pd.DataFrame:
    """Give every stock the post-ranking beta of the portfolio it sits in.

    This is FF's substitute for a stock-level beta. A stock's beta then only
    changes when it moves to a different cell at the next June sort.
    """
    return assignments.merge(post[["size_group", "beta_group", "post_beta"]],
                             on=["size_group", "beta_group"], how="left")


# ---------------------------------------------------------------------------
# the tables
# ---------------------------------------------------------------------------


def table_grid(values: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot a 25-row frame into the 5x5 grid FF print, size down, beta across."""
    grid = values.pivot(index="size_group", columns="beta_group", values=value_col)
    grid.index = [f"Size {i}" for i in grid.index]
    grid.columns = [f"Beta {j}" for j in grid.columns]
    return grid
