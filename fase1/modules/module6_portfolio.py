"""Module 6 - covariance matrix and tangency portfolios.

The covariance matrix comes from daily returns over the three years to
31-12-2025, shrunk with Ledoit-Wolf. Daily data is used because it is the only
way to get a well-conditioned matrix here: monthly returns give 120
observations for ~490 stocks, which cannot even be inverted, while three years
of daily data give about 750. Even 750 against 490 is not comfortable - the
sample eigenvalues are badly spread when N/T is this large - which is exactly
the situation Ledoit-Wolf shrinkage is designed for. It pulls the sample
matrix towards a scaled identity by an amount chosen to minimise expected
squared error, and the intensity it picks is reported so the size of the
correction is visible rather than hidden.

Two tangency portfolios are computed:

  unconstrained   w proportional to inverse(Sigma) (mu - rf), the textbook
                  solution from week 2. It is reported because it is what the
                  theory says, not because it is investable: with 490
                  estimated means it takes enormous offsetting long and short
                  positions.

  long-only, capped  the same objective maximised over w >= 0, w <= 5%,
                  sum(w) = 1. This is the one we can actually buy.

Maximising a Sharpe ratio over a polytope is not a quadratic program as it
stands, because the objective is a ratio. The Schaible transformation makes it
one: substituting y = w / (mu' w), the problem

    max (mu' w) / sqrt(w' Sigma w)   s.t.  w >= 0,  w <= c,  1'w = 1

becomes

    min y' Sigma y   s.t.  mu' y = 1,  y >= 0,  y <= c * (1'y)

which is convex, and w = y / (1'y) recovers the answer. This is exact, not an
approximation, and it avoids having to search over a risk-aversion grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import TRADING_DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# the return matrix
# ---------------------------------------------------------------------------


def daily_return_matrix(daily: pd.DataFrame, start: pd.Timestamp,
                        end: pd.Timestamp, candidates: list[int],
                        min_coverage: float = 0.98) -> tuple[pd.DataFrame, dict]:
    """Wide matrix of daily returns for the covariance estimate.

    Ledoit-Wolf needs a rectangular block with no holes, and there are two
    ways to get one. Dropping every stock with a single missing day would
    throw out firms that are missing two days out of 752. Dropping every day
    on which any stock is missing would, for a stock that listed part-way
    through, cost the other 480 stocks months of history.

    So we do both, in the order that costs least: keep the stocks that trade
    on at least `min_coverage` of the days, then drop the handful of days on
    which any of *those* stocks is missing. A stock that listed inside the
    window falls below the threshold, is named, and gets zero weight.
    """
    window = daily[(daily["date"] >= start) & (daily["date"] <= end)]
    wide = window.pivot(index="date", columns="permno", values="ret")
    wide = wide[[c for c in candidates if c in wide.columns]]

    n_days_raw = len(wide)
    coverage = wide.notna().sum() / n_days_raw
    keep = wide.columns[coverage >= min_coverage]
    dropped = {int(c): int(wide[c].notna().sum()) for c in wide.columns
               if c not in keep}

    block = wide[keep].dropna(axis=0, how="any")

    info = {
        "n_days_raw": n_days_raw,
        "n_days": len(block),
        "n_days_dropped": n_days_raw - len(block),
        "start": block.index.min(),
        "end": block.index.max(),
        "n_candidates": len(wide.columns),
        "n_kept": len(keep),
        "min_coverage": min_coverage,
        "excluded": dropped,
    }
    return block, info


# ---------------------------------------------------------------------------
# covariance
# ---------------------------------------------------------------------------


def ledoit_wolf_covariance(returns: pd.DataFrame,
                           periods_per_year: int = TRADING_DAYS_PER_YEAR
                           ) -> tuple[pd.DataFrame, dict]:
    """Ledoit-Wolf shrunk covariance, annualised.

    The comparison with the plain sample covariance is returned too: the
    condition number of the two matrices is the clearest evidence of whether
    the shrinkage was doing real work.
    """
    X = returns.to_numpy()
    lw = LedoitWolf(assume_centered=False).fit(X)

    shrunk = pd.DataFrame(lw.covariance_ * periods_per_year,
                          index=returns.columns, columns=returns.columns)
    sample = pd.DataFrame(np.cov(X, rowvar=False) * periods_per_year,
                          index=returns.columns, columns=returns.columns)

    info = {
        "shrinkage": float(lw.shrinkage_),
        "n_days": X.shape[0],
        "n_stocks": X.shape[1],
        "cond_sample": float(np.linalg.cond(sample.to_numpy())),
        "cond_shrunk": float(np.linalg.cond(shrunk.to_numpy())),
        "mean_vol_shrunk": float(np.sqrt(np.diag(shrunk)).mean()),
        "mean_corr": float(returns.corr().to_numpy()[
            np.triu_indices(X.shape[1], k=1)].mean()),
    }
    return shrunk, info


# ---------------------------------------------------------------------------
# tangency portfolios
# ---------------------------------------------------------------------------


def tangency_unconstrained(mu_excess: pd.Series, sigma: pd.DataFrame) -> pd.Series:
    """w proportional to inverse(Sigma) (mu - rf), scaled to sum to one.

    Solved with `np.linalg.solve` rather than by forming the inverse: the
    answer is the same and it is the numerically better-behaved way to do it.
    """
    assets = mu_excess.index
    w = np.linalg.solve(sigma.loc[assets, assets].to_numpy(),
                        mu_excess.to_numpy())
    return pd.Series(w / w.sum(), index=assets, name="weight")


def tangency_long_only(mu_excess: pd.Series, sigma: pd.DataFrame,
                       cap: float) -> tuple[pd.Series, dict]:
    """Maximum Sharpe ratio subject to w >= 0, w <= cap, sum(w) = 1.

    Uses the Schaible transformation described in the module docstring, so the
    problem solved is an exact convex quadratic program.
    """
    import cvxpy as cp

    assets = mu_excess.index
    mu = mu_excess.to_numpy()
    S = sigma.loc[assets, assets].to_numpy()
    S = (S + S.T) / 2.0                       # kill any asymmetry from rounding

    n = len(assets)
    y = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(y, cp.psd_wrap(S)))
    constraints = [mu @ y == 1, y >= 0, y <= cap * cp.sum(y)]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)

    if y.value is None:
        raise RuntimeError(f"the capped tangency problem did not solve: "
                           f"{problem.status}")

    w = y.value / y.value.sum()
    w = np.clip(w, 0.0, cap)
    w = w / w.sum()
    info = {"status": problem.status, "n_assets": n, "cap": cap,
            "n_at_cap": int((w > cap - 1e-6).sum()),
            "n_nonzero": int((w > 1e-6).sum())}
    return pd.Series(w, index=assets, name="weight"), info


def portfolio_stats(weights: pd.Series, mu_excess: pd.Series,
                    sigma: pd.DataFrame, rf_annual: float) -> dict:
    """Expected return, volatility and Sharpe ratio of a set of weights."""
    assets = weights.index
    w = weights.to_numpy()
    mu = mu_excess.reindex(assets).to_numpy()
    S = sigma.loc[assets, assets].to_numpy()

    excess = float(w @ mu)
    var = float(w @ S @ w)
    vol = float(np.sqrt(var))
    return {
        "excess_return": excess,
        "total_return": excess + rf_annual,
        "volatility": vol,
        "sharpe": excess / vol,
        "gross_exposure": float(np.abs(w).sum()),
        "long_exposure": float(w[w > 0].sum()),
        "short_exposure": float(-w[w < 0].sum()),
        "n_long": int((w > 1e-6).sum()),
        "n_short": int((w < -1e-6).sum()),
        "max_weight": float(w.max()),
        "min_weight": float(w.min()),
        "effective_n": float(1.0 / np.sum(w ** 2)),
    }


# ---------------------------------------------------------------------------
# the single-index covariance matrix
# ---------------------------------------------------------------------------


def realised_beta(portfolio_returns: pd.Series,
                  market_returns: pd.Series) -> dict:
    """Regress a portfolio's realised returns on the market.

    This is the consistency check that motivates everything below. If the
    expected return says a portfolio has a beta of 1.09 but its own returns
    say 0.53, then mu and Sigma disagree about what the portfolio is, and the
    optimiser has been exploiting the disagreement rather than the economics.
    """
    import statsmodels.api as sm

    df = pd.concat([portfolio_returns.rename("p"),
                    market_returns.rename("m")], axis=1).dropna()
    fit = sm.OLS(df["p"], sm.add_constant(df["m"])).fit()
    return {
        "beta": float(fit.params["m"]),
        "t": float(fit.tvalues["m"]),
        "r2": float(fit.rsquared),
        "alpha_annual": float(fit.params["const"]) * TRADING_DAYS_PER_YEAR,
        "n_days": len(df),
    }


def residual_variances(returns: pd.DataFrame, market: pd.Series,
                       betas: pd.Series,
                       periods_per_year: int = TRADING_DAYS_PER_YEAR
                       ) -> tuple[pd.Series, dict]:
    """Annualised variance of r_i - beta_i * r_m, with beta *imposed*.

    The betas are the Fama-French post-ranking betas already used in mu, not
    betas re-estimated from this daily window. That is the whole point: the
    single-index matrix has to describe the same asset that mu describes.

    An imposed beta that is too high for a stock leaves the market factor
    partly in the residual, which inflates that stock's residual variance and
    makes the optimiser want less of it. That self-correction is a feature: it
    is how the model answers "if you claim this stock has a beta of 1.33, then
    you must also accept the risk that comes with it".
    """
    aligned = market.reindex(returns.index).to_numpy()[:, None]
    b = betas.reindex(returns.columns).to_numpy()[None, :]
    resid = pd.DataFrame(returns.to_numpy() - aligned * b,
                         index=returns.index, columns=returns.columns)
    var = resid.var(ddof=1) * periods_per_year

    info = {
        "n_days": len(returns),
        "mean_resid_vol": float(np.sqrt(var).mean()),
        "min_resid_vol": float(np.sqrt(var).min()),
        "max_resid_vol": float(np.sqrt(var).max()),
    }
    return var, info


def single_index_covariance(betas: pd.Series, resid_var: pd.Series,
                            market_var: float) -> pd.DataFrame:
    """Sigma = sigma_m^2 * beta beta' + D.

    Every pair of stocks is correlated only through the market, and each
    stock's own risk is its residual variance. With mu - rf = MRP * beta this
    makes the model internally consistent, and the tangency portfolio has a
    closed form: by the Sherman-Morrison identity,

        inverse(Sigma) beta = D^-1 beta / (1 + sigma_m^2 beta' D^-1 beta)

    so the unconstrained tangency weights are exactly proportional to
    beta_i / residual variance_i. There is nothing left for the optimiser to
    arbitrage - it can only trade off a stock's beta against its own risk.
    """
    assets = betas.index
    b = betas.to_numpy()
    sigma = pd.DataFrame(market_var * np.outer(b, b), index=assets, columns=assets)
    np.fill_diagonal(sigma.values,
                     market_var * b ** 2 + resid_var.reindex(assets).to_numpy())
    return sigma


def volatility_screen(daily: pd.DataFrame, recent_start: pd.Timestamp,
                      recent_end: pd.Timestamp, prior_start: pd.Timestamp,
                      prior_end: pd.Timestamp, min_recent_days: int = 55,
                      min_prior_days: int = 150) -> pd.DataFrame:
    """Find stocks whose volatility collapsed in the most recent quarter.

    A stock under an agreed all-cash bid stops behaving like equity: its price
    pins to the offer and its volatility falls to almost nothing. Comparing
    the recent quarter with the rest of the year isolates that collapse, which
    a level screen cannot do - a utility has low volatility all year round.
    """
    def _vol(start, end):
        window = daily[(daily["date"] >= start) & (daily["date"] <= end)]
        grouped = window.dropna(subset=["ret"]).groupby("permno")["ret"]
        return (grouped.std() * np.sqrt(TRADING_DAYS_PER_YEAR),
                grouped.size(), grouped.mean() * TRADING_DAYS_PER_YEAR)

    recent_vol, recent_n, recent_mean = _vol(recent_start, recent_end)
    prior_vol, prior_n, _ = _vol(prior_start, prior_end)

    out = pd.DataFrame({
        "recent_vol": recent_vol, "recent_n": recent_n,
        "recent_mean": recent_mean, "prior_vol": prior_vol, "prior_n": prior_n,
    }).dropna()
    out = out[(out["recent_n"] >= min_recent_days)
              & (out["prior_n"] >= min_prior_days)]
    out["ratio"] = out["recent_vol"] / out["prior_vol"]
    return out.sort_values("ratio")
