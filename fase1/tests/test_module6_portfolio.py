"""Tests for module 6.

Three things here could be wrong without looking wrong: the tangency formula,
the capped optimiser (the Schaible transformation is not obviously correct at
a glance), and the claim made in module 5 that the market risk premium does
not affect the weights. Each is tested against something independently known.

Run from the fase1 folder:   python -m pytest tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import module6_portfolio as m6  # noqa: E402


def _two_assets():
    """Two uncorrelated assets: A has twice the excess return of B and the
    same variance, so the tangency portfolio must hold twice as much A."""
    mu = pd.Series({1: 0.10, 2: 0.05})
    sigma = pd.DataFrame([[0.04, 0.0], [0.0, 0.04]], index=[1, 2], columns=[1, 2])
    return mu, sigma


def _three_assets(seed=0):
    rng = np.random.default_rng(seed)
    n = 3
    A = rng.normal(size=(n, n))
    sigma = pd.DataFrame(A @ A.T + np.eye(n) * 0.5,
                         index=range(n), columns=range(n))
    mu = pd.Series(rng.uniform(0.02, 0.15, n), index=range(n))
    return mu, sigma


# ---------------------------------------------------------------------------
# the unconstrained tangency portfolio
# ---------------------------------------------------------------------------


def test_tangency_matches_the_textbook_formula_on_a_case_we_can_do_by_hand():
    mu, sigma = _two_assets()
    w = m6.tangency_unconstrained(mu, sigma)

    # inverse(Sigma) mu is proportional to (0.10, 0.05), so w = (2/3, 1/3)
    assert w.loc[1] == pytest.approx(2 / 3)
    assert w.loc[2] == pytest.approx(1 / 3)
    assert w.sum() == pytest.approx(1.0)


def test_tangency_equals_the_explicit_inverse_solution():
    mu, sigma = _three_assets()
    w = m6.tangency_unconstrained(mu, sigma)

    raw = np.linalg.inv(sigma.to_numpy()) @ mu.to_numpy()
    assert w.to_numpy() == pytest.approx(raw / raw.sum())


def test_the_tangency_portfolio_has_the_highest_sharpe_ratio_available():
    # Nudging the optimal weights in any direction must lower the Sharpe ratio.
    mu, sigma = _three_assets(seed=3)
    w = m6.tangency_unconstrained(mu, sigma)
    best = m6.portfolio_stats(w, mu, sigma, 0.01)["sharpe"]

    rng = np.random.default_rng(11)
    for _ in range(200):
        step = rng.normal(scale=0.05, size=len(w))
        step -= step.mean()                       # keep the weights summing to 1
        other = pd.Series(w.to_numpy() + step, index=w.index)
        assert m6.portfolio_stats(other, mu, sigma, 0.01)["sharpe"] <= best + 1e-9


# ---------------------------------------------------------------------------
# the claim that the market risk premium does not change the weights
# ---------------------------------------------------------------------------


def test_scaling_every_expected_return_leaves_the_weights_unchanged():
    # This is the reason the choice of MRP does not affect which stocks we
    # hold: mu - rf = MRP * beta, and MRP is a positive scalar on the vector.
    mu, sigma = _three_assets(seed=5)

    base = m6.tangency_unconstrained(mu, sigma)
    doubled = m6.tangency_unconstrained(mu * 2.0, sigma)
    tenth = m6.tangency_unconstrained(mu * 0.1, sigma)

    assert doubled.to_numpy() == pytest.approx(base.to_numpy())
    assert tenth.to_numpy() == pytest.approx(base.to_numpy())


def test_scaling_leaves_the_capped_portfolio_unchanged_too():
    mu, sigma = _three_assets(seed=6)

    base, _ = m6.tangency_long_only(mu, sigma, cap=0.6)
    scaled, _ = m6.tangency_long_only(mu * 7.0, sigma, cap=0.6)

    assert scaled.to_numpy() == pytest.approx(base.to_numpy(), abs=1e-6)


def test_scaling_expected_returns_does_scale_the_sharpe_ratio():
    # The weights do not move, but the reported Sharpe ratio does - which is
    # why the MRP still matters for how much we hold at the risk-free rate.
    mu, sigma = _three_assets(seed=7)
    w = m6.tangency_unconstrained(mu, sigma)

    one = m6.portfolio_stats(w, mu, sigma, 0.01)["sharpe"]
    two = m6.portfolio_stats(w, mu * 2.0, sigma, 0.01)["sharpe"]

    assert two == pytest.approx(2 * one)


# ---------------------------------------------------------------------------
# the long-only capped optimiser
# ---------------------------------------------------------------------------


def test_capped_weights_respect_the_cap_and_never_go_short():
    mu, sigma = _three_assets(seed=8)
    w, info = m6.tangency_long_only(mu, sigma, cap=0.5)

    assert w.min() >= -1e-9
    assert w.max() <= 0.5 + 1e-6
    assert w.sum() == pytest.approx(1.0)
    assert info["status"] == "optimal"


def test_a_cap_that_does_not_bind_reproduces_the_unconstrained_answer():
    # With two positive-weight assets and a generous cap, the constrained and
    # unconstrained solutions must coincide.
    mu, sigma = _two_assets()
    unc = m6.tangency_unconstrained(mu, sigma)
    con, _ = m6.tangency_long_only(mu, sigma, cap=0.99)

    assert con.to_numpy() == pytest.approx(unc.to_numpy(), abs=1e-6)


def test_a_tight_cap_forces_equal_weights():
    # Five assets and a 20% cap leaves exactly one feasible portfolio.
    n = 5
    rng = np.random.default_rng(9)
    A = rng.normal(size=(n, n))
    sigma = pd.DataFrame(A @ A.T + np.eye(n), index=range(n), columns=range(n))
    mu = pd.Series(rng.uniform(0.05, 0.20, n), index=range(n))

    w, _ = m6.tangency_long_only(mu, sigma, cap=0.20)

    assert w.to_numpy() == pytest.approx(np.full(n, 0.2), abs=1e-6)


def test_the_capped_portfolio_beats_nearby_feasible_portfolios():
    mu, sigma = _three_assets(seed=10)
    w, _ = m6.tangency_long_only(mu, sigma, cap=0.5)
    best = m6.portfolio_stats(w, mu, sigma, 0.01)["sharpe"]

    rng = np.random.default_rng(12)
    for _ in range(300):
        cand = rng.dirichlet(np.ones(len(w)))
        if cand.max() > 0.5:
            continue                              # infeasible, skip
        other = pd.Series(cand, index=w.index)
        assert m6.portfolio_stats(other, mu, sigma, 0.01)["sharpe"] <= best + 1e-6


# ---------------------------------------------------------------------------
# covariance
# ---------------------------------------------------------------------------


def test_ledoit_wolf_is_annualised_and_better_conditioned_than_the_sample():
    rng = np.random.default_rng(4)
    # 60 days for 40 stocks: far too few, which is when shrinkage matters
    R = pd.DataFrame(rng.normal(scale=0.01, size=(60, 40)))
    sigma, info = m6.ledoit_wolf_covariance(R, periods_per_year=252)

    assert 0.0 < info["shrinkage"] < 1.0
    assert info["cond_shrunk"] < info["cond_sample"]
    assert np.allclose(sigma.to_numpy(), sigma.to_numpy().T)
    assert np.linalg.eigvalsh(sigma.to_numpy()).min() > 0     # positive definite

    # a daily variance of 0.01^2 annualises to 252 * 0.0001
    assert np.diag(sigma).mean() == pytest.approx(252 * 1e-4, rel=0.3)


# ---------------------------------------------------------------------------
# building the return matrix
# ---------------------------------------------------------------------------


def _daily_frame():
    """Three stocks over ten days. Stock 3 lists half way through; stock 2 is
    missing a single day."""
    dates = pd.bdate_range("2025-01-01", periods=10)
    rows = []
    for d_i, d in enumerate(dates):
        rows.append({"permno": 1, "date": d, "ret": 0.01})
        rows.append({"permno": 2, "date": d, "ret": np.nan if d_i == 3 else 0.02})
        if d_i >= 5:
            rows.append({"permno": 3, "date": d, "ret": 0.03})
    return pd.DataFrame(rows), dates


def test_a_stock_that_listed_inside_the_window_is_excluded():
    # Stock 3 trades on 5 of 10 days, so it is out at any sensible threshold.
    # At 98% stock 2 (9 of 10 days) is out as well - with only ten days a
    # single gap is 10%, which is why the real window of 752 days tolerates a
    # couple of missing days and this toy one does not.
    daily, dates = _daily_frame()
    block, info = m6.daily_return_matrix(daily, dates[0], dates[-1],
                                         candidates=[1, 2, 3],
                                         min_coverage=0.98)

    assert 3 not in block.columns
    assert info["excluded"] == {2: 9, 3: 5}
    assert list(block.columns) == [1]


def test_a_stock_missing_one_day_is_kept_and_the_day_is_dropped():
    # Stock 2 trades on 9 of 10 days, above the threshold, so it stays and the
    # single bad day is removed for everyone instead.
    daily, dates = _daily_frame()
    block, info = m6.daily_return_matrix(daily, dates[0], dates[-1],
                                         candidates=[1, 2],
                                         min_coverage=0.85)

    assert list(block.columns) == [1, 2]
    assert info["n_days"] == 9
    assert info["n_days_dropped"] == 1
    assert not block.isna().any().any()


def test_the_returned_block_has_no_holes():
    daily, dates = _daily_frame()
    block, _ = m6.daily_return_matrix(daily, dates[0], dates[-1],
                                      candidates=[1, 2, 3], min_coverage=0.5)
    assert not block.isna().any().any()


# ---------------------------------------------------------------------------
# portfolio statistics
# ---------------------------------------------------------------------------


def test_portfolio_statistics_are_computed_as_defined():
    mu = pd.Series({1: 0.10, 2: 0.05})
    sigma = pd.DataFrame([[0.04, 0.0], [0.0, 0.09]], index=[1, 2], columns=[1, 2])
    w = pd.Series({1: 0.5, 2: 0.5})

    st = m6.portfolio_stats(w, mu, sigma, rf_annual=0.01)

    assert st["excess_return"] == pytest.approx(0.075)
    assert st["total_return"] == pytest.approx(0.085)
    # variance = 0.25*0.04 + 0.25*0.09 = 0.0325
    assert st["volatility"] == pytest.approx(np.sqrt(0.0325))
    assert st["sharpe"] == pytest.approx(0.075 / np.sqrt(0.0325))
    assert st["effective_n"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# the single-index covariance matrix
# ---------------------------------------------------------------------------


def test_single_index_matrix_has_the_structure_it_claims():
    betas = pd.Series({1: 0.8, 2: 1.2})
    resid = pd.Series({1: 0.04, 2: 0.09})
    sigma = m6.single_index_covariance(betas, resid, market_var=0.0225)

    # off-diagonal is purely the market: 0.0225 * 0.8 * 1.2
    assert sigma.loc[1, 2] == pytest.approx(0.0225 * 0.8 * 1.2)
    assert sigma.loc[2, 1] == pytest.approx(sigma.loc[1, 2])
    # diagonal is market risk plus own risk
    assert sigma.loc[1, 1] == pytest.approx(0.0225 * 0.64 + 0.04)
    assert sigma.loc[2, 2] == pytest.approx(0.0225 * 1.44 + 0.09)


def test_the_tangency_weights_are_exactly_beta_over_residual_variance():
    # This is the claim the whole single-index choice rests on: with
    # mu - rf = MRP * beta and Sigma = sigma_m^2 beta beta' + D, the
    # Sherman-Morrison identity makes w proportional to beta_i / D_ii.
    rng = np.random.default_rng(21)
    n = 30
    betas = pd.Series(rng.uniform(0.5, 1.7, n), index=range(n))
    resid = pd.Series(rng.uniform(0.02, 0.20, n), index=range(n))
    market_var = 0.0225

    sigma = m6.single_index_covariance(betas, resid, market_var)
    mu = betas * 0.08                       # MRP * beta, any positive MRP
    w = m6.tangency_unconstrained(mu, sigma)

    expected = betas / resid
    expected = expected / expected.sum()

    assert w.to_numpy() == pytest.approx(expected.to_numpy(), abs=1e-12)


def test_the_single_index_tangency_portfolio_is_naturally_long_only():
    # Every beta and every residual variance is positive, so beta_i / D_ii is
    # positive for every stock: this portfolio never wants to short anything,
    # which is why the 5% cap is the only constraint that can bind.
    rng = np.random.default_rng(22)
    n = 50
    betas = pd.Series(rng.uniform(0.5, 1.7, n), index=range(n))
    resid = pd.Series(rng.uniform(0.02, 0.20, n), index=range(n))

    sigma = m6.single_index_covariance(betas, resid, 0.0225)
    w = m6.tangency_unconstrained(betas * 0.08, sigma)

    assert w.min() > 0


def test_a_higher_beta_earns_more_weight_when_residual_risk_is_equal():
    betas = pd.Series({1: 1.5, 2: 0.75})
    resid = pd.Series({1: 0.05, 2: 0.05})
    sigma = m6.single_index_covariance(betas, resid, 0.0225)
    w = m6.tangency_unconstrained(betas * 0.08, sigma)

    assert w.loc[1] == pytest.approx(2 * w.loc[2])


def test_residual_variance_uses_the_imposed_beta_not_a_refitted_one():
    # A stock that is exactly 1.2 times the market has no residual risk when
    # beta is imposed at 1.2, and visible residual risk when it is imposed at
    # 0.8 - because the leftover market exposure lands in the residual.
    rng = np.random.default_rng(23)
    market = pd.Series(rng.normal(scale=0.01, size=500))
    returns = pd.DataFrame({1: 1.2 * market})

    right, _ = m6.residual_variances(returns, market, pd.Series({1: 1.2}))
    wrong, _ = m6.residual_variances(returns, market, pd.Series({1: 0.8}))

    assert right.iloc[0] == pytest.approx(0.0, abs=1e-20)
    assert wrong.iloc[0] > 0


def test_realised_beta_recovers_a_known_exposure():
    rng = np.random.default_rng(24)
    market = pd.Series(rng.normal(scale=0.01, size=500))
    portfolio = 0.6 * market + pd.Series(rng.normal(scale=0.001, size=500))

    out = m6.realised_beta(portfolio, market)

    assert out["beta"] == pytest.approx(0.6, abs=0.02)


# ---------------------------------------------------------------------------
# the takeover screen
# ---------------------------------------------------------------------------


def test_the_screen_ranks_a_volatility_collapse_first():
    # Stock 1 is a deal: normal volatility, then almost none. Stock 2 is a
    # utility: low volatility throughout, so its ratio is near one and it must
    # not be confused with a deal.
    rng = np.random.default_rng(25)
    prior = pd.bdate_range("2025-01-01", periods=190)
    recent = pd.bdate_range("2025-10-01", periods=60)

    rows = []
    for d in prior:
        rows.append({"permno": 1, "date": d, "ret": rng.normal(scale=0.02)})
        rows.append({"permno": 2, "date": d, "ret": rng.normal(scale=0.005)})
    for d in recent:
        rows.append({"permno": 1, "date": d, "ret": rng.normal(scale=0.0005)})
        rows.append({"permno": 2, "date": d, "ret": rng.normal(scale=0.005)})
    daily = pd.DataFrame(rows)

    out = m6.volatility_screen(daily, recent[0], recent[-1], prior[0], prior[-1])

    assert out.index[0] == 1
    assert out.loc[1, "ratio"] < 0.15
    assert out.loc[2, "ratio"] == pytest.approx(1.0, abs=0.3)
