"""Tests for module 4.

Fama-MacBeth is two passes, and each can go wrong quietly: the monthly
cross-sections could pick up the wrong variables, and the second pass could
compute a t-statistic from the wrong standard error. Both are checked here
against data where the true slopes are known by construction.

Run from the fase1 folder:   python -m pytest tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import module4_fama_macbeth as m4  # noqa: E402


def _panel_with_known_slopes(n_months=120, n_stocks=60, slope=0.01, seed=0):
    """A panel built so that the true cross-sectional slope on beta is `slope`
    every month, with noise that averages out across months."""
    rng = np.random.default_rng(seed)
    months = pd.period_range("2010-01", periods=n_months, freq="M")
    betas = rng.uniform(0.5, 1.6, n_stocks)

    rows = []
    for ym in months:
        noise = rng.normal(scale=0.02, size=n_stocks)
        rows.append(pd.DataFrame({
            "ym": ym,
            "permno": np.arange(n_stocks),
            "beta": betas,
            "exret": slope * betas + noise,
            "ln_me": rng.normal(10, 1, n_stocks),
            "ln_bm": rng.normal(-1, 0.5, n_stocks),
            "is_financial": False,
            "neg_be": False,
        }))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# first pass: the monthly cross-sections
# ---------------------------------------------------------------------------


def test_one_regression_per_month():
    panel = _panel_with_known_slopes(n_months=36)
    slopes = m4.cross_section_regressions(panel, ["beta"])

    assert len(slopes) == 36
    assert slopes["ym"].is_monotonic_increasing
    assert set(slopes.columns) >= {"ym", "n", "r2", "intercept", "beta"}


def test_the_average_slope_recovers_the_true_one():
    panel = _panel_with_known_slopes(slope=0.01, n_months=240, seed=1)
    slopes = m4.cross_section_regressions(panel, ["beta"])

    assert slopes["beta"].mean() == pytest.approx(0.01, abs=0.002)


def test_a_variable_with_no_relation_gets_a_slope_near_zero():
    panel = _panel_with_known_slopes(n_months=240, seed=2)
    slopes = m4.cross_section_regressions(panel, ["beta", "ln_me"])

    assert abs(slopes["ln_me"].mean()) < 0.002


def test_a_month_with_too_few_stocks_is_skipped_not_fitted():
    panel = _panel_with_known_slopes(n_months=4, n_stocks=60)
    # leave two observations in the first month: fewer than the coefficients
    first = panel["ym"].min()
    keep = (panel["ym"] != first) | (panel.groupby("ym").cumcount() < 2)
    slopes = m4.cross_section_regressions(panel[keep], ["beta", "ln_me"])

    assert first not in set(slopes["ym"])
    assert len(slopes) == 3


def test_missing_values_drop_the_stock_not_the_month():
    panel = _panel_with_known_slopes(n_months=12)
    panel.loc[panel.index[:5], "ln_bm"] = np.nan
    slopes = m4.cross_section_regressions(panel, ["ln_bm"])

    assert len(slopes) == 12
    assert slopes["n"].iloc[0] == 55        # 60 stocks minus the 5 missing


# ---------------------------------------------------------------------------
# second pass: the average slope and its t-statistic
# ---------------------------------------------------------------------------


def test_the_fm_t_statistic_is_the_mean_over_its_own_standard_error():
    slopes = pd.DataFrame({
        "ym": pd.period_range("2020-01", periods=100, freq="M"),
        "beta": np.linspace(-0.01, 0.03, 100),
    })
    out = m4.fm_summary(slopes, ["beta"]).set_index("variable")

    series = slopes["beta"]
    expected_t = series.mean() / (series.std(ddof=1) / np.sqrt(len(series)))
    assert out.loc["beta", "mean"] == pytest.approx(series.mean())
    assert out.loc["beta", "fm_t"] == pytest.approx(expected_t)


def test_a_noisier_slope_series_gets_a_smaller_t_statistic():
    rng = np.random.default_rng(3)
    months = pd.period_range("2020-01", periods=200, freq="M")
    quiet = pd.DataFrame({"ym": months, "x": 0.01 + rng.normal(0, 0.01, 200)})
    noisy = pd.DataFrame({"ym": months, "x": 0.01 + rng.normal(0, 0.05, 200)})

    t_quiet = m4.fm_summary(quiet, ["x"])["fm_t"].iloc[0]
    t_noisy = m4.fm_summary(noisy, ["x"])["fm_t"].iloc[0]

    assert t_quiet > t_noisy


def test_newey_west_and_fm_agree_when_the_slopes_are_independent():
    # With no autocorrelation the HAC correction has nothing to correct.
    rng = np.random.default_rng(4)
    slopes = pd.DataFrame({
        "ym": pd.period_range("2010-01", periods=300, freq="M"),
        "x": rng.normal(0.01, 0.02, 300),
    })
    out = m4.fm_summary(slopes, ["x"]).iloc[0]

    assert out["nw_t"] == pytest.approx(out["fm_t"], rel=0.25)


def test_share_positive_counts_the_months_with_a_positive_slope():
    slopes = pd.DataFrame({
        "ym": pd.period_range("2020-01", periods=4, freq="M"),
        "x": [0.01, -0.01, 0.02, 0.03],
    })
    out = m4.fm_summary(slopes, ["x"]).iloc[0]

    assert out["share_positive"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# the CAPM test
# ---------------------------------------------------------------------------


def test_the_capm_test_compares_the_slope_with_the_market_month_by_month():
    months = pd.period_range("2020-01", periods=60, freq="M")
    rng = np.random.default_rng(5)
    mktrf = rng.normal(0.008, 0.04, 60)
    slopes = pd.DataFrame({"ym": months, "beta": mktrf, "intercept": 0.0})
    factors = pd.DataFrame({"ym": months, "mktrf": mktrf})

    out = m4.capm_slope_test(slopes, factors)

    # gamma_1 equals mktrf exactly here, so the difference is exactly zero
    assert out["mean_diff"] == pytest.approx(0.0, abs=1e-15)
    assert out["mean_gamma1"] == pytest.approx(out["mean_mktrf"])


def test_a_slope_that_is_not_the_market_premium_shows_up_as_a_difference():
    months = pd.period_range("2020-01", periods=120, freq="M")
    rng = np.random.default_rng(6)
    mktrf = rng.normal(0.008, 0.04, 120)
    slopes = pd.DataFrame({"ym": months, "beta": mktrf + 0.005,
                           "intercept": 0.0})
    factors = pd.DataFrame({"ym": months, "mktrf": mktrf})

    out = m4.capm_slope_test(slopes, factors)

    assert out["mean_diff"] == pytest.approx(0.005, abs=1e-12)
    assert out["t_diff"] > 10            # a constant gap has no sampling error


# ---------------------------------------------------------------------------
# samples
# ---------------------------------------------------------------------------


def test_the_main_sample_drops_financials_and_negative_book_equity():
    panel = pd.DataFrame({
        "permno": [1, 2, 3, 4],
        "is_financial": [False, True, False, True],
        "neg_be": [False, False, True, True],
    })
    main = m4.apply_sample(panel, exclude_financials=True,
                           exclude_negative_be=True)
    assert list(main["permno"]) == [1]

    with_fin = m4.apply_sample(panel, exclude_financials=False,
                               exclude_negative_be=True)
    assert list(with_fin["permno"]) == [1, 2]

    keep_neg = m4.apply_sample(panel, exclude_financials=True,
                               exclude_negative_be=False)
    assert list(keep_neg["permno"]) == [1, 3]


def test_the_specifications_are_the_six_from_table_three():
    assert len(m4.SPECS) == 6
    assert m4.SPECS["(1) beta"] == ["beta"]
    assert m4.SPECS["(6) beta + ln(ME) + ln(BE/ME)"] == ["beta", "ln_me", "ln_bm"]
    # the mu specification adds the negative book equity dummy
    assert "neg_be_dummy" in m4.MU_SPEC


def test_a_degenerate_difference_series_does_not_return_nan():
    # If the slope equals the market premium in every single month, the
    # difference is exactly zero with no variation. 0/0 would be nan, which
    # reads as a broken calculation rather than as "no difference".
    months = pd.period_range("2020-01", periods=24, freq="M")
    mktrf = np.full(24, 0.008)
    slopes = pd.DataFrame({"ym": months, "beta": mktrf, "intercept": 0.0})
    factors = pd.DataFrame({"ym": months, "mktrf": mktrf})

    out = m4.capm_slope_test(slopes, factors)

    assert out["mean_diff"] == pytest.approx(0.0)
    assert out["t_diff"] == 0.0
    assert not np.isnan(out["t_diff"])
