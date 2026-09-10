"""Tests for module 3.

The things worth testing here are: that the sum-of-slopes beta really is the
sum of the two slopes, that a stock without enough history gets no beta at
all rather than a bad one, that the double sort sorts on size first and beta
second, and that the holding window is July t to June t+1 and nothing else.

Run from the fase1 folder:   python -m pytest tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import module3_betas as m3  # noqa: E402


# ---------------------------------------------------------------------------
# the sum-of-slopes beta
# ---------------------------------------------------------------------------


def test_sum_slope_beta_recovers_the_two_slopes_exactly():
    # Build a return series that is exactly 0.5 * market_t + 0.3 * market_{t-1}.
    # The beta FF report is the sum, so the answer must be 0.8.
    rng = np.random.default_rng(0)
    x_now = rng.normal(size=120)
    x_lag = np.roll(x_now, 1)
    x_lag[0] = np.nan
    y = 0.5 * x_now + 0.3 * x_lag

    assert m3.sum_slope_beta(y, x_now, x_lag) == pytest.approx(0.8)


def test_sum_slope_beta_is_not_the_single_slope():
    # A stock that only responds to last month's market has a current-month
    # slope near zero but a beta of 0.9. Using one slope would report ~0.
    rng = np.random.default_rng(1)
    x_now = rng.normal(size=120)
    x_lag = np.roll(x_now, 1)
    x_lag[0] = np.nan
    y = 0.9 * x_lag

    assert m3.sum_slope_beta(y, x_now, x_lag) == pytest.approx(0.9)


def test_beta_is_missing_when_there_are_fewer_than_24_usable_months():
    rng = np.random.default_rng(2)
    x_now = rng.normal(size=23)
    x_lag = np.roll(x_now, 1)
    y = 1.0 * x_now

    assert np.isnan(m3.sum_slope_beta(y, x_now, x_lag))


def test_months_with_a_missing_return_are_dropped_not_treated_as_zero():
    rng = np.random.default_rng(3)
    x_now = rng.normal(size=60)
    x_lag = np.roll(x_now, 1)
    x_lag[0] = np.nan
    y = 1.5 * x_now + 0.0 * x_lag
    y[5:15] = np.nan            # ten missing months

    # A zero-filled return would drag the estimate towards zero; dropping the
    # months leaves the true 1.5 intact.
    assert m3.sum_slope_beta(y, x_now, x_lag) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# factors
# ---------------------------------------------------------------------------


def test_factors_after_the_trading_date_are_dropped(tmp_path):
    csv = tmp_path / "ff.csv"
    csv.write_text(
        "dateff,smb,hml,mktrf,rf,umd\n"
        "2025-11-28,0.01,0.01,0.02,0.003,0.01\n"
        "2025-12-31,0.01,0.01,0.02,0.003,0.01\n"
        "2026-01-30,0.01,0.01,0.02,0.003,0.01\n",
        encoding="utf-8",
    )
    out, info = m3.load_ff_factors(csv)

    assert info["n_dropped_future"] == 1
    assert out["date"].max() == pd.Timestamp("2025-12-31")


def test_factors_in_percent_are_refused_instead_of_used(tmp_path):
    # The same numbers in percent would make every beta 100 times too small.
    csv = tmp_path / "ff_pct.csv"
    rows = "\n".join(
        f"2020-{m:02d}-28,1.0,1.0,{v},0.3,1.0" for m, v in
        zip(range(1, 13), [5, -4, 6, -3, 4, -5, 3, -6, 5, -4, 6, -3])
    )
    csv.write_text("dateff,smb,hml,mktrf,rf,umd\n" + rows + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="percent"):
        m3.load_ff_factors(csv)


def test_market_return_is_the_excess_return_plus_the_risk_free_rate(tmp_path):
    csv = tmp_path / "ff.csv"
    csv.write_text(
        "dateff,smb,hml,mktrf,rf,umd\n2025-12-31,0.01,0.01,0.0200,0.0030,0.01\n",
        encoding="utf-8",
    )
    out, _ = m3.load_ff_factors(csv)
    assert out["mkt"].iloc[0] == pytest.approx(0.023)


# ---------------------------------------------------------------------------
# the double sort
# ---------------------------------------------------------------------------


def _sortable_universe(n=100):
    """n stocks whose size and beta are deliberately uncorrelated."""
    rng = np.random.default_rng(7)
    permnos = np.arange(1, n + 1)
    size = pd.Series(np.exp(rng.normal(10, 1, n)), index=permnos)
    beta = pd.Series(rng.normal(1.0, 0.4, n), index=permnos)
    return size, beta


def test_size_groups_are_equal_sized_and_ordered():
    size, beta = _sortable_universe(100)
    out = m3.form_portfolios(size, beta)

    counts = out["size_group"].value_counts()
    assert set(counts.index) == {1, 2, 3, 4, 5}
    assert counts.min() == counts.max() == 20

    means = out.groupby("size_group")["me"].mean()
    assert means.is_monotonic_increasing     # group 1 really is the smallest


def test_beta_is_sorted_inside_each_size_group_not_across_them():
    size, beta = _sortable_universe(100)
    out = m3.form_portfolios(size, beta)

    # within every size group, the beta groups are ordered by beta ...
    for sg, g in out.groupby("size_group"):
        means = g.groupby("beta_group")["pre_beta"].mean()
        assert means.is_monotonic_increasing
        assert len(g) == 20 and g["beta_group"].value_counts().max() == 4

    # ... and each size group contains all five beta groups
    assert out.groupby("size_group")["beta_group"].nunique().eq(5).all()


def test_a_stock_without_a_beta_is_left_out_of_the_sort_entirely():
    size, beta = _sortable_universe(100)
    beta.iloc[:10] = np.nan
    out = m3.form_portfolios(size, beta.dropna())

    assert len(out) == 90
    assert not out["permno"].isin(size.index[:10]).any()


# ---------------------------------------------------------------------------
# the holding window
# ---------------------------------------------------------------------------


def test_holding_window_is_july_t_to_june_t_plus_one():
    index = pd.period_range("2019-01", "2022-12", freq="M")
    months = m3.holding_months(2020, index)

    assert months.min() == pd.Period("2020-07", freq="M")
    assert months.max() == pd.Period("2021-06", freq="M")
    assert len(months) == 12


def test_holding_window_is_clipped_at_the_end_of_the_data():
    # The June 2025 sort can only be held until the data stops in December.
    index = pd.period_range("2024-01", "2025-12", freq="M")
    months = m3.holding_months(2025, index)

    assert months.min() == pd.Period("2025-07", freq="M")
    assert months.max() == pd.Period("2025-12", freq="M")
    assert len(months) == 6


# ---------------------------------------------------------------------------
# portfolio returns
# ---------------------------------------------------------------------------


def test_portfolio_return_is_the_equal_weighted_mean_of_its_members():
    index = pd.period_range("2020-07", "2020-08", freq="M")
    panel = pd.DataFrame({1: [0.10, 0.20], 2: [0.00, 0.40]}, index=index)
    assignments = pd.DataFrame(
        {"ff_year": [2020, 2020], "permno": [1, 2],
         "size_group": [1, 1], "beta_group": [1, 1]}
    )
    out = m3.portfolio_returns(assignments, panel)

    assert out["exret"].tolist() == pytest.approx([0.05, 0.30])
    assert out["n_stocks"].tolist() == [2, 2]


def test_a_stock_that_stops_trading_drops_out_of_the_average():
    # In August only stock 1 trades, so the portfolio return is its return -
    # not an average that silently treats the missing stock as a zero.
    index = pd.period_range("2020-07", "2020-08", freq="M")
    panel = pd.DataFrame({1: [0.10, 0.20], 2: [0.00, np.nan]}, index=index)
    assignments = pd.DataFrame(
        {"ff_year": [2020, 2020], "permno": [1, 2],
         "size_group": [1, 1], "beta_group": [1, 1]}
    )
    out = m3.portfolio_returns(assignments, panel).set_index("ym")

    assert out.loc[pd.Period("2020-08", freq="M"), "exret"] == pytest.approx(0.20)
    assert out.loc[pd.Period("2020-08", freq="M"), "n_stocks"] == 1


# ---------------------------------------------------------------------------
# handing the portfolio beta back to the stocks
# ---------------------------------------------------------------------------


def test_every_stock_receives_the_beta_of_its_own_cell():
    assignments = pd.DataFrame(
        {"ff_year": [2020, 2020, 2020], "permno": [1, 2, 3],
         "size_group": [1, 1, 2], "beta_group": [1, 2, 1]}
    )
    post = pd.DataFrame(
        {"size_group": [1, 1, 2], "beta_group": [1, 2, 1],
         "post_beta": [0.7, 1.3, 0.9]}
    )
    out = m3.assign_post_ranking_betas(assignments, post).set_index("permno")

    assert out.loc[1, "post_beta"] == pytest.approx(0.7)
    assert out.loc[2, "post_beta"] == pytest.approx(1.3)
    assert out.loc[3, "post_beta"] == pytest.approx(0.9)
