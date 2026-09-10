"""Tests for module 5.

The expected return is a one-line formula, so what is worth testing is the
plumbing around it: that a factor file in percent is converted rather than
used as-is, that the long history is truncated at the trading date, and that
anchoring a robustness variant replaces the level without disturbing the
tilts.

Run from the fase1 folder:   python -m pytest tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import module5_expected_returns as m5  # noqa: E402


# ---------------------------------------------------------------------------
# E[r] - rf = MRP * beta
# ---------------------------------------------------------------------------


def test_expected_excess_return_is_the_premium_times_beta():
    # 0.5% per month is 6% a year; a beta of 1.5 should give 9%.
    betas = pd.Series({1: 1.5, 2: 1.0, 3: 0.0})
    out = m5.mu_capm(betas, mrp_monthly=0.005, rf_annual=0.01)

    assert out.loc[1, "mu_excess"] == pytest.approx(0.09)
    assert out.loc[2, "mu_excess"] == pytest.approx(0.06)
    assert out.loc[3, "mu_excess"] == pytest.approx(0.00)


def test_total_expected_return_adds_the_risk_free_rate():
    betas = pd.Series({1: 1.0})
    out = m5.mu_capm(betas, mrp_monthly=0.005, rf_annual=0.01)

    assert out.loc[1, "mu_total"] == pytest.approx(0.07)     # 6% + 1%


def test_a_zero_beta_stock_earns_only_the_risk_free_rate():
    out = m5.mu_capm(pd.Series({1: 0.0}), mrp_monthly=0.005, rf_annual=0.01)
    assert out.loc[1, "mu_total"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# the market risk premium
# ---------------------------------------------------------------------------


def test_market_risk_premium_annualises_by_twelve():
    factors = pd.DataFrame({
        "date": pd.date_range("2020-01-31", periods=12, freq="ME"),
        "mktrf": [0.01] * 12,
    })
    mrp = m5.market_risk_premium(factors)

    assert mrp["monthly"] == pytest.approx(0.01)
    assert mrp["annual"] == pytest.approx(0.12)
    assert mrp["n_months"] == 12


def test_long_factor_file_in_percent_is_converted_to_decimals(tmp_path):
    # Ken French publishes percent; WRDS publishes decimals. Reading percent
    # as decimals would make the premium 100 times too large.
    csv = tmp_path / "ff_long.csv"
    rows = "\n".join(f"{2000 + i // 12}{i % 12 + 1:02d},{v}"
                     for i, v in enumerate([5, -4, 6, -3, 4, -5] * 4))
    csv.write_text("Date,Mkt-RF\n" + rows + "\n", encoding="utf-8")

    out = m5.load_long_factors(csv)

    assert out.attrs["converted_from_percent"] is True
    assert abs(out["mktrf"]).max() < 0.5          # now in decimals


def test_long_factor_file_already_in_decimals_is_left_alone(tmp_path):
    csv = tmp_path / "ff_long.csv"
    rows = "\n".join(f"2020-{m:02d}-28,0.0{m}" for m in range(1, 10))
    csv.write_text("dateff,mktrf\n" + rows + "\n", encoding="utf-8")

    out = m5.load_long_factors(csv)

    assert out.attrs["converted_from_percent"] is False
    assert out["mktrf"].iloc[0] == pytest.approx(0.01)


def test_long_factor_history_stops_at_the_trading_date(tmp_path):
    csv = tmp_path / "ff_long.csv"
    csv.write_text(
        "dateff,mktrf\n2025-11-28,0.01\n2025-12-31,0.02\n2026-01-30,0.03\n",
        encoding="utf-8")

    out = m5.load_long_factors(csv)

    assert len(out) == 2
    assert out["date"].max() == pd.Timestamp("2025-12-31")


# ---------------------------------------------------------------------------
# anchoring a robustness variant
# ---------------------------------------------------------------------------


def test_anchoring_sets_the_average_expected_return_exactly():
    chars = pd.DataFrame({"beta": [0.8, 1.0, 1.2], "ln_me": [9.0, 10.0, 11.0]})
    out = m5.mu_from_slopes(chars, {"beta": 0.007, "ln_me": -0.003},
                            anchor_excess=0.08)

    assert out.mean() == pytest.approx(0.08)


def test_anchoring_shifts_the_level_but_not_the_spread():
    chars = pd.DataFrame({"beta": [0.8, 1.0, 1.2], "ln_me": [9.0, 10.0, 11.0]})
    slopes = {"beta": 0.007, "ln_me": -0.003}

    raw = m5.mu_from_slopes(chars, slopes, anchor_excess=None)
    anchored = m5.mu_from_slopes(chars, slopes, anchor_excess=0.08)

    # every pairwise difference survives the anchoring untouched
    assert (anchored - raw).std() == pytest.approx(0.0, abs=1e-12)
    assert (anchored.diff().dropna() - raw.diff().dropna()).abs().max() < 1e-12


def test_a_negative_slope_gives_a_lower_expected_return_to_a_bigger_firm():
    chars = pd.DataFrame({"ln_me": [9.0, 12.0]})
    out = m5.mu_from_slopes(chars, {"ln_me": -0.003}, anchor_excess=0.08)

    assert out.iloc[1] < out.iloc[0]
    # 3 log points at -0.3% a month is -10.8% a year of difference
    assert (out.iloc[0] - out.iloc[1]) == pytest.approx(0.108)
