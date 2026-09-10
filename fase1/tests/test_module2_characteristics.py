"""Tests for module 2.

The two things that can silently produce a wrong answer here are the book
equity fallback ladder and the Fama-French timing. Both are tested on tiny
frames where the right answer can be worked out on paper in a few seconds.

Run from the fase1 folder:   python -m pytest tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import module2_characteristics as m2  # noqa: E402


def _accounting_row(**kwargs) -> pd.DataFrame:
    """One Compustat row, with every field missing unless given."""
    fields = ["seq", "ceq", "at", "lt", "pstkrv", "pstkl", "pstk",
              "txditc", "txdb", "itcb", "sich"]
    row = {f: np.nan for f in fields}
    row.update(kwargs)
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# book equity: BE = stockholders' equity + deferred taxes - preferred stock
# ---------------------------------------------------------------------------


def test_book_equity_uses_seq_first():
    # 100 + 10 - 5 = 105. CEQ and AT-LT are present but must be ignored.
    df = _accounting_row(seq=100, ceq=80, at=500, lt=300,
                         txditc=10, pstkrv=5)
    assert m2.book_equity(df)["be"].iloc[0] == pytest.approx(105.0)


def test_book_equity_falls_back_to_ceq_plus_pstk_when_seq_missing():
    # SEQ missing -> 80 + 5 (=CEQ+PSTK) + 10 - 5 = 90
    df = _accounting_row(ceq=80, pstk=5, at=500, lt=300, txditc=10)
    assert m2.book_equity(df)["be"].iloc[0] == pytest.approx(90.0)


def test_book_equity_falls_back_to_assets_minus_liabilities_last():
    # SEQ and CEQ missing -> 500 - 300 = 200, plus 10, minus 0
    df = _accounting_row(at=500, lt=300, txditc=10)
    assert m2.book_equity(df)["be"].iloc[0] == pytest.approx(210.0)


def test_book_equity_is_missing_when_there_is_no_equity_number_at_all():
    df = _accounting_row(txditc=10, pstk=5)
    assert np.isnan(m2.book_equity(df)["be"].iloc[0])


def test_deferred_tax_falls_back_to_txdb_plus_itcb():
    # TXDITC missing -> 7 + 3 = 10 deferred taxes; 100 + 10 - 0 = 110
    df = _accounting_row(seq=100, txdb=7, itcb=3)
    assert m2.book_equity(df)["be"].iloc[0] == pytest.approx(110.0)


def test_deferred_tax_is_zero_when_nothing_is_reported():
    df = _accounting_row(seq=100)
    assert m2.book_equity(df)["be"].iloc[0] == pytest.approx(100.0)


def test_preferred_stock_prefers_redemption_then_liquidating_then_carrying():
    # all three present -> PSTKRV (=5) wins:  100 + 0 - 5 = 95
    both = _accounting_row(seq=100, pstkrv=5, pstkl=6, pstk=7)
    assert m2.book_equity(both)["be"].iloc[0] == pytest.approx(95.0)

    # PSTKRV missing -> PSTKL (=6):  100 - 6 = 94
    liq = _accounting_row(seq=100, pstkl=6, pstk=7)
    assert m2.book_equity(liq)["be"].iloc[0] == pytest.approx(94.0)

    # only carrying value -> PSTK (=7):  100 - 7 = 93
    carry = _accounting_row(seq=100, pstk=7)
    assert m2.book_equity(carry)["be"].iloc[0] == pytest.approx(93.0)


def test_negative_book_equity_is_flagged():
    # A firm that bought back so much stock that book equity went negative.
    df = _accounting_row(seq=-50, pstk=10)
    out = m2.book_equity(df)
    assert out["be"].iloc[0] == pytest.approx(-60.0)
    assert bool(out["neg_be"].iloc[0]) is True


def test_financial_firms_are_flagged_on_historical_sic():
    banks = m2.book_equity(_accounting_row(seq=100, sich=6021))
    software = m2.book_equity(_accounting_row(seq=100, sich=7372))
    assert bool(banks["is_financial"].iloc[0]) is True
    assert bool(software["is_financial"].iloc[0]) is False


# ---------------------------------------------------------------------------
# market equity
# ---------------------------------------------------------------------------


def test_market_equity_converts_thousands_to_millions():
    monthly = pd.DataFrame(
        {
            "permno": [1],
            "permco": [10],
            "date": pd.to_datetime(["2020-06-30"]),
            "mktcap": [2_500_000.0],       # CRSP thousands
        }
    )
    out = m2.market_equity(monthly, month=6)
    assert out["me"].iloc[0] == pytest.approx(2_500.0)   # millions


def test_company_market_equity_sums_the_share_classes():
    # Two classes of one company (PERMCO 10) plus an unrelated firm.
    monthly = pd.DataFrame(
        {
            "permno": [1, 2, 3],
            "permco": [10, 10, 20],
            "date": pd.to_datetime(["2020-06-30"] * 3),
            "mktcap": [1_000_000.0, 3_000_000.0, 5_000_000.0],
        }
    )
    out = m2.market_equity(monthly, month=6).set_index("permno")

    # each class keeps its own ME ...
    assert out.loc[1, "me"] == pytest.approx(1_000.0)
    assert out.loc[2, "me"] == pytest.approx(3_000.0)
    # ... but both see the company total, which is what B/M is divided by
    assert out.loc[1, "me_firm"] == pytest.approx(4_000.0)
    assert out.loc[2, "me_firm"] == pytest.approx(4_000.0)
    assert out.loc[3, "me_firm"] == pytest.approx(5_000.0)


def test_a_company_with_no_price_that_month_gets_no_market_equity_not_zero():
    # Super Micro (PERMNO 91907) had no December price in 2018 and 2019 while
    # it was off Nasdaq. A plain sum of an all-missing group returns 0.0, and
    # a market equity of zero makes B/M infinite instead of missing.
    monthly = pd.DataFrame(
        {
            "permno": [1],
            "permco": [10],
            "date": pd.to_datetime(["2019-12-31"]),
            "mktcap": [np.nan],
        }
    )
    out = m2.market_equity(monthly, month=12)

    assert np.isnan(out["me"].iloc[0])
    assert np.isnan(out["me_firm"].iloc[0])      # not 0.0


def test_bm_is_missing_rather_than_infinite_when_market_equity_is_missing():
    compustat, monthly = _timing_fixture()
    monthly.loc[monthly["date"] == pd.Timestamp("2020-12-31"), "mktcap"] = np.nan

    out = m2.build_characteristics(compustat, monthly)
    row = out[out["ff_year"] == 2021].iloc[0]

    assert np.isnan(row["bm"])
    assert not np.isinf(row["bm"])


def test_market_equity_only_keeps_the_requested_month():
    monthly = pd.DataFrame(
        {
            "permno": [1, 1],
            "permco": [10, 10],
            "date": pd.to_datetime(["2020-06-30", "2020-12-31"]),
            "mktcap": [1_000_000.0, 2_000_000.0],
        }
    )
    june = m2.market_equity(monthly, month=6)
    december = m2.market_equity(monthly, month=12)

    assert len(june) == 1 and june["me"].iloc[0] == pytest.approx(1_000.0)
    assert len(december) == 1 and december["me"].iloc[0] == pytest.approx(2_000.0)


# ---------------------------------------------------------------------------
# the Fama-French timing - the test that matters most
# ---------------------------------------------------------------------------


def _timing_fixture():
    """One firm with two fiscal years and three years of market equity.

    For FF year 2021 (July 2021 - June 2022) the right answer is:
        size = ME June 2021        = 300
        BE   = fiscal year ending in 2020 = 50
        B/M  = 50 / ME Dec 2020    = 50 / 200 = 0.25
    Everything dated 2021 in the accounting data must stay unused, because in
    June 2021 a fiscal year ending in December 2021 has not happened yet.
    """
    compustat = pd.DataFrame(
        {
            "permno": [1, 1],
            "datadate": pd.to_datetime(["2020-12-31", "2021-12-31"]),
            "fyear": [2020, 2021],
            "conm": ["TEST CO", "TEST CO"],
            "seq": [50.0, 999.0],          # 999 is the number that must NOT be used
            "ceq": [np.nan, np.nan],
            "at": [np.nan, np.nan],
            "lt": [np.nan, np.nan],
            "pstkrv": [np.nan, np.nan],
            "pstkl": [np.nan, np.nan],
            "pstk": [np.nan, np.nan],
            "txditc": [0.0, 0.0],
            "txdb": [np.nan, np.nan],
            "itcb": [np.nan, np.nan],
            "sich": [7372.0, 7372.0],
        }
    )
    monthly = pd.DataFrame(
        {
            "permno": [1, 1, 1, 1],
            "permco": [10, 10, 10, 10],
            "date": pd.to_datetime(["2020-06-30", "2020-12-31",
                                    "2021-06-30", "2021-12-31"]),
            # thousands -> 100, 200, 300, 400 million
            "mktcap": [100_000.0, 200_000.0, 300_000.0, 400_000.0],
        }
    )
    return compustat, monthly


def test_ff_year_uses_june_me_of_year_t():
    compustat, monthly = _timing_fixture()
    out = m2.build_characteristics(compustat, monthly)
    row = out[out["ff_year"] == 2021].iloc[0]

    assert row["me_june"] == pytest.approx(300.0)      # June 2021, not June 2020


def test_ff_year_uses_book_equity_of_the_fiscal_year_ending_in_t_minus_1():
    compustat, monthly = _timing_fixture()
    out = m2.build_characteristics(compustat, monthly)
    row = out[out["ff_year"] == 2021].iloc[0]

    assert row["be"] == pytest.approx(50.0)            # FY2020, not FY2021
    assert row["datadate"] == pd.Timestamp("2020-12-31")


def test_ff_year_divides_by_december_me_of_t_minus_1():
    compustat, monthly = _timing_fixture()
    out = m2.build_characteristics(compustat, monthly)
    row = out[out["ff_year"] == 2021].iloc[0]

    # 50 / 200, not 50 / 400 (December 2021 is in the future in June 2021)
    assert row["bm"] == pytest.approx(0.25)


def test_bm_is_missing_when_book_equity_is_negative():
    compustat, monthly = _timing_fixture()
    compustat.loc[compustat["fyear"] == 2020, "seq"] = -50.0
    out = m2.build_characteristics(compustat, monthly)
    row = out[out["ff_year"] == 2021].iloc[0]

    assert row["be"] == pytest.approx(-50.0)
    assert np.isnan(row["bm"])
    assert bool(row["neg_be"]) is True


def test_a_second_fiscal_year_ending_in_the_same_calendar_year_keeps_the_later_one():
    # A firm that changes its fiscal year end can report twice in one calendar
    # year. In June of the next year the later one is the current information.
    compustat, monthly = _timing_fixture()
    extra = compustat.iloc[[0]].copy()
    extra["datadate"] = pd.Timestamp("2020-03-31")
    extra["seq"] = 11.0                     # older, must lose
    compustat = pd.concat([extra, compustat], ignore_index=True)

    out = m2.build_characteristics(compustat, monthly)
    row = out[out["ff_year"] == 2021].iloc[0]

    assert row["be"] == pytest.approx(50.0)
    assert row["datadate"] == pd.Timestamp("2020-12-31")
