"""Tests for module 1.

These test the things that could quietly give a wrong answer: the handling of
CRSP's missing-value codes, the de-duplication of the sheet, and the gap
detector. Each test is built on a tiny hand-written frame where the right
answer is obvious by eye, so you can check the test itself in a few seconds.

Run from the fase1 folder:   python -m pytest tests -q
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import module1_data as m1  # noqa: E402


# ---------------------------------------------------------------------------
# coerce_returns
# ---------------------------------------------------------------------------


def test_letter_codes_become_nan_and_are_counted():
    # 'B' and 'C' are WRDS text codes for a missing return, not returns.
    raw = pd.Series(["0.01", "B", "-0.02", "C"])
    out, info = m1.coerce_returns(raw)

    assert out.tolist()[0] == pytest.approx(0.01)
    assert out.tolist()[2] == pytest.approx(-0.02)
    assert out.isna().tolist() == [False, True, False, True]
    assert info["n_letter_codes"] == 2
    assert info["n_missing_codes"] == 0


def test_minus_66_to_minus_99_are_missing_codes_not_returns():
    # This is the dangerous one: -66.0 read as a number is a -6600% return.
    raw = pd.Series([-66.0, -77.0, -88.0, -99.0, -0.66])
    out, info = m1.coerce_returns(raw)

    assert out.isna().tolist() == [True, True, True, True, False]
    assert out.iloc[4] == pytest.approx(-0.66)   # a real -66% return survives
    assert info["n_missing_codes"] == 4
    assert info["n_letter_codes"] == 0


def test_ordinary_returns_are_untouched():
    raw = pd.Series([0.0, 0.05, -0.10, 1.5])
    out, info = m1.coerce_returns(raw)

    assert out.tolist() == pytest.approx([0.0, 0.05, -0.10, 1.5])
    assert info["n_missing_total"] == 0


# ---------------------------------------------------------------------------
# duplicate detection
# ---------------------------------------------------------------------------


def test_duplicate_rows_returns_both_copies_of_a_repeated_key():
    df = pd.DataFrame(
        {
            "permno": [1, 1, 2, 2],
            "date": pd.to_datetime(["2020-01-31", "2020-01-31",
                                    "2020-01-31", "2020-02-29"]),
            "ret": [0.01, 0.01, 0.02, 0.03],
        }
    )
    dups = m1.duplicate_rows(df)

    # PERMNO 1 has January twice -> both rows are flagged; PERMNO 2 is clean.
    assert len(dups) == 2
    assert set(dups["permno"]) == {1}


def test_exact_duplicates_are_dropped_and_counted():
    # PERMNO 1 has January listed three times, identical every time.
    df = pd.DataFrame(
        {
            "permno": [1, 1, 1, 2],
            "date": pd.to_datetime(["2020-01-31"] * 3 + ["2020-01-31"]),
            "ret": [0.01, 0.01, 0.01, 0.02],
        }
    )
    out, info = m1.drop_exact_duplicates(df)

    assert len(out) == 2
    assert info["n_dropped"] == 2
    assert out.loc[out["permno"] == 1, "ret"].iloc[0] == pytest.approx(0.01)


def test_conflicting_duplicates_raise_instead_of_being_guessed():
    # Same key, different return: dropping one would be picking a number.
    df = pd.DataFrame(
        {
            "permno": [1, 1],
            "date": pd.to_datetime(["2020-01-31", "2020-01-31"]),
            "ret": [0.01, 0.09],
        }
    )
    with pytest.raises(ValueError, match="different values"):
        m1.drop_exact_duplicates(df)


def test_no_duplicates_gives_an_empty_frame():
    df = pd.DataFrame(
        {
            "permno": [1, 1, 2],
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-01-31"]),
            "ret": [0.01, 0.02, 0.03],
        }
    )
    assert len(m1.duplicate_rows(df)) == 0


# ---------------------------------------------------------------------------
# universe de-duplication and dual classes
# ---------------------------------------------------------------------------


def _fake_universe():
    """Three firms; PERMNO 46703 listed twice, and one company (PERMCO 9) with
    two share classes."""
    return pd.DataFrame(
        {
            "permno": [46703, 46703, 100, 101],
            "ticker": ["HST", "HST", "GOOG", "GOOGL"],
            "permco": [5, 5, 9, 9],
            "sheet_row": [22, 23, 24, 25],
            "is_duplicate_row": [False, True, False, False],
        }
    )


def test_repeated_permno_is_marked_but_the_row_is_kept():
    uni = _fake_universe()

    # both sheet rows survive, because both have to be filled in on hand-in
    assert len(uni) == 4
    # but only three of them count as firms
    assert (~uni["is_duplicate_row"]).sum() == 3


def test_dual_class_detection_finds_the_two_share_classes():
    dual = m1.dual_class_permcos(_fake_universe())

    assert set(dual["ticker"]) == {"GOOG", "GOOGL"}
    # HST is a repeated *line*, not a second share class, so it is not here
    assert "HST" not in set(dual["ticker"])


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------


def test_coverage_reports_a_firm_that_is_absent_from_the_panel():
    uni = _fake_universe()
    panel = pd.DataFrame({"permno": [46703, 100], "date": pd.to_datetime(
        ["2020-01-31", "2020-01-31"]), "ret": [0.01, 0.02]})

    cov = m1.coverage(panel, uni)

    assert cov["n_firms_universe"] == 3
    assert cov["n_firms_in_panel"] == 2
    assert cov["missing_permnos"] == [101]     # GOOGL never shows up


# ---------------------------------------------------------------------------
# internal gaps
# ---------------------------------------------------------------------------


def test_a_hole_in_the_middle_of_a_history_is_reported():
    # The panel spans four months. PERMNO 1 skips March, PERMNO 2 simply
    # starts late. Only the first is a problem for compounding.
    dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31",
                            "2020-04-30"])
    df = pd.DataFrame(
        {
            "permno": [1, 1, 1, 2, 2],
            "date": [dates[0], dates[1], dates[3], dates[2], dates[3]],
            "ret": [0.01, 0.02, 0.03, 0.04, 0.05],
        }
    )
    gaps = m1.internal_gaps(df)

    assert list(gaps["permno"]) == [1]
    assert gaps.iloc[0]["largest_gap"] == 1     # exactly one month missing


def test_a_short_but_unbroken_history_is_not_a_gap():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    df = pd.DataFrame(
        {
            "permno": [1, 1, 1, 2, 2],
            "date": list(dates) + list(dates[1:]),
            "ret": [0.01, 0.02, 0.03, 0.04, 0.05],
        }
    )
    assert len(m1.internal_gaps(df)) == 0


# ---------------------------------------------------------------------------
# price cross-check
# ---------------------------------------------------------------------------


def test_price_cross_check_flags_a_disagreement_and_a_missing_price():
    uni = pd.DataFrame(
        {
            "permno": [1, 2, 3],
            "ticker": ["AAA", "BBB", "CCC"],
            "price": [10.00, 20.00, 30.00],
            "is_duplicate_row": [False, False, False],
        }
    )
    monthly = pd.DataFrame(
        {
            "permno": [1, 2],
            "date": pd.to_datetime(["2025-12-31", "2025-12-31"]),
            "price": [10.00, 25.00],   # BBB disagrees; CCC is absent
        }
    )
    bad = m1.price_cross_check(uni, monthly)

    assert set(bad["ticker"]) == {"BBB", "CCC"}


def test_price_cross_check_passes_when_everything_matches():
    uni = pd.DataFrame(
        {
            "permno": [1],
            "ticker": ["AAA"],
            "price": [10.00],
            "is_duplicate_row": [False],
        }
    )
    monthly = pd.DataFrame(
        {
            "permno": [1],
            "date": pd.to_datetime(["2025-12-31"]),
            "price": [10.00],
        }
    )
    assert len(m1.price_cross_check(uni, monthly)) == 0


# ---------------------------------------------------------------------------
# look-ahead
# ---------------------------------------------------------------------------


def test_no_lookahead_counts_rows_after_the_trading_date():
    df = pd.DataFrame(
        {"date": pd.to_datetime(["2025-12-30", "2025-12-31", "2026-01-02"])}
    )
    assert m1.no_lookahead(df) == 1
