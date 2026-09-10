"""Tests for module 7.

The rules that must not break: the budget is exactly EUR 10 million, never a
cent more; shares are whole numbers; the leftover cash goes to the risk-free
asset; and the duplicated HST row does not get the position twice.

Run from the fase1 folder:   python -m pytest tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import module7_allocation as m7  # noqa: E402


# ---------------------------------------------------------------------------
# y* = (E[r] - rf) / (A sigma^2)
# ---------------------------------------------------------------------------


def test_optimal_risky_share_follows_the_formula():
    # 8% excess return, 20% volatility, A = 2  ->  0.08 / (2 * 0.04) = 1.0
    out = m7.optimal_risky_share(0.08, 0.20, risk_aversion=2.0)
    assert out["y_raw"] == pytest.approx(1.0)

    # doubling risk aversion halves the risky share
    out4 = m7.optimal_risky_share(0.08, 0.20, risk_aversion=4.0)
    assert out4["y_raw"] == pytest.approx(0.5)


def test_a_more_risk_averse_investor_holds_less_of_the_risky_asset():
    a = m7.optimal_risky_share(0.10, 0.16, 3.0)["y"]
    b = m7.optimal_risky_share(0.10, 0.16, 5.0)["y"]
    assert b < a


def test_borrowing_is_not_allowed_by_default():
    # This investor wants 2.0; without borrowing they get 1.0.
    out = m7.optimal_risky_share(0.16, 0.20, risk_aversion=2.0)
    assert out["y_raw"] == pytest.approx(2.0)
    assert out["y"] == pytest.approx(1.0)
    assert out["binding"] is True


def test_borrowing_can_be_switched_on():
    out = m7.optimal_risky_share(0.16, 0.20, 2.0, allow_borrowing=True)
    assert out["y"] == pytest.approx(2.0)
    assert out["binding"] is False


def test_the_risky_and_riskfree_amounts_add_to_the_budget():
    out = m7.optimal_risky_share(0.08, 0.20, 5.0)
    assert out["risky_amount"] + out["riskfree_amount"] == pytest.approx(1e7)


# ---------------------------------------------------------------------------
# whole shares
# ---------------------------------------------------------------------------


def _simple_book():
    weights = pd.Series({1: 0.5, 2: 0.3, 3: 0.2})
    prices = pd.Series({1: 100.0, 2: 33.0, 3: 7.0})
    return weights, prices


def test_share_counts_are_whole_numbers_and_rounded_down():
    weights, prices = _simple_book()
    table, _ = m7.allocate(weights, prices, y=1.0, budget=1000.0)

    # 500/100 = 5 exactly; 300/33 = 9.09 -> 9; 200/7 = 28.57 -> 28
    assert table.loc[1, "shares"] == 5
    assert table.loc[2, "shares"] == 9
    assert table.loc[3, "shares"] == 28
    assert table["shares"].dtype.kind == "i"


def test_the_budget_is_never_exceeded():
    weights, prices = _simple_book()
    _, info = m7.allocate(weights, prices, y=1.0, budget=1000.0)

    assert info["invested"] <= 1000.0
    assert info["riskfree"] >= 0.0


def test_everything_adds_up_to_the_budget_exactly():
    weights, prices = _simple_book()
    _, info = m7.allocate(weights, prices, y=1.0, budget=1000.0)

    assert info["invested"] + info["riskfree"] == pytest.approx(1000.0)
    assert info["total"] == pytest.approx(1000.0)


def test_a_partial_risky_share_leaves_the_rest_at_the_risk_free_rate():
    weights, prices = _simple_book()
    _, info = m7.allocate(weights, prices, y=0.6, budget=1000.0)

    # 600 is meant for stocks, so at least 400 must be risk-free
    assert info["deliberate_riskfree"] == pytest.approx(400.0)
    assert info["riskfree"] >= 400.0
    assert info["invested"] <= 600.0 + 1e-9


def test_a_position_too_small_to_buy_one_share_becomes_zero():
    weights = pd.Series({1: 0.999, 2: 0.001})
    prices = pd.Series({1: 10.0, 2: 5000.0})       # 0.001 * 1000 = EUR 1
    table, info = m7.allocate(weights, prices, y=1.0, budget=1000.0)

    assert table.loc[2, "shares"] == 0
    assert info["n_dropped_by_rounding"] == 1
    assert info["n_positions"] == 1


def test_a_zero_weight_stays_zero():
    weights = pd.Series({1: 1.0, 2: 0.0})
    prices = pd.Series({1: 10.0, 2: 10.0})
    table, _ = m7.allocate(weights, prices, y=1.0, budget=1000.0)

    assert table.loc[2, "shares"] == 0


# ---------------------------------------------------------------------------
# statistics of the rounded portfolio
# ---------------------------------------------------------------------------


def test_statistics_use_the_shares_actually_held():
    # Two stocks, equal weights after rounding, uncorrelated, 20% vol each.
    table = pd.DataFrame({
        "shares": [10, 10],
        "value": [500.0, 500.0],
        "weight_target": [0.5, 0.5],
        "weight_actual": [0.5, 0.5],
    }, index=[1, 2])
    mu = pd.Series({1: 0.10, 2: 0.06})
    sigma = pd.DataFrame([[0.04, 0.0], [0.0, 0.04]], index=[1, 2], columns=[1, 2])

    st = m7.realised_portfolio_stats(table, mu, sigma, budget=1000.0)

    assert st["y_actual"] == pytest.approx(1.0)
    assert st["risky_excess_return"] == pytest.approx(0.08)
    assert st["risky_volatility"] == pytest.approx(np.sqrt(0.02))


def test_a_partly_invested_portfolio_scales_return_and_risk_together():
    table = pd.DataFrame({
        "shares": [10], "value": [500.0],
        "weight_target": [1.0], "weight_actual": [0.5],
    }, index=[1])
    mu = pd.Series({1: 0.10})
    sigma = pd.DataFrame([[0.04]], index=[1], columns=[1])

    st = m7.realised_portfolio_stats(table, mu, sigma, budget=1000.0)

    assert st["y_actual"] == pytest.approx(0.5)
    assert st["total_excess_return"] == pytest.approx(0.05)   # half of 0.10
    assert st["total_volatility"] == pytest.approx(0.10)      # half of 0.20


# ---------------------------------------------------------------------------
# writing the sheet
# ---------------------------------------------------------------------------


def _fake_sheet(path: Path):
    """A miniature hand-in sheet: header row 20, risk-free row 21, firms 22-24
    with the third row repeating the second firm, as HST does."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws["A20"], ws["B20"] = "PERMNO", "Ticker Symbol"
    ws["I16"] = "=SUM(I22:I23)"          # deliberately two rows short
    for row, (permno, ticker, price) in enumerate(
            [(10, "AAA", 100.0), (20, "BBB", 50.0), (20, "BBB", 50.0)], start=22):
        ws[f"A{row}"], ws[f"B{row}"], ws[f"F{row}"] = permno, ticker, price
    wb.save(path)


def _fake_universe():
    return pd.DataFrame({
        "permno": [10, 20, 20],
        "ticker": ["AAA", "BBB", "BBB"],
        "price": [100.0, 50.0, 50.0],
        "sheet_row": [22, 23, 24],
        "is_duplicate_row": [False, False, True],
    })


def test_shares_are_written_once_and_the_repeated_row_gets_zero(tmp_path,
                                                                monkeypatch):
    monkeypatch.setattr(m7, "SHEET_FIRST_FIRM_ROW", 22)
    monkeypatch.setattr(m7, "SHEET_LAST_FIRM_ROW", 24)
    monkeypatch.setattr(m7, "SHEET_RF_ROW", 21)

    src, dst = tmp_path / "in.xlsx", tmp_path / "out.xlsx"
    _fake_sheet(src)
    table = pd.DataFrame({"shares": [7, 3]}, index=[10, 20])

    info = m7.fill_hand_in_sheet(table, _fake_universe(), 1234.5, src, dst)

    import openpyxl
    ws = openpyxl.load_workbook(dst).active
    assert ws["H22"].value == 7
    assert ws["H23"].value == 3
    assert ws["H24"].value == 0          # the repeat, not another 3
    assert info["rows_with_shares"] == 2


def test_the_risk_free_amount_goes_in_its_own_row(tmp_path, monkeypatch):
    monkeypatch.setattr(m7, "SHEET_FIRST_FIRM_ROW", 22)
    monkeypatch.setattr(m7, "SHEET_LAST_FIRM_ROW", 24)
    monkeypatch.setattr(m7, "SHEET_RF_ROW", 21)

    src, dst = tmp_path / "in.xlsx", tmp_path / "out.xlsx"
    _fake_sheet(src)
    m7.fill_hand_in_sheet(pd.DataFrame({"shares": [7, 3]}, index=[10, 20]),
                          _fake_universe(), 1234.5, src, dst)

    import openpyxl
    assert openpyxl.load_workbook(dst).active["H21"].value == pytest.approx(1234.5)


def test_the_short_subtotal_formula_is_repaired(tmp_path, monkeypatch):
    monkeypatch.setattr(m7, "SHEET_FIRST_FIRM_ROW", 22)
    monkeypatch.setattr(m7, "SHEET_LAST_FIRM_ROW", 24)
    monkeypatch.setattr(m7, "SHEET_RF_ROW", 21)

    src, dst = tmp_path / "in.xlsx", tmp_path / "out.xlsx"
    _fake_sheet(src)
    info = m7.fill_hand_in_sheet(pd.DataFrame({"shares": [7, 3]}, index=[10, 20]),
                                 _fake_universe(), 0.0, src, dst)

    assert info["old_subtotal_formulas"]["I16"] == "=SUM(I22:I23)"
    assert info["new_subtotal_formulas"]["I16"] == "=SUM(I22:I24)"
    # parts II and III use the same sheet, so their subtotals are fixed too
    assert info["new_subtotal_formulas"]["L16"] == "=SUM(L22:L24)"
    assert info["new_subtotal_formulas"]["O16"] == "=SUM(O22:O24)"
