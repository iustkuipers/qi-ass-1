"""Step 7 - the capital allocation and the completed hand-in sheet.

Run from the fase1 folder:   python scripts/07_allocation.py

Produces
    data/output/allocation.csv              every firm, shares and weight
    data/output/Hand_in_sheet_part1.xlsx    the sheet to submit
    output/07_allocation.txt                the report printed below
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (ALLOW_BORROWING, BUDGET, FASE1, HAND_IN_SHEET,  # noqa: E402
                    OUTPUT, PROCESSED, RF_ANNUAL, RISK_AVERSION,
                    RISK_AVERSION_ALTERNATIVES, SHEET_SHARES_COL_PART1)
from modules import module7_allocation as m7  # noqa: E402

# Fill these in before the final submission; they go into the sheet header.
GROUP: dict = {
    # "group": "", "student1": "", "student1_nr": "",
    # "student2": "", "student2_nr": "", "student3": "", "student3_nr": "",
}

OUT_DIR = FASE1 / "data" / "output"


def main() -> None:
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say("=" * 78)
    say("STEP 7 - CAPITAL ALLOCATION AND HAND-IN SHEET")
    say("=" * 78)

    weights_all = pd.read_csv(PROCESSED / "tangency_weights.csv", index_col="permno")
    sigma = pd.read_csv(PROCESSED / "covariance_single_index.csv", index_col=0)
    sigma.columns = sigma.columns.astype(int)
    universe = pd.read_csv(PROCESSED / "universe.csv")
    mrp = pd.read_csv(PROCESSED / "mrp.csv", index_col=0)["0"]

    weights = weights_all["main_single_index"].dropna()
    mu = weights_all["mu_excess"].reindex(weights.index)
    prices = universe.drop_duplicates("permno").set_index("permno")["price"]

    excess = float((weights * mu).sum())
    vol = float(np.sqrt(weights.to_numpy()
                        @ sigma.loc[weights.index, weights.index].to_numpy()
                        @ weights.to_numpy()))

    say()
    say("THE RISKY PORTFOLIO (from step 6, single-index, long-only)")
    say(f"  stocks                   : {len(weights)}")
    say(f"  expected excess return   : {excess:.2%}")
    say(f"  volatility               : {vol:.2%}")
    say(f"  Sharpe ratio             : {excess / vol:.3f}")
    if str(mrp.get("placeholder")).lower() == "true":
        say()
        say("  !! The market risk premium is still the 2006-2025 placeholder")
        say(f"     ({float(mrp['mrp_annual']):.2%} a year). famafrench_long.csv is not in")
        say("     data/raw. This does not affect the weights, but it does move")
        say("     y* below: y* is proportional to the expected excess return, so")
        say("     a long-run premium of, say, 8% instead of 10.15% scales every")
        say("     y* in the table below by 8/10.15.")

    # --- how much of the budget goes into it ------------------------------
    say()
    say("=" * 78)
    say("HOW MUCH TO INVEST:  y* = (E[r] - rf) / (A sigma^2)")
    say("=" * 78)
    say()
    say(f"      {'A':>4} {'y* raw':>9} {'y* used':>9} {'risky EUR':>14} "
        f"{'risk-free EUR':>15} {'E[r]':>8} {'vol':>8}")

    chosen = None
    for A in sorted({RISK_AVERSION, *RISK_AVERSION_ALTERNATIVES}):
        split = m7.optimal_risky_share(excess, vol, A, ALLOW_BORROWING)
        mark = "   <- chosen" if A == RISK_AVERSION else ""
        say(f"      {A:>4.0f} {split['y_raw']:>9.3f} {split['y']:>9.3f} "
            f"{split['risky_amount']:>14,.0f} {split['riskfree_amount']:>15,.0f} "
            f"{split['y'] * excess + RF_ANNUAL:>8.2%} "
            f"{split['y'] * vol:>8.2%}{mark}")
        if A == RISK_AVERSION:
            chosen = split

    say()
    if chosen["binding"]:
        say(f"  At A = {RISK_AVERSION:.0f} the unconstrained y* is "
            f"{chosen['y_raw']:.3f}, so the no-borrowing rule binds and we hold")
        say("  the risky portfolio with the whole budget. Anything left at the")
        say("  risk-free rate is then only the cash that whole shares leave behind.")
    else:
        say(f"  At A = {RISK_AVERSION:.0f} the investor wants "
            f"{chosen['y']:.1%} in the risky portfolio and "
            f"{1 - chosen['y']:.1%} at the risk-free rate.")

    # --- whole shares ------------------------------------------------------
    table, info = m7.allocate(weights, prices, chosen["y"])
    stats = m7.realised_portfolio_stats(table, mu, sigma)

    say()
    say("=" * 78)
    say("ROUNDING TO WHOLE SHARES")
    say("=" * 78)
    say(f"  positions with at least one share : {info['n_positions']}")
    say(f"  positions rounded down to zero    : {info['n_dropped_by_rounding']}")
    say(f"  invested in stocks                : EUR {info['invested']:>14,.2f}")
    say(f"  deliberate risk-free holding      : EUR "
        f"{info['deliberate_riskfree']:>14,.2f}")
    say(f"  cash left over by rounding        : EUR "
        f"{info['rounding_cash']:>14,.2f}")
    say(f"  total risk-free                   : EUR {info['riskfree']:>14,.2f}")
    say(f"  TOTAL                             : EUR {info['total']:>14,.2f}")
    say(f"  budget                            : EUR {BUDGET:>14,.2f}")
    say(f"  difference                        : EUR "
        f"{info['total'] - BUDGET:>14,.2f}")
    assert abs(info["total"] - BUDGET) < 1e-6, "the budget does not add up"
    say(f"  largest single position           : EUR "
        f"{info['largest_position']:>14,.2f} "
        f"({info['largest_position'] / BUDGET:.3%})")
    say(f"  smallest position held            : EUR "
        f"{info['smallest_position']:>14,.2f}")

    say()
    say("  Positions are rounded DOWN, never to nearest: rounding to nearest can")
    say("  spend more than EUR 10 million, and the budget is fixed. The cash that")
    say("  rounding leaves behind is not wasted - it earns the 1% risk-free rate.")

    say()
    say("THE PORTFOLIO WE ACTUALLY HOLD (after rounding)")
    say(f"  share of budget in stocks : {stats['y_actual']:.4f}")
    say(f"  risky-book excess return  : {stats['risky_excess_return']:.2%}")
    say(f"  risky-book volatility     : {stats['risky_volatility']:.2%}")
    say(f"  risky-book Sharpe         : {stats['risky_sharpe']:.3f}")
    say(f"  total expected return     : {stats['total_expected_return']:.2%}")
    say(f"  total volatility          : {stats['total_volatility']:.2%}")
    say(f"  largest drift from target : {stats['tracking_to_target']:.4%} "
        f"of the budget")

    # --- the biggest and smallest holdings --------------------------------
    say()
    say("LARGEST POSITIONS")
    tick = universe.drop_duplicates("permno").set_index("permno")["ticker"]
    top = table.sort_values("value", ascending=False).head(15)
    say(f"      {'ticker':<8} {'shares':>9} {'price':>9} {'value EUR':>13} "
        f"{'weight':>8}")
    for permno, r in top.iterrows():
        say(f"      {str(tick.get(permno, permno)):<8} {int(r['shares']):>9,} "
            f"{r['price']:>9.2f} {r['value']:>13,.2f} {r['weight_actual']:>8.3%}")

    dropped = table[(table["target_value"] > 0) & (table["shares"] == 0)]
    if len(dropped):
        say()
        say(f"  rounded to zero ({len(dropped)}): "
            f"{', '.join(str(tick.get(p, p)) for p in dropped.index[:12])}")

    # --- the sheet ---------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "Hand_in_sheet_part1.xlsx"
    sheet_info = m7.fill_hand_in_sheet(
        table, universe, info["riskfree"], HAND_IN_SHEET, out_path,
        shares_column=SHEET_SHARES_COL_PART1, group=GROUP)

    say()
    say("=" * 78)
    say("HAND-IN SHEET")
    say("=" * 78)
    say(f"  written to               : {sheet_info['path']}")
    say(f"  column                   : {SHEET_SHARES_COL_PART1} "
        f"(shares held, Part I)")
    say(f"  firm rows with a holding : {sheet_info['rows_with_shares']}")
    say(f"  risk-free cell           : {sheet_info['riskfree_cell']} = "
        f"EUR {info['riskfree']:,.2f}  (its price is 1, so the cell is the amount)")
    say(f"  HST duplicate row        : shares on the first line, 0 on the second")
    say()
    say("  The template's subtotal formulas stop two rows short of the firm")
    say("  block, silently omitting ZBRA and ZTS. All three parts share this")
    say("  sheet, so all three are repaired:")
    for cell in ["I16", "L16", "O16"]:
        say(f"      {cell}: {sheet_info['old_subtotal_formulas'][cell]}"
            f"  ->  {sheet_info['new_subtotal_formulas'][cell]}")

    # --- full allocation ---------------------------------------------------
    allocation = universe.merge(
        table.reset_index().rename(columns={"index": "permno"}),
        on="permno", how="left")
    allocation["shares"] = allocation["shares"].fillna(0).astype("int64")
    allocation.loc[allocation["is_duplicate_row"], "shares"] = 0
    allocation["value"] = allocation["shares"] * allocation["price_x"].fillna(
        allocation.get("price_y", 0))
    allocation["weight"] = allocation["value"] / BUDGET
    allocation.to_csv(OUT_DIR / "allocation.csv", index=False)

    say()
    say("WRITTEN")
    say(f"  {OUT_DIR / 'allocation.csv'}")
    say(f"  {out_path}")

    report = OUTPUT / "07_allocation.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport: {report}")


if __name__ == "__main__":
    main()
