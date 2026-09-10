"""Step 6 - covariance matrices and tangency portfolios.

Run from the fase1 folder:   python scripts/06_portfolio.py

The main portfolio uses a single-index covariance matrix built on the same
Fama-French post-ranking betas that the expected returns use. The reason is in
the consistency check at the top of the report: with a Ledoit-Wolf covariance
estimated stock by stock, the optimiser bought stocks whose *assigned* beta was
high but whose *actual* market exposure was low, and the resulting Sharpe ratio
was mostly a measure of that inconsistency. The Ledoit-Wolf portfolio is kept
as a robustness case.

Produces
    data/processed/covariance_single_index.csv
    data/processed/covariance_ledoit_wolf.csv
    data/processed/tangency_weights.csv
    output/06_portfolio.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (COV_YEARS, OUTPUT, PENDING_DEALS, PROCESSED,  # noqa: E402
                    RF_ANNUAL, TRADE_DATE, TRADING_DAYS_PER_YEAR, WEIGHT_CAP)
from modules import module6_portfolio as m6  # noqa: E402


def main() -> None:
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say("=" * 78)
    say("STEP 6 - COVARIANCE AND TANGENCY PORTFOLIOS")
    say("=" * 78)

    daily = pd.read_parquet(PROCESSED / "daily.parquet")
    market = pd.read_parquet(PROCESSED / "market_daily.parquet").set_index("date")["vwretd"]
    mu_all = pd.read_csv(PROCESSED / "expected_returns.csv", index_col="permno")
    universe = pd.read_csv(PROCESSED / "universe.csv")
    tickers = universe.drop_duplicates("permno").set_index("permno")["ticker"]

    start = TRADE_DATE - pd.DateOffset(years=COV_YEARS) + pd.Timedelta(days=1)

    # --- takeover screen --------------------------------------------------
    say()
    say("=" * 78)
    say("TAKEOVER SCREEN")
    say("=" * 78)
    say("  A stock under an agreed all-cash bid is no longer equity: its price")
    say("  is pinned to the offer, so its past volatility understates nothing -")
    say("  it simply describes a different asset. Left in, the optimiser sees a")
    say("  near-riskless stock and loads up on it.")
    say()
    screen = m6.volatility_screen(
        daily,
        recent_start=pd.Timestamp("2025-10-01"), recent_end=TRADE_DATE,
        prior_start=pd.Timestamp("2025-01-01"), prior_end=pd.Timestamp("2025-09-30"))
    screen["ticker"] = [tickers.get(p, p) for p in screen.index]

    say(f"  ratio of Q4-2025 volatility to Jan-Sep 2025 volatility, "
        f"{len(screen)} stocks screened")
    say(f"  median ratio {screen['ratio'].median():.2f}, "
        f"5th percentile {screen['ratio'].quantile(0.05):.2f}")
    say()
    say(f"      {'ticker':<8} {'Jan-Sep vol':>12} {'Q4 vol':>9} {'ratio':>7} "
        f"{'Q4 return':>11}")
    for permno, r in screen.head(8).iterrows():
        mark = "  <- known deal" if permno in PENDING_DEALS else ""
        say(f"      {str(r['ticker']):<8} {r['prior_vol']:>12.1%} "
            f"{r['recent_vol']:>9.1%} {r['ratio']:>7.2f} "
            f"{r['recent_mean']:>11.1%}{mark}")
    say()
    say("  The three deals the group identified are the three lowest ratios and")
    say("  the screen finds no fourth: after HOLX at 0.42 the next names have")
    say("  Q4 volatilities of 30-40%, which is a quiet quarter, not a bid.")
    say(f"  Excluded: "
        f"{', '.join(str(tickers.get(p, p)) for p in PENDING_DEALS)}")

    # --- return matrix ----------------------------------------------------
    candidates = [p for p in mu_all.index if p not in PENDING_DEALS]
    returns, rinfo = m6.daily_return_matrix(daily, start, TRADE_DATE, candidates)

    say()
    say(f"DAILY RETURN WINDOW ({COV_YEARS} years to the trading date)")
    say(f"  {rinfo['start']:%Y-%m-%d} to {rinfo['end']:%Y-%m-%d}, "
        f"{rinfo['n_days']} days used of {rinfo['n_days_raw']}")
    say(f"  investable after the takeover screen : {rinfo['n_candidates']}")
    say(f"  kept (>= {rinfo['min_coverage']:.0%} of days)          : {rinfo['n_kept']}")
    for permno, n in sorted(rinfo["excluded"].items(), key=lambda kv: kv[1]):
        say(f"      excluded: {str(tickers.get(permno, permno)):6s} {n} of "
            f"{rinfo['n_days_raw']} days")

    assets = list(returns.columns)
    mu = mu_all.loc[assets, "mu_excess"]
    betas = mu_all.loc[assets, "beta"]
    mkt = market.reindex(returns.index)
    market_var = float(mkt.var(ddof=1) * TRADING_DAYS_PER_YEAR)

    # --- the consistency check that drives the choice ---------------------
    sigma_lw, cinfo = m6.ledoit_wolf_covariance(returns)
    w_lw, _ = m6.tangency_long_only(mu, sigma_lw, WEIGHT_CAP)
    lw_realised = m6.realised_beta((returns * w_lw).sum(axis=1), mkt)

    say()
    say("=" * 78)
    say("CONSISTENCY CHECK: does the portfolio have the beta its mu claims?")
    say("=" * 78)
    say(f"  market volatility over the window : {np.sqrt(market_var):.2%}")
    say()
    say(f"  Ledoit-Wolf long-only portfolio")
    say(f"      assigned beta (weighted average of the FF betas) : "
        f"{float((w_lw * betas).sum()):.3f}")
    say(f"      realised beta (regression on the market)         : "
        f"{lw_realised['beta']:.3f}  (t = {lw_realised['t']:.1f}, "
        f"R2 = {lw_realised['r2']:.2f})")
    say()
    say("  The two disagree by a factor of two. mu says this portfolio carries")
    say("  1.09 units of market risk and should be paid for them; its own")
    say("  returns say it carries 0.53. The optimiser found stocks sitting in a")
    say("  high-beta 5x5 cell whose individual covariance with the market is")
    say("  low, and bought the discrepancy. That is an artefact of using coarse")
    say("  portfolio betas in mu and fine stock-level covariances in Sigma, and")
    say("  it is why the Ledoit-Wolf Sharpe ratio cannot be believed.")

    # --- the single-index covariance --------------------------------------
    resid_var, vinfo = m6.residual_variances(returns, mkt, betas)
    sigma_si = m6.single_index_covariance(betas, resid_var, market_var)

    say()
    say("=" * 78)
    say("SINGLE-INDEX COVARIANCE:  Sigma = sigma_m^2 beta beta' + D")
    say("=" * 78)
    say(f"  betas                    : the same FF post-ranking betas as in mu")
    say(f"  market variance          : {market_var:.5f} "
        f"(volatility {np.sqrt(market_var):.2%})")
    say(f"  residual volatility      : mean {vinfo['mean_resid_vol']:.2%}, "
        f"min {vinfo['min_resid_vol']:.2%}, max {vinfo['max_resid_vol']:.2%}")
    say(f"  condition number         : "
        f"{np.linalg.cond(sigma_si.to_numpy()):,.0f}  "
        f"(Ledoit-Wolf: {cinfo['cond_shrunk']:,.0f})")

    w_si_unc = m6.tangency_unconstrained(mu, sigma_si)
    w_si, siinfo = m6.tangency_long_only(mu, sigma_si, WEIGHT_CAP)

    # the closed form the Sherman-Morrison identity predicts
    predicted = betas / resid_var
    predicted = predicted / predicted.sum()
    corr = float(np.corrcoef(w_si_unc.to_numpy(), predicted.to_numpy())[0, 1])
    max_dev = float((w_si_unc - predicted).abs().max())

    say()
    say("  CHECK: the unconstrained weights should be exactly proportional to")
    say("  beta_i / residual variance_i (Sherman-Morrison on the rank-one term).")
    say(f"      correlation with beta_i / resvar_i : {corr:.10f}")
    say(f"      largest absolute deviation         : {max_dev:.2e}")
    say(f"      -> they are the same portfolio, so the optimiser is only")
    say(f"         trading each stock's beta against its own risk.")

    say()
    say(f"      {'ticker':<8} {'beta':>7} {'resid vol':>10} {'b/resvar':>10} "
        f"{'weight':>9}")
    for permno in predicted.sort_values(ascending=False).head(8).index:
        say(f"      {str(tickers.get(permno, permno)):<8} {betas[permno]:>7.3f} "
            f"{np.sqrt(resid_var[permno]):>10.2%} "
            f"{betas[permno] / resid_var[permno]:>10.2f} "
            f"{w_si_unc[permno]:>9.3%}")

    si_realised = m6.realised_beta((returns * w_si).sum(axis=1), mkt)

    # --- the comparison ---------------------------------------------------
    stats_si = m6.portfolio_stats(w_si, mu, sigma_si, RF_ANNUAL)
    stats_si_unc = m6.portfolio_stats(w_si_unc, mu, sigma_si, RF_ANNUAL)
    stats_lw = m6.portfolio_stats(w_lw, mu, sigma_lw, RF_ANNUAL)

    realised_vol_si = float((returns * w_si).sum(axis=1).std(ddof=1)
                            * np.sqrt(TRADING_DAYS_PER_YEAR))
    realised_vol_lw = float((returns * w_lw).sum(axis=1).std(ddof=1)
                            * np.sqrt(TRADING_DAYS_PER_YEAR))

    say()
    say("=" * 78)
    say("THE PORTFOLIOS")
    say("=" * 78)
    say()
    say(f"  {'':<26} {'MAIN':>14} {'single-index':>14} {'robustness':>14}")
    say(f"  {'':<26} {'single-index':>14} {'unconstrained':>14} "
        f"{'Ledoit-Wolf':>14}")
    say(f"  {'':<26} {'long-only 5%':>14} {'':>14} {'long-only 5%':>14}")
    say("  " + "-" * 72)
    for label, key, fmt in [
        ("expected excess return", "excess_return", "{:>14.2%}"),
        ("model volatility", "volatility", "{:>14.2%}"),
        ("model Sharpe ratio", "sharpe", "{:>14.3f}"),
        ("positions held", "n_long", "{:>14.0f}"),
        ("largest weight", "max_weight", "{:>14.2%}"),
        ("effective N", "effective_n", "{:>14.1f}"),
        ("gross exposure", "gross_exposure", "{:>14.2f}"),
    ]:
        say(f"  {label:<26} {fmt.format(stats_si[key])} "
            f"{fmt.format(stats_si_unc[key])} {fmt.format(stats_lw[key])}")

    say(f"  {'realised volatility':<26} {realised_vol_si:>14.2%} "
        f"{'':>14} {realised_vol_lw:>14.2%}")
    say(f"  {'assigned beta':<26} {float((w_si * betas).sum()):>14.3f} "
        f"{float((w_si_unc * betas).sum()):>14.3f} "
        f"{float((w_lw * betas).sum()):>14.3f}")
    say(f"  {'realised beta':<26} {si_realised['beta']:>14.3f} "
        f"{'':>14} {lw_realised['beta']:>14.3f}")

    si_gap = abs(float((w_si * betas).sum()) - si_realised["beta"])
    lw_gap = abs(float((w_lw * betas).sum()) - lw_realised["beta"])
    say()
    say(f"  Assigned minus realised beta: {si_gap:.2f} for the main portfolio,")
    say(f"  {lw_gap:.2f} for the Ledoit-Wolf one. That gap, not the volatility,")
    say("  is what separates them - both matrices predict their own portfolio's")
    say("  volatility perfectly well, because both were fitted on these same 750")
    say("  days. The difference is that the Ledoit-Wolf portfolio is *paid* for")
    say("  1.09 units of market risk by mu while carrying 0.53 of them. Its")
    say("  11.10% expected return is therefore not a return it can earn, and the")
    say("  1.11 Sharpe ratio is the size of the inconsistency rather than a")
    say("  reward. Under the single-index matrix that trade does not exist: mu")
    say("  and Sigma are built from one set of betas, so a stock can only be")
    say("  bought for its beta by also accepting the risk that beta implies.")
    say()
    say("  The residual 0.17 gap is real and worth naming in the video: the FF")
    say("  post-ranking betas come from monthly data over 2008-2025, while the")
    say("  realised beta is measured on daily data over 2023-2025. The same")
    say("  stock does not have the same beta in both.")

    # --- what it holds -----------------------------------------------------
    say()
    say("LARGEST POSITIONS, main portfolio")
    say(f"      {'ticker':<8} {'weight':>8} {'beta':>7} {'mu-rf':>8} "
        f"{'resid vol':>10} {'total vol':>10}")
    for permno, weight in w_si.sort_values(ascending=False).head(20).items():
        say(f"      {str(tickers.get(permno, permno)):<8} {weight:>8.3%} "
            f"{betas[permno]:>7.3f} {mu[permno]:>8.2%} "
            f"{np.sqrt(resid_var[permno]):>10.2%} "
            f"{np.sqrt(sigma_si.loc[permno, permno]):>10.2%}")

    grp = pd.DataFrame({"w": w_si}).join(mu_all[["size_group", "beta_group"]])
    say()
    say(f"      {'beta group':<12} {'weight':>9} {'held':>6}     "
        f"{'size group':<12} {'weight':>9} {'held':>6}")
    for g in range(1, 6):
        b = grp[grp["beta_group"] == g]
        s = grp[grp["size_group"] == g]
        say(f"      {g:<12} {b['w'].sum():>9.2%} {int((b['w'] > 1e-6).sum()):>6}     "
            f"{g:<12} {s['w'].sum():>9.2%} {int((s['w'] > 1e-6).sum()):>6}")

    say()
    say(f"  weighted ln(ME) {float((w_si * mu_all.loc[assets, 'ln_me']).sum()):.3f} "
        f"vs {mu_all.loc[assets, 'ln_me'].mean():.3f} equal-weighted")

    # --- write -------------------------------------------------------------
    sigma_si.to_csv(PROCESSED / "covariance_single_index.csv")
    sigma_lw.to_csv(PROCESSED / "covariance_ledoit_wolf.csv")
    out = pd.DataFrame({
        "main_single_index": w_si,
        "single_index_unconstrained": w_si_unc,
        "ledoit_wolf": w_lw,
        "beta": betas,
        "mu_excess": mu,
        "resid_var": resid_var,
    })
    out["ticker"] = [tickers.get(p) for p in out.index]
    out.index.name = "permno"
    out.to_csv(PROCESSED / "tangency_weights.csv")
    pd.Series({"market_var": market_var,
               "model_vol": stats_si["volatility"],
               "realised_vol": realised_vol_si,
               "excess_return": stats_si["excess_return"],
               "sharpe": stats_si["sharpe"]}).to_csv(PROCESSED / "portfolio_stats.csv")

    say()
    say("WRITTEN")
    for f in ["covariance_single_index.csv", "covariance_ledoit_wolf.csv",
              "tangency_weights.csv", "portfolio_stats.csv"]:
        say(f"  {PROCESSED / f}")

    report = OUTPUT / "06_portfolio.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport: {report}")


if __name__ == "__main__":
    main()
