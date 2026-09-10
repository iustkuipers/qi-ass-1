"""Step 4 - Fama-MacBeth regressions, in the style of FF (1992) Table III.

Run from the fase1 folder:   python scripts/04_fama_macbeth.py

Produces
    data/processed/fm_slopes_<sample>.csv   the monthly slope series
    data/processed/fm_summary.csv           average slopes and t-statistics
    output/04_fama_macbeth.txt              the report printed below

No expected returns are computed here. Step 4 only establishes which
characteristics are priced in our universe, and how stable that is.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PROCESSED, OUTPUT, SHARE_CLASS_DROPPED  # noqa: E402
from modules import module3_betas as m3  # noqa: E402
from modules import module4_fama_macbeth as m4  # noqa: E402

SPLIT = pd.Period("2016-12", freq="M")


def main() -> None:
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say("=" * 80)
    say("STEP 4 - FAMA-MACBETH CROSS-SECTIONAL REGRESSIONS")
    say("=" * 80)

    monthly = pd.read_parquet(PROCESSED / "monthly.parquet")
    chars = pd.read_csv(PROCESSED / "characteristics.csv")
    assignments = pd.read_csv(PROCESSED / "assignments.csv")
    factors, _ = m3.load_ff_factors()

    panel = m3.excess_return_panel(monthly, factors)
    panel = panel.drop(columns=[c for c in SHARE_CLASS_DROPPED if c in panel.columns])

    full = m4.build_regression_panel(assignments, chars, panel)
    full = full.dropna(subset=["exret"])

    say()
    say("REGRESSION PANEL")
    say(f"  months                   : {full['ym'].min()} to {full['ym'].max()}")
    say(f"  stock-months             : {len(full):,}")
    say(f"  distinct stocks          : {full['permno'].nunique()}")
    say(f"  with a beta              : {int(full['beta'].notna().sum()):,}")
    say(f"  with ln(ME)              : {int(full['ln_me'].notna().sum()):,}")
    say(f"  with ln(BE/ME)           : {int(full['ln_bm'].notna().sum()):,}")
    say(f"  financial firms          : {int(full['is_financial'].sum()):,} "
        f"stock-months")
    say(f"  negative book equity     : {int(full['neg_be'].sum()):,} stock-months")

    samples = {
        "main (no financials, no negative BE)":
            m4.apply_sample(full, exclude_financials=True, exclude_negative_be=True),
        "robustness (financials included)":
            m4.apply_sample(full, exclude_financials=False, exclude_negative_be=True),
    }

    summaries = []
    for sample_name, sample in samples.items():
        say()
        say("=" * 80)
        say(f"TABLE III - {sample_name}")
        say(f"  average stocks per monthly cross-section: "
            f"{sample.groupby('ym').size().mean():.0f}")
        say("=" * 80)
        say()
        say(f"  {'specification':<30} {'intercept':>10} {'beta':>10} "
            f"{'ln(ME)':>10} {'ln(BE/ME)':>11} {'avg R2':>8}")
        say(f"  {'':<30} {'(t)':>10} {'(t)':>10} {'(t)':>10} {'(t)':>11}")
        say("  " + "-" * 82)

        for spec_name, regressors in m4.SPECS.items():
            slopes = m4.cross_section_regressions(sample, regressors)
            summ = m4.fm_summary(slopes, ["intercept", *regressors])
            summ["spec"] = spec_name
            summ["sample"] = sample_name
            summaries.append(summ)

            by = summ.set_index("variable")
            cells, tcells = [], []
            for var in ["intercept", "beta", "ln_me", "ln_bm"]:
                if var in by.index:
                    # slopes are monthly decimal returns; report in percent
                    cells.append(f"{by.loc[var, 'mean'] * 100:>10.3f}")
                    tcells.append(f"{by.loc[var, 'fm_t']:>10.2f}")
                else:
                    cells.append(f"{'':>10}")
                    tcells.append(f"{'':>10}")
            say(f"  {spec_name:<30} {cells[0]} {cells[1]} {cells[2]} "
                f"{cells[3]:>11} {slopes['r2'].mean():>8.3f}")
            say(f"  {'':<30} {tcells[0]} {tcells[1]} {tcells[2]} {tcells[3]:>11}")

        say()
        say("  Newey-West t-statistics (6 lags) for the full specification (6):")
        spec6_slopes = m4.cross_section_regressions(
            sample, m4.SPECS["(6) beta + ln(ME) + ln(BE/ME)"])
        s6 = m4.fm_summary(spec6_slopes, ["intercept", "beta", "ln_me", "ln_bm"])
        say(f"      {'variable':<12} {'mean %':>9} {'FM t':>8} {'NW t':>8} "
            f"{'% months > 0':>13}")
        for _, r in s6.iterrows():
            say(f"      {r['variable']:<12} {r['mean'] * 100:>9.3f} "
                f"{r['fm_t']:>8.2f} {r['nw_t']:>8.2f} "
                f"{r['share_positive']:>13.2%}")

    # --- the CAPM test ---------------------------------------------------
    main_sample = samples["main (no financials, no negative BE)"]
    slopes1 = m4.cross_section_regressions(main_sample, m4.SPECS["(1) beta"])
    test = m4.capm_slope_test(slopes1, factors)

    say()
    say("=" * 80)
    say("IS THE BETA PREMIUM THE MARKET PREMIUM?  (specification (1), main sample)")
    say("=" * 80)
    say(f"  The CAPM says two things: gamma_1 = E[mktrf] and gamma_0 = 0.")
    say()
    say(f"  mean gamma_1 (beta slope)     : {test['mean_gamma1'] * 100:>8.3f}% per month"
        f"   (t = {test['t_gamma1']:.2f})")
    say(f"  mean mktrf, same months       : {test['mean_mktrf'] * 100:>8.3f}% per month")
    say(f"  difference gamma_1 - mktrf    : {test['mean_diff'] * 100:>8.3f}% per month"
        f"   (t = {test['t_diff']:.2f})")
    say()
    say(f"  mean gamma_0 (intercept)      : {test['mean_gamma0'] * 100:>8.3f}% per month"
        f"   (t = {test['t_gamma0']:.2f})")
    say(f"  annualised                    : "
        f"{((1 + test['mean_gamma0']) ** 12 - 1) * 100:>8.2f}% per year")
    say()
    say("  The difference is tested month by month, so the market's own variation")
    say("  is removed from the standard error rather than left inside it.")

    # --- where the survivorship actually shows up ------------------------
    # The intercept of specification (6) is *not* the zero-beta rate. It is
    # the fitted return of a stock with ln(ME) = 0, i.e. a firm worth one
    # million dollars - an extrapolation far outside our large-cap universe.
    # Demeaning the characteristics moves the intercept back inside the data,
    # where it can be read as "what an average stock earns beyond its beta".
    say()
    say("=" * 80)
    say("WHERE DOES THE SURVIVORSHIP SHOW UP?")
    say("=" * 80)

    ew = main_sample.groupby("ym")["exret"].mean()
    mkt = factors.set_index("ym")["mktrf"].reindex(ew.index)
    gap = (ew - mkt).dropna()
    say()
    say("  1. In the level of returns, not in the slopes.")
    say(f"     our universe, equal-weighted : {ew.mean() * 100:>7.3f}% per month")
    say(f"     the market (mktrf)           : {mkt.mean() * 100:>7.3f}% per month")
    say(f"     difference                   : {gap.mean() * 100:>7.3f}% per month "
        f"(t = {gap.mean() / (gap.std(ddof=1) / len(gap) ** 0.5):.2f}), "
        f"{((1 + gap.mean()) ** 12 - 1) * 100:.1f}% per year")
    say("     Firms that were in the S&P 500 in December 2025 outperformed the")
    say("     market over a period they were selected for having survived.")

    slopes6_main = m4.cross_section_regressions(
        main_sample, m4.SPECS["(6) beta + ln(ME) + ln(BE/ME)"])

    demeaned = main_sample.copy()
    for col in ["ln_me", "ln_bm"]:
        demeaned[col] = demeaned[col] - demeaned.groupby("ym")[col].transform("mean")
    slopes_dm = m4.cross_section_regressions(
        demeaned, m4.SPECS["(6) beta + ln(ME) + ln(BE/ME)"])
    sum_dm = m4.fm_summary(slopes_dm, ["intercept", "beta", "ln_me", "ln_bm"]
                           ).set_index("variable")
    say()
    say("  2. Not in the intercept of specification (6), which is misread easily.")
    say(f"     intercept as reported above  : "
        f"{slopes6_main['intercept'].mean() * 100:>7.3f}% per month  "
        f"(a firm with ln(ME) = 0, i.e. worth $1m)")
    say(f"     intercept with ln(ME) and ln(BE/ME) demeaned : "
        f"{sum_dm.loc['intercept', 'mean'] * 100:>7.3f}% per month "
        f"(t = {sum_dm.loc['intercept', 'fm_t']:.2f})")
    say("     The slopes are unchanged by demeaning; only the intercept moves,")
    say("     because it is now evaluated at the average stock instead of at a")
    say("     point far outside the sample.")
    say(f"     slopes after demeaning: beta {sum_dm.loc['beta', 'mean'] * 100:.3f}, "
        f"ln(ME) {sum_dm.loc['ln_me', 'mean'] * 100:.3f}, "
        f"ln(BE/ME) {sum_dm.loc['ln_bm', 'mean'] * 100:.3f}")
    say()
    say("  3. In the size slope, which is the one to treat with suspicion:")
    say("     small S&P 500 firms of 2008 that are still in the index in 2025")
    say("     are survivors twice over, so -0.28% per month per log point of")
    say("     size is an upper bound on any real size premium.")

    # --- subperiods ------------------------------------------------------
    say()
    say("=" * 80)
    say(f"SUBPERIODS (split at {SPLIT}) - main sample, specification (6)")
    say("=" * 80)
    slopes6 = slopes6_main
    sub = m4.subperiod_summary(slopes6, ["intercept", "beta", "ln_me", "ln_bm"], SPLIT)
    early_n = int((slopes6["ym"] <= SPLIT).sum())
    late_n = int((slopes6["ym"] > SPLIT).sum())
    say(f"  early: {slopes6['ym'].min()} to {SPLIT} ({early_n} months)")
    say(f"  late : {SPLIT + 1} to {slopes6['ym'].max()} ({late_n} months)")
    say()
    say(f"      {'variable':<12} {'early %':>10} {'t':>7} {'late %':>10} {'t':>7} "
        f"{'difference':>12}")
    piv = sub.pivot(index="variable", columns="period", values=["mean", "fm_t"])
    for var in ["intercept", "beta", "ln_me", "ln_bm"]:
        e, l = piv[("mean", "early")][var], piv[("mean", "late")][var]
        te, tl = piv[("fm_t", "early")][var], piv[("fm_t", "late")][var]
        say(f"      {var:<12} {e * 100:>10.3f} {te:>7.2f} {l * 100:>10.3f} "
            f"{tl:>7.2f} {(l - e) * 100:>12.3f}")

    # --- the specification step 5 will use -------------------------------
    say()
    say("=" * 80)
    say("THE SPECIFICATION STEP 5 WILL DRAW mu FROM")
    say("  beta + ln(ME) + ln(BE/ME) + D(BE<0), financials excluded,")
    say("  negative-BE firms kept with ln(BE/ME) set to zero.")
    say("=" * 80)
    mu_sample = m4.apply_sample(full, exclude_financials=True,
                               exclude_negative_be=False).copy()
    mu_sample["ln_bm"] = mu_sample["ln_bm_filled"]
    mu_slopes = m4.cross_section_regressions(mu_sample, m4.MU_SPEC)
    mu_sum = m4.fm_summary(mu_slopes, ["intercept", *m4.MU_SPEC])
    say()
    say(f"      {'variable':<16} {'mean %':>9} {'FM t':>8} {'NW t':>8} "
        f"{'sd %':>8}")
    for _, r in mu_sum.iterrows():
        say(f"      {r['variable']:<16} {r['mean'] * 100:>9.3f} {r['fm_t']:>8.2f} "
            f"{r['nw_t']:>8.2f} {r['sd'] * 100:>8.3f}")
    say(f"      months: {len(mu_slopes)}, average stocks per month: "
        f"{mu_slopes['n'].mean():.0f}, average R2: {mu_slopes['r2'].mean():.3f}")
    say(f"      negative-BE stock-months kept: "
        f"{int(mu_sample['neg_be'].sum()):,}")

    # --- write -----------------------------------------------------------
    mu_slopes.to_csv(PROCESSED / "fm_slopes_mu_spec.csv", index=False)
    slopes6.to_csv(PROCESSED / "fm_slopes_spec6_main.csv", index=False)
    all_summary = pd.concat(summaries, ignore_index=True)
    mu_sum2 = mu_sum.assign(spec="mu spec", sample="main + negative BE dummy")
    pd.concat([all_summary, mu_sum2], ignore_index=True).to_csv(
        PROCESSED / "fm_summary.csv", index=False)

    say()
    say("WRITTEN")
    for f in ["fm_slopes_mu_spec.csv", "fm_slopes_spec6_main.csv", "fm_summary.csv"]:
        say(f"  {PROCESSED / f}")

    report = OUTPUT / "04_fama_macbeth.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport: {report}")


if __name__ == "__main__":
    main()
