"""Module 4 - Fama-MacBeth cross-sectional regressions (FF 1992, Table III).

The method is two passes. Every month we run one cross-sectional regression of
stock excess returns on characteristics known before the month started:

    r_it - rf_t = g0_t + g1_t * beta_i + g2_t * ln(ME_i) + g3_t * ln(BE/ME_i) + e_it

That gives a time series of slopes, one per month. The reported "risk premium"
is the average of that series, and its standard error comes from the
variability of the slopes across months rather than from any single
regression. That is the whole trick of Fama and MacBeth (1973): the monthly
slopes are (nearly) independent draws, so the cross-sectional correlation in
the residuals that would wreck a pooled regression never enters.

Timing, following FF and fixed in step 2:
    for July of year t through June of year t+1
        beta       post-ranking beta of the 5x5 cell the stock was sorted
                   into in June of year t
        ln(ME)     June of year t
        ln(BE/ME)  book equity of the fiscal year ending in t-1, over market
                   equity at the end of December of t-1

Nothing on the right-hand side of a month-t regression was published after
month t started.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Newey-West lag length. The usual rule of thumb, 4*(T/100)^(2/9), gives 5 for
# our 210 months; 6 is the conventional round number for monthly data and is
# the more conservative of the two.
NEWEY_WEST_LAGS = 6

# The six specifications of FF (1992) Table III that we can build from CRSP
# and Compustat. FF also report leverage and E/P variants; those need extra
# Compustat items and are not part of what we set out to replicate.
SPECS: dict[str, list[str]] = {
    "(1) beta": ["beta"],
    "(2) ln(ME)": ["ln_me"],
    "(3) beta + ln(ME)": ["beta", "ln_me"],
    "(4) ln(BE/ME)": ["ln_bm"],
    "(5) ln(ME) + ln(BE/ME)": ["ln_me", "ln_bm"],
    "(6) beta + ln(ME) + ln(BE/ME)": ["beta", "ln_me", "ln_bm"],
}

# The specification the expected returns will come from in step 5. Negative
# book equity firms stay in, with ln(BE/ME) set to zero and a dummy carrying
# whatever they earn on average, which is how FF handle negative E/P.
MU_SPEC = ["beta", "ln_me", "ln_bm", "neg_be_dummy"]


# ---------------------------------------------------------------------------
# assembling the panel
# ---------------------------------------------------------------------------


def build_regression_panel(assignments: pd.DataFrame,
                           characteristics: pd.DataFrame,
                           excess_panel: pd.DataFrame) -> pd.DataFrame:
    """One row per (month, stock) with the return and its explanatory variables.

    `assignments` supplies the post-ranking beta for the FF year, and the FF
    year is expanded into its twelve holding months here, so that a stock
    keeps the same beta, size and B/M from July to June - which is exactly the
    timing FF describe.
    """
    chars = characteristics[["permno", "ff_year", "ln_me", "ln_bm", "bm",
                             "neg_be", "is_financial", "me_june"]]
    base = assignments[["ff_year", "permno", "size_group", "beta_group",
                        "post_beta"]].merge(chars, on=["permno", "ff_year"],
                                            how="left")

    months = excess_panel.index
    rows = []
    for ff_year, group in base.groupby("ff_year"):
        start = pd.Period(f"{ff_year}-07", freq="M")
        end = pd.Period(f"{ff_year + 1}-06", freq="M")
        window = months[(months >= start) & (months <= end)]
        for ym in window:
            block = group.copy()
            block["ym"] = ym
            rows.append(block)

    long = pd.concat(rows, ignore_index=True)

    # attach the realised excess return of that month
    ret = (excess_panel.stack().rename("exret").reset_index()
           .rename(columns={"level_0": "ym", "level_1": "permno"}))
    ret.columns = ["ym", "permno", "exret"]
    long = long.merge(ret, on=["ym", "permno"], how="left")

    long = long.rename(columns={"post_beta": "beta"})
    long["neg_be_dummy"] = long["neg_be"].astype(float)

    # FF set the value variable to zero for firms where it is not defined, and
    # let the dummy pick up their average return instead.
    long["ln_bm_filled"] = long["ln_bm"].fillna(0.0)
    return long


def apply_sample(panel: pd.DataFrame, exclude_financials: bool,
                 exclude_negative_be: bool) -> pd.DataFrame:
    """FF's main sample drops financial firms and firms with negative BE."""
    out = panel
    if exclude_financials:
        out = out[~out["is_financial"]]
    if exclude_negative_be:
        out = out[~out["neg_be"]]
    return out


# ---------------------------------------------------------------------------
# the two passes
# ---------------------------------------------------------------------------


def cross_section_regressions(panel: pd.DataFrame,
                              regressors: list[str]) -> pd.DataFrame:
    """One OLS per month; returns the slopes, R-squared and sample size.

    A month is skipped only if it has fewer observations than coefficients,
    which cannot happen in our data but would otherwise produce a silent
    garbage slope.
    """
    needed = ["exret", *regressors]
    rows = []
    for ym, month in panel.groupby("ym"):
        data = month[needed].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) <= len(regressors) + 1:
            continue
        X = sm.add_constant(data[regressors], has_constant="add")
        fit = sm.OLS(data["exret"], X).fit()
        row = {"ym": ym, "n": len(data), "r2": fit.rsquared,
               "intercept": fit.params["const"]}
        row.update({r: fit.params[r] for r in regressors})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("ym").reset_index(drop=True)


def fm_summary(slopes: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Average slope with a Fama-MacBeth t-statistic and a Newey-West one.

    The FM t-statistic treats the monthly slopes as independent. The
    Newey-West version allows them to be autocorrelated, which they can be
    when the characteristics themselves are persistent - as size and B/M are,
    since they only change once a year.
    """
    rows = []
    T = len(slopes)
    for col in columns:
        series = slopes[col].dropna()
        mean = series.mean()
        fm_t = mean / (series.std(ddof=1) / np.sqrt(len(series)))

        const = np.ones((len(series), 1))
        nw = sm.OLS(series.to_numpy(), const).fit(
            cov_type="HAC", cov_kwds={"maxlags": NEWEY_WEST_LAGS})
        rows.append({
            "variable": col,
            "mean": mean,
            "fm_t": fm_t,
            "nw_t": float(nw.tvalues[0]),
            "sd": series.std(ddof=1),
            "n_months": len(series),
            "share_positive": float((series > 0).mean()),
        })
    out = pd.DataFrame(rows)
    out.attrs["T"] = T
    return out


# ---------------------------------------------------------------------------
# the CAPM test the assignment asks for
# ---------------------------------------------------------------------------


def capm_slope_test(slopes: pd.DataFrame, factors: pd.DataFrame) -> dict:
    """Compare the beta premium with the realised market premium.

    Under the CAPM the cross-sectional slope on beta should equal the market's
    own excess return in that month, and the intercept should be zero. Testing
    the difference series g1_t - mktrf_t month by month is the sharpest form of
    the test, because it removes the market's own variation from the standard
    error instead of leaving it in.
    """
    merged = slopes.merge(factors[["ym", "mktrf"]], on="ym", how="left")
    diff = (merged["beta"] - merged["mktrf"]).dropna()

    def _t(series: pd.Series) -> tuple[float, float]:
        m = series.mean()
        se = series.std(ddof=1) / np.sqrt(len(series))
        if se == 0:
            # A series with no variation: either it is identically zero, in
            # which case there is nothing to test, or it is a constant gap,
            # which is infinitely well measured. Returning 0/0 = nan here
            # would look like a failed computation rather than either.
            return m, 0.0 if m == 0 else np.inf * np.sign(m)
        return m, m / se

    mean_diff, t_diff = _t(diff)
    mean_g1, t_g1 = _t(merged["beta"].dropna())
    mean_g0, t_g0 = _t(merged["intercept"].dropna())

    return {
        "mean_gamma1": mean_g1, "t_gamma1": t_g1,
        "mean_mktrf": merged["mktrf"].mean(),
        "mean_diff": mean_diff, "t_diff": t_diff,
        "mean_gamma0": mean_g0, "t_gamma0": t_g0,
        "n_months": len(diff),
    }


def subperiod_summary(slopes: pd.DataFrame, columns: list[str],
                      split: pd.Period) -> pd.DataFrame:
    """The same average slopes, computed on each half of the sample."""
    early = slopes[slopes["ym"] <= split]
    late = slopes[slopes["ym"] > split]
    a = fm_summary(early, columns).assign(period="early")
    b = fm_summary(late, columns).assign(period="late")
    return pd.concat([a, b], ignore_index=True)
