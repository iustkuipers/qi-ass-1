# Quantitative Investing — Empirical Assignment, Part 1

Replication of **Fama & French (1992), "The Cross-Section of Expected Stock
Returns", *Journal of Finance* 47(2)**, used to build a EUR 10,000,000 portfolio
over the 504 S&P 500 firms in `Hand_in_sheet.xlsx`, traded at the closing prices
of **31 December 2025**.

Every number quoted below is produced by the pipeline and collected in
[`fase1/output/summary_numbers.txt`](fase1/output/summary_numbers.txt). If you
re-run the code, re-run step 8 and this README's numbers can be checked against
it.

---

## The answer, in one table

| | |
|---|---|
| Expected return rule | E[rᵢ] − r_f = MRP × βᵢ |
| Market risk premium | **8.30%** a year (mean `mktrf`, 1926-07 → 2025-12, t = 4.50) |
| Risk-free rate | 1.00% a year (given by the assignment) |
| Covariance | single index, Σ = σ²_m ββ' + D |
| Risky portfolio | long-only, 5% cap, **486 stocks**, largest weight 0.543% |
| Expected excess return | 8.77% |
| Volatility | 15.80% |
| Sharpe ratio | 0.555 |
| Risk aversion A | 4 |
| y\* = (E[r]−r_f)/(A σ²) | **0.878** |
| **Invested in stocks** | **EUR 8,725,445.51** |
| **Risk-free** | **EUR 1,274,554.49** |
| **Total** | **EUR 10,000,000.00** (exact) |

Ten largest positions: BLK, TROW, MCO, MA, TEL, AME, ITW, HLT, XYL, GS — each
between 0.39% and 0.47% of the budget.

---

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt

cd fase1
python scripts/01_load_and_check.py     # data, checks, clean copies
python scripts/02_characteristics.py    # book equity, market equity, B/M
python scripts/03_betas.py              # pre- and post-ranking betas, Table I/II
python scripts/04_fama_macbeth.py       # Table III, the CAPM test
python scripts/05_expected_returns.py   # mu
python scripts/06_portfolio.py          # covariance, tangency portfolios
python scripts/07_allocation.py         # y*, whole shares, the hand-in sheet
python scripts/08_summary_numbers.py    # every headline number in one file

python -m pytest tests -q               # 111 tests
```

Each script writes a readable report to `fase1/output/` and its data to
`fase1/data/processed/`. They run in order and each depends on the previous one.

**The raw data is not in this repository** — see [Data](#data) below.

---

## Repository layout

```
.
├── README.md                      this file
├── requirements.txt
├── .gitignore                     keeps all WRDS data out of git
├── S&P500_2025.txt                ticker list (public)
│
└── fase1/
    ├── config.py                  every choice that is a choice, in one place:
    │                              budget, rf, trade date, sheet layout, the
    │                              excluded PERMNOs, the weight cap, A
    ├── prompts.md                 AI statement: every prompt, verbatim, with
    │                              what was decided at each step
    │
    ├── modules/                   the logic, one module per step
    │   ├── module1_data.py            loading and validating CRSP
    │   ├── module2_characteristics.py book equity, market equity, B/M timing
    │   ├── module3_betas.py           pre/post-ranking betas, 5×5 sorts
    │   ├── module4_fama_macbeth.py    the cross-sectional regressions
    │   ├── module5_expected_returns.py  mu, and the market risk premium
    │   ├── module6_portfolio.py       covariance and tangency optimisers
    │   └── module7_allocation.py      y*, whole shares, the Excel sheet
    │
    ├── scripts/                   one runnable step each, 01 … 08
    │
    ├── tests/                     111 tests, one file per module
    │
    ├── output/                    the reports (committed — they are the audit
    │   ├── 01_data_check.txt        trail for everything claimed here)
    │   ├── … 07_allocation.txt
    │   └── summary_numbers.txt
    │
    └── data/
        ├── raw/                   WRDS downloads — NOT in git
        ├── processed/             cleaned panels — NOT in git
        └── output/                the deliverables
            ├── allocation.csv
            └── Hand_in_sheet_part1.xlsx
```

---

## Data

Nothing under `data/raw/` or `data/processed/` is committed. CRSP, Compustat and
the Fama-French factor files are licensed to the VU and may not be republished.
To reproduce, download into `fase1/data/raw/`:

| file | source | contents |
|---|---|---|
| any `*.csv` with a `MthRet` column | CRSP Monthly Stock File | 504 PERMNOs, 2006-01 → 2025-12 |
| any `*.csv` with a `DlyRet` column | CRSP Daily Stock File | 504 PERMNOs, 2016-01 → 2025-12 |
| `compustat.csv` | CRSP/Compustat Merged, Fundamentals Annual | fiscal years 2004–2025, `LINKPRIM`, `LPERMNO`, `SEQ`, `CEQ`, `AT`, `LT`, `PSTKRV`, `PSTKL`, `PSTK`, `TXDITC`, `TXDB`, `ITCB`, `IB`, `SICH` |
| `famafrench.csv` | WRDS FF factors, monthly | 2006–2025 |
| `famafrenchlong.csv` | WRDS FF factors, monthly | 1926-07 onwards, for the long-run MRP |
| `data2025.csv` (repo root) | the assignment's own WRDS extract | used only for `DlyVol` and the daily market return `vwretd` |

The two CRSP files are found by the columns they contain, not by filename, so a
re-download with a different WRDS name needs no code change.

`Hand_in_sheet.xlsx` sits in the repository root locally but is git-ignored: it
is course material.

---

## The method, step by step

### 1 — Load and check (`01_load_and_check.py`)

The universe is read from the hand-in sheet itself: rows 22–526 hold **505 rows
for 504 firms**, because HST (PERMNO 46703) is listed twice on rows 253 and 254.

What the checks found:

- **All 504 sheet prices equal CRSP's 31-12-2025 December price** to within a
  cent. No row in either panel is dated after the trading date.
- **600 duplicate monthly rows and 231 duplicate daily rows.** They are exact
  copies in every column, so dropping them loses nothing. The code *verifies*
  this and refuses to drop anything if two rows sharing a key ever differ —
  otherwise de-duplication would be a silent data choice.
- **Six dual share classes, not three.** Besides GOOG/GOOGL, FOX/FOXA and
  NWS/NWSA, the sheet also contains Molson Coors, McCormick and Lennar twice.
  Those three carry the *same ticker* on both rows and differ only by CUSIP, so
  a ticker filter would have missed them. Detection is by PERMCO.
- No firm has a gap *inside* its own history, in either panel, so compounding
  is safe.
- The sheet omits BRK.B and BF.B, which do appear in `S&P500_2025.txt`. We
  invest in what the sheet prescribes.

### 2 — Book equity, market equity, B/M (`02_characteristics.py`)

Book equity follows **Davis, Fama & French**:

```
BE = stockholders' equity + deferred taxes − preferred stock
     SEQ → CEQ+PSTK → AT−LT
           TXDITC → TXDB+ITCB → 0
                              PSTKRV → PSTKL → PSTK → 0
```

The Fama-French timing, which is the easy thing to get wrong:

> for July of year *t* through June of *t+1*: size = ME at end of June *t*;
> B/M = BE of the fiscal year ending in calendar *t−1*, divided by ME at end of
> December *t−1*.

The six-month gap is what keeps the exercise honest — accounting for a fiscal
year ending in *t−1* is public well before July of *t*.

- 71 Compustat rows were dated after 31-12-2025 and were dropped.
- `LINKPRIM ∈ {P, C}` leaves exactly one row per (PERMNO, fiscal year); the code
  raises rather than guessing if that ever fails.
- **338 firm-years have negative book equity**, across 67 firms — DPZ (22
  years), AZO (17), HCA, SBAC, PM. These are the well-known heavy-buyback names,
  which is good evidence the calculation is right.
- 103 firms are financials by `SICH` 6000–6999. Note this sweeps in 28 REITs,
  the exchanges (CME, ICE, NDAQ, CBOE) and the alternative managers (BX, KKR,
  APO) — and it splits **V (SICH 6099, financial) from MA (7389, not)**.

### 3 — Betas and the 5×5 sorts (`03_betas.py`)

Pre-ranking beta at each June from the preceding 60 months (≥24 required), then
5 size groups × 5 beta groups, equal-weighted monthly returns from July *t* to
June *t+1*, and a **post-ranking beta** estimated on the full 210-month series.
Every stock inherits its cell's beta. Betas are "sum of slopes" — the stock
regressed on the current *and* previous month's market return, as FF do, to
undo non-synchronous trading.

The double sort does exactly what it is designed to do:

| | across beta groups | across size groups |
|---|---|---|
| spread in post-ranking beta | **0.80** (0.66 → 1.46) | 0.15 |
| spread in ln(ME) | 0.05 | **3.11** |

Beta varies while size is held flat and vice versa — the precondition for
step 4 meaning anything.

### 4 — Fama-MacBeth (`04_fama_macbeth.py`)

Six specifications, as in FF Table III. Main sample excludes financials and
negative book equity. 210 months, ~335 stocks per cross-section.

| specification | intercept | beta | ln(ME) | ln(BE/ME) | R² |
|---|---|---|---|---|---|
| (1) beta | 0.255 (0.83) | **1.002 (2.27)** | | | 0.047 |
| (2) ln(ME) | 4.290 (5.42) | | −0.309 (−5.42) | | 0.016 |
| (3) beta + ln(ME) | 3.196 (5.23) | 0.745 (1.71) | −0.276 (−5.54) | | 0.060 |
| (4) ln(BE/ME) | 1.125 (3.21) | | | −0.126 (−1.60) | 0.020 |
| (5) ln(ME) + ln(BE/ME) | 4.175 (5.37) | | −0.319 (−5.72) | −0.168 (−2.14) | 0.035 |
| (6) all three | 3.100 (5.25) | 0.726 (1.67) | **−0.283 (−5.98)** | −0.149 (−1.95) | 0.078 |

Percent per month, Fama-MacBeth t in brackets. Newey-West (6 lags) changes
almost nothing. Adding financials back changes almost nothing either
(beta 0.738, ln(ME) −0.268, ln(BE/ME) −0.139), which is why we felt able to
apply the premia to financials.

**Three results that matter:**

1. **The CAPM is not rejected when beta is the only regressor.** γ₁ = 1.002% per
   month against a market premium of 0.972% over the *same* months; the
   difference is 0.030% with **t = 0.10**, tested month by month so the market's
   own variation is out of the standard error. The intercept, 0.255% (t = 0.83),
   is not distinguishable from zero. **This is the opposite of FF (1992).**
2. **B/M has the wrong sign for us**: −0.149% per month (t = −1.95), against
   FF's +0.50% (t = 5.71). Within large-cap US equity over 2008–2025, growth beat
   value. We do not reproduce the paper's headline result, and say so.
3. **Size is the strongest slope** (−0.283, t = −5.98) and also the least
   trustworthy — see below.

**Where the survivorship is.** Our universe is the S&P 500 *as constituted in
December 2025*, projected back to 2008. Firms that failed or fell out are not in
the list. Equal-weighted, our universe earns 1.282% per month against 0.972% for
the market: **+0.310% per month, t = 3.90, about 3.8% a year of pure selection.**

It is *not* in the specification-(6) intercept, which is easy to misread. That
3.100% is the fitted return of a firm with ln(ME) = 0 — worth one million
dollars, far outside our data. Demeaning ln(ME) and ln(BE/ME) leaves every slope
identical (as OLS algebra requires) and moves the intercept to 0.539%
(t = 1.78). That is the real zero-beta premium.

Subperiods: size and B/M are stable in sign; beta does all its work after 2016
(0.272 with t = 0.44 in 2008-2016, 1.156 with t = 1.91 in 2017-2025).

### 5 — Expected returns (`05_expected_returns.py`)

The decision rule was: **use only characteristics that are significant and not
an artefact of how the universe was selected.** Size fails the second test, B/M
fails the first, the negative-BE dummy exists only to support B/M, and the
intercept is insignificant. What survives is the CAPM:

```
E[rᵢ] − r_f = MRP × βᵢ ,   MRP = 8.30% a year
```

Two consequences worth knowing:

- There are only **25 distinct expected returns**, one per 5×5 cell, ranging
  from 4.60% to 13.25% excess. Within a cell the optimiser chooses purely on
  covariance.
- **The choice of MRP does not change the portfolio weights at all.** The
  tangency portfolio depends only on the *direction* of μ − r_f, and MRP is a
  positive scalar on that vector. It moves the expected return, the Sharpe
  ratio and y\*, but not which stocks are held. (Tested:
  `test_scaling_every_expected_return_leaves_the_weights_unchanged`.)

Two robustness sets are reported but not submitted: (a) specification (6) with
the level anchored, which tilts small and growth; (b) beta + B/M without size,
which tilts *large* and hard into growth, because the low-B/M names in the S&P
500 are the mega-caps.

### 6 — Covariance and the tangency portfolio (`06_portfolio.py`)

**The diagnostic that decided the design.** With a Ledoit-Wolf covariance
estimated stock by stock, the long-only portfolio had an assigned beta of 1.093
but a **realised beta of 0.530** (regression of its daily returns on `vwretd`,
t = 34.3). μ was paying it for market risk it did not carry: the optimiser had
found stocks in high-beta 5×5 cells whose own covariance with the market is low,
and bought the discrepancy. Its Sharpe of 1.11 measured that inconsistency, not
a reward.

**The fix** is a single-index covariance built on *the same betas as μ*:

```
Σ = σ²_m ββ' + D ,   D_ii = Var(rᵢ − βᵢ r_m) over the last 3 years of daily data
```

By the Sherman-Morrison identity, Σ⁻¹β = D⁻¹β / (1 + σ²_m β'D⁻¹β), so the
unconstrained tangency weights are **exactly proportional to βᵢ / residual
varianceᵢ**. Verified numerically: correlation 1.0000000000, largest deviation
5.9 × 10⁻¹⁶. There is nothing left to arbitrage — a stock can only be bought for
its beta by also accepting the risk that beta implies.

A consequence: since every β and every residual variance is positive, the
portfolio is **naturally long-only**. It holds all 486 stocks with a maximum
weight of 0.543% and an effective N of 397, and the 5% cap never binds.

| | **MAIN** single-index | robustness Ledoit-Wolf |
|---|---|---|
| expected excess return | 8.77% | 9.09% |
| volatility | 15.80% | 9.97% |
| Sharpe | **0.555** | 0.910 |
| positions | 486 | 50 |
| assigned β | 1.056 | 1.093 |
| **realised β** | **0.888** | 0.530 |

The remaining 0.17 gap is real and worth naming: the FF post-ranking betas come
from monthly data over 2008–2025, the realised beta from daily data over
2023–2025. The same stock does not have the same beta in both.

**Takeover screen.** A stock under an agreed all-cash bid is no longer equity —
its price pins to the offer, so its volatility describes a different asset.
Comparing Q4-2025 volatility with Jan-Sep 2025 volatility (a *level* screen only
finds utilities):

| ticker | Jan–Sep vol | Q4 vol | ratio |
|---|---|---|---|
| DAY | 49.0% | 2.8% | **0.06** |
| EA | 37.3% | 3.0% | **0.08** |
| HOLX | 32.6% | 13.7% | **0.42** |
| TTD | 83.9% | 36.8% | 0.44 |

Median ratio 0.82. The three known deals are the three lowest and the screen
finds no fourth — after HOLX the absolute volatilities are 30–40%, a quiet
quarter rather than a bid. All three excluded.

### 7 — Allocation and the sheet (`07_allocation.py`)

Two-fund separation: *which* risky portfolio is settled in step 6 and does not
depend on risk aversion; *how much* of it does.

| A | y\* raw | y\* used | risky | risk-free |
|---|---|---|---|---|
| 3 | 1.170 | 1.000 | 10,000,000 | 0 |
| **4** | **0.878** | **0.878** | **8,778,502** | **1,221,498** |
| 5 | 0.702 | 0.702 | 7,022,802 | 2,977,198 |

Borrowing is not allowed, so y\* is capped at 1 — which binds at A = 3.

Shares are rounded **down**, never to nearest: rounding to nearest can overspend
a fixed budget. The leftover joins the risk-free holding, so it earns 1% rather
than sitting idle. Nothing was rounded out of the portfolio.

```
invested in stocks   EUR 8,725,445.51   (486 positions)
risk-free            EUR 1,274,554.49   (1,221,498 deliberate + 53,057 rounding)
TOTAL                EUR 10,000,000.00  exact
```

The completed sheet writes shares into column **H** and the risk-free amount
into **H21** (that row's price is 1, so the cell is the euro amount). HST gets
its shares on row 253 and zero on row 254.

**We repaired the template.** Its subtotals read `=SUM(I22:I524)`, two rows short
of the firm block, silently omitting ZBRA and ZTS — the sheet would not have
summed to 100%. `I16`, `L16` and `O16` now run to row 526.

---

## Who decided what

The assignment requires us to be able to defend every choice, so this is
explicit. `fase1/prompts.md` has the full record.

### Decided by us (the group)

- **The paper**: Fama & French (1992) — it is the basis of the course and likely
  exam material.
- **Long-only, 5% cap**; monthly estimation of the cross-section.
- **One share class per PERMCO**, keeping the one with the higher average 2025
  volume; LEN 52708, MKC 52090, TAP 59248 named directly.
- **Size breakpoints within our own universe** rather than NYSE breakpoints, and
  naming that as a deviation from FF.
- **Financials**: main specification without them (as FF), robustness with; apply
  the premia to them if the slopes agree, which they did.
- **Negative book equity**: excluded from the Table III replication as FF do, but
  kept for the μ regression with a D(BE<0) dummy and ln(BE/ME) set to zero —
  the same treatment FF give negative E/P.
- **The rule for μ**: only characteristics that are significant *and* not created
  by our sample selection. This is what eliminated size and B/M and left the
  CAPM.
- **The long-run MRP** from 1926 rather than the in-sample premium.
- **The takeover exclusions** DAY, EA, HOLX, and the instruction to screen for
  others.
- **A = 4**, no borrowing, with A = 3 and A = 5 reported.
- **The correction that mattered most**: after step 3 Claude attributed the
  positive beta-return relation to survivorship. We pointed out that the Table I
  slope (0.69/0.80 = 0.86% per month) is almost exactly the mean `mktrf`, so the
  beta effect is CAPM-consistent and the survivorship is in the *level* and in
  the size effect. Step 4 confirmed this precisely.
- **The beta-consistency diagnostic**: we predicted the Ledoit-Wolf portfolio
  would have a realised beta near 0.5 against an assigned 1.09, and specified the
  single-index fix. The realised beta came out at 0.530.

### Decided by Claude (and flagged for us to check)

- **B/M denominator is company-level market equity**, summed across share
  classes, while size stays the individual stock's ME. Book equity is a
  company-level number; this is also FF's own convention.
- **Excess-on-excess betas** rather than FF's raw-on-raw, for consistency with
  step 4 — with a check showing it makes no difference (correlation 0.9992).
- **D uses the imposed beta**, not a refitted one, so a stock whose assigned beta
  is too high keeps market risk in its residual and automatically gets less
  weight.
- **The Schaible transformation** to turn the capped max-Sharpe problem into an
  exact convex QP instead of a risk-aversion grid search.
- **Covariance coverage rule**: keep stocks trading on ≥98% of days, then drop the
  few days with any gap. A 100% rule would have thrown out GEHC over two missing
  days; a per-stock rule would have cost everyone months of history for KVUE.
- **Rounding down rather than to nearest**, so the budget can never be exceeded.
- **Repairing the sheet's subtotal formulas.**
- **Truncating both look-ahead problems**: `famafrench.csv` ran to July 2026 and
  `compustat.csv` to August 2026.

### Things Claude found that we had not asked about

- Fama & MacBeth (1973) is in the *JPE* and is therefore not an eligible source
  paper; the method enters legitimately through FF (1992).
- Three extra dual share classes (TAP, MKC, LEN) that a ticker filter misses.
- CRH and VLTO fall out with 22 and 21 months at the June-2025 formation — our
  "too short" list of six was really eight.
- The exclusion of ETN and DAY for missing Compustat became unnecessary once μ
  reduced to MRP × β; ETN was put back.
- That the MRP choice cannot affect the weights.

### Bugs Claude found and fixed in its own work

1. **Infinite B/M.** Company market equity was summed with `sum()`, which returns
   0.0 for an all-missing group rather than NaN. Super Micro has no CRSP
   December price for 2018 and 2019 (it was off Nasdaq over late SEC filings), so
   its B/M became infinite and silently destroyed twelve monthly regressions in a
   diagnostic. Fixed with `sum(min_count=1)` and a zero guard.
2. **Variable shadowing** in `04_fama_macbeth.py`, which made the survivorship
   section report the robustness sample's intercept.
3. **A 0/0 t-statistic** when a difference series has no variation, which
   returned `nan` where the right answer is "no difference".

---

## Known limitations

Say these out loud in the video rather than being asked about them.

1. **Survivorship bias, ~3.8% a year.** Baked into the assignment — we are
   required to use the December-2025 constituent list — but it inflates the level
   of returns and the size effect. It is the reason size was dropped from μ.
2. **We do not reproduce FF's value result.** Our B/M slope is negative and
   marginally significant. That is what our sample says.
3. **Only 210 monthly cross-sections** (July 2008 → December 2025), because
   pre-ranking betas need 24 months and CRSP monthly starts in 2006. FF had 330.
4. **25 distinct expected returns.** Assigning a portfolio's beta to each of its
   stocks is FF's method, but it makes μ coarse. Within a cell the portfolio is
   chosen entirely by covariance.
5. **The assigned and realised betas still differ by 0.17**, because the betas
   come from monthly 2008–2025 data and the covariance from daily 2023–2025 data.
6. **Σ is estimated in-sample.** The Sharpe ratio is a model statement, not an
   out-of-sample result.
7. **Size breakpoints are within-universe**, not NYSE. Every firm here is a large
   cap, so "small" means "small among the S&P 500".

---

## Testing

```bash
cd fase1 && python -m pytest tests -q      # 111 passed
```

One test file per module. They test the things that could quietly produce a
wrong answer rather than every function: CRSP's missing-value codes, the FF
timing, the sum-of-slopes beta, that de-meaning does not move slopes, that
scaling μ does not move weights, that the capped optimiser respects its
constraints, that the single-index weights really are β/resvar, and that the
budget lands on exactly EUR 10,000,000.

---

## Hand-in checklist

1. **Excel** — `fase1/data/output/Hand_in_sheet_part1.xlsx` ✅
2. **Summary (max 1 page)** — to write, from `output/summary_numbers.txt`
3. **Code** — this repository ✅
4. **AI statement** — `fase1/prompts.md` ✅
5. **Video (2–3 min)** — to record

Before submitting, fill in `GROUP` at the top of
`fase1/scripts/07_allocation.py` with the group number and student names and
re-run step 7, so they appear in the sheet header.
