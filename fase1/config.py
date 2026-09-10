"""Project-wide paths and constants for Part 1.

Everything that is a *choice* rather than a computation lives here, so that the
scripts read as a description of the method and not as a pile of magic numbers.
"""

from pathlib import Path

import pandas as pd

# --- paths ---------------------------------------------------------------
FASE1 = Path(__file__).resolve().parent
PROJECT = FASE1.parent

RAW = FASE1 / "data" / "raw"
PROCESSED = FASE1 / "data" / "processed"
OUTPUT = FASE1 / "output"

HAND_IN_SHEET = PROJECT / "Hand_in_sheet.xlsx"

for _d in (PROCESSED, OUTPUT):
    _d.mkdir(parents=True, exist_ok=True)

# --- assignment rules ----------------------------------------------------
BUDGET = 10_000_000.0          # EUR, must be fully allocated
RF_ANNUAL = 0.01               # 1% per year, given by the assignment
TRADE_DATE = pd.Timestamp("2025-12-31")   # no information after this date

# --- layout of Hand_in_sheet.xlsx ---------------------------------------
# Read off the file itself: row 20 is the header, row 21 is the risk-free
# line, rows 22-526 are the 505 firm lines (504 firms; HST appears twice).
SHEET_HEADER_ROW = 20
SHEET_RF_ROW = 21
SHEET_FIRST_FIRM_ROW = 22
SHEET_LAST_FIRM_ROW = 526
SHEET_SHARES_COL_PART1 = "H"
SHEET_WEIGHT_COL_PART1 = "I"

N_FIRMS_EXPECTED = 504

# CRSP uses these numeric codes for "return not available"; they are not
# returns of -66 to -99 and must not be treated as such.
CRSP_MISSING_RETURN_CODES = (-66.0, -77.0, -88.0, -99.0)

# --- who can receive weight ---------------------------------------------
# Six companies appear twice, as two share classes. We hold one class each.
# The class kept is the one with the higher average daily volume in 2025;
# that rule independently reproduces the CRSP/Compustat primary link
# (LINKPRIM = 'P') for all six, which is a useful cross-check.
#
#   PERMCO           kept                  dropped        2025 avg volume
#   33     Molson Coors  TAP   59248   TAP   90562     2,637,082  vs        96
#   2801   McCormick     MKC   52090   MKC   89155     2,420,822  vs     3,921
#   21095  Lennar        LEN   52708   LEN   89731     3,934,685  vs    58,386
#   45483  Alphabet      GOOGL 90319   GOOG  14542    35,928,254  vs 23,342,467
#   54433  News Corp     NWSA  13963   NWS   13964     3,501,054  vs   900,695
#   56662  Fox           FOXA  18420   FOX   18421     3,742,945  vs 1,401,265
SHARE_CLASS_KEPT = (59248, 52090, 52708, 90319, 13963, 18420)
SHARE_CLASS_DROPPED = (90562, 89155, 89731, 14542, 13964, 18421)

# Firms with fewer than 24 months of return history at the trading date. They
# cannot get a pre-ranking beta, so they get no expected return and no weight.
TOO_SHORT_HISTORY = (27598, 27083, 26181, 25146, 24877, 24878)  # Q PSKY SNDK SW SOLV GEV
MIN_MONTHS_HISTORY = 24

# Eaton and Dayforce have no Compustat row. Once the expected return reduced
# to MRP * beta, B/M stopped mattering and Eaton came back in. Dayforce is
# still excluded, but as a takeover target rather than for missing data.
EXCLUDE_NO_COMPUSTAT: tuple[int, ...] = ()

# Firms under a cash takeover bid that was public before the trading date.
# Their price is pinned to the offer, so their historical return distribution
# says nothing about what they will do next: the remaining return is the deal
# spread, not equity risk. Leaving them in would let the optimiser load up on
# what looks like a near-riskless asset.
#   DAY   Dayforce, $70 cash
#   EA    Electronic Arts, $210 cash
#   HOLX  Hologic, $76 cash plus a CVR
# The Q4-2025 volatility screen in step 6 finds exactly these three and no
# others, which is the check that we have not missed a deal.
PENDING_DEALS = (17700, 75828, 76095)      # DAY, EA, HOLX

# --- covariance and portfolio -------------------------------------------
COV_YEARS = 3                  # daily returns over the three years to 31-12-2025
TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12
WEIGHT_CAP = 0.05              # long-only variant: at most 5% in any one stock

# --- risk aversion -------------------------------------------------------
# y* = (E[r] - rf) / (A * sigma^2), the week 1 / week 2 result for a
# mean-variance investor. A = 4 is the group's choice; 3 and 5 are reported
# alongside to show how much the split moves. Borrowing is not allowed, so
# y* is capped at 1.
RISK_AVERSION = 4.0
RISK_AVERSION_ALTERNATIVES = (3.0, 5.0)
ALLOW_BORROWING = False
