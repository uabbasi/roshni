"""
Tax Tables for Financial Planning - 2026 Tax Year

Single source of truth for all tax constants used across financial planning.
This module has NO dependencies on other modules to prevent import cycles.

Sources:
- Federal brackets: IRS Rev. Proc. 2025-XX, Tax Foundation, OBBBA adjustments
- California brackets: FTB Publication 1031
- IRMAA: CMS-8089-N, CMS-8090-N (November 2025)
- RMD: IRS Publication 590-B, SECURE Act 2.0

Last updated: January 2026
"""

from enum import Enum

# =============================================================================
# FILING STATUS
# =============================================================================


class FilingStatus(Enum):
    """Tax filing status."""

    SINGLE = "single"
    MARRIED_FILING_JOINTLY = "married_filing_jointly"
    MARRIED_FILING_SEPARATELY = "married_filing_separately"
    HEAD_OF_HOUSEHOLD = "head_of_household"


# =============================================================================
# FEDERAL TAX BRACKETS 2026 (Married Filing Jointly)
# =============================================================================
# Updated per IRS Rev. Proc. 2025-XX and OBBBA (One Big Beautiful Bill Act)
# OBBBA: 4% inflation adjustment for 10%/12% brackets, 2.3% for higher brackets

TAX_BRACKETS_2026_FEDERAL_MFJ = [
    (24_800, 0.10),  # 10% on first $24,800
    (100_800, 0.12),  # 12% on $24,801 - $100,800
    (211_400, 0.22),  # 22% on $100,801 - $211,400
    (403_550, 0.24),  # 24% on $211,401 - $403,550
    (512_450, 0.32),  # 32% on $403,551 - $512,450
    (768_700, 0.35),  # 35% on $512,451 - $768,700
    (float("inf"), 0.37),  # 37% on $768,701+
]

# Standard deduction 2026 MFJ
STANDARD_DEDUCTION_2026_FEDERAL_MFJ = 32_200

# Additional standard deduction for seniors (65+)
ADDITIONAL_DEDUCTION_SENIOR_SINGLE = 2_050
ADDITIONAL_DEDUCTION_SENIOR_MFJ = 1_650  # Per spouse

# OBBBA Senior Deduction (2025-2028) - phases out at $75K MAGI
OBBBA_SENIOR_DEDUCTION = 6_000
OBBBA_SENIOR_PHASEOUT_START = 75_000


# =============================================================================
# CALIFORNIA TAX BRACKETS 2026 (Married Filing Jointly)
# =============================================================================
# Source: FTB Publication 1031 (2026 estimates based on inflation adjustment)
# Note: CA taxes capital gains as ordinary income (no preferential rate)

TAX_BRACKETS_2026_CA_MFJ = [
    (22_106, 0.01),  # 1% on first $22,106
    (52_376, 0.02),  # 2% on $22,107 - $52,376
    (82_646, 0.04),  # 4% on $52,377 - $82,646
    (114_692, 0.06),  # 6% on $82,647 - $114,692
    (144_960, 0.08),  # 8% on $114,693 - $144,960
    (737_424, 0.093),  # 9.3% on $144,961 - $737,424
    (884_908, 0.103),  # 10.3% on $737,425 - $884,908
    (1_474_848, 0.113),  # 11.3% on $884,909 - $1,474,848
    (float("inf"), 0.123),  # 12.3% on $1,474,849+
]

# California standard deduction 2026 MFJ
STANDARD_DEDUCTION_2026_CA_MFJ = 11_080

# California Mental Health Services Tax (Proposition 63)
# Additional 1% on taxable income over $1M = effective 13.3% top rate
CA_MENTAL_HEALTH_TAX_THRESHOLD = 1_000_000
CA_MENTAL_HEALTH_TAX_RATE = 0.01


# =============================================================================
# LONG-TERM CAPITAL GAINS 2026 (Married Filing Jointly)
# =============================================================================
# Federal LTCG rates: 0%, 15%, 20% based on taxable income
# Note: State (CA) taxes all gains as ordinary income

LTCG_BRACKETS_2026_MFJ = [
    (96_700, 0.00),  # 0% on first $96,700
    (600_050, 0.15),  # 15% on $96,701 - $600,050
    (float("inf"), 0.20),  # 20% on $600,051+
]


# =============================================================================
# NET INVESTMENT INCOME TAX (NIIT)
# =============================================================================
# 3.8% surtax on net investment income for high earners
# Threshold is NOT inflation-adjusted (set by ACA)

NIIT_THRESHOLD_MFJ = 250_000
NIIT_THRESHOLD_SINGLE = 200_000
NIIT_RATE = 0.038


# =============================================================================
# IRMAA THRESHOLDS 2026 (Medicare Premium Surcharges)
# =============================================================================
# Income-Related Monthly Adjustment Amount
# Based on MAGI from 2 years prior (2024 tax return for 2026 premiums)
# Source: CMS-8089-N, CMS-8090-N (November 2025)
# Standard Part B premium 2026: $202.90/month

# Format: (lower_bound, upper_bound, annual_part_b_surcharge)
IRMAA_THRESHOLDS_2026_MFJ = [
    (0, 218_000, 0),  # Standard premium, no surcharge
    (218_000, 274_000, 973.92),  # 1.4x: $81.16/mo x 12
    (274_000, 342_000, 2_434.80),  # 2.0x: $202.90/mo x 12
    (342_000, 410_000, 3_895.68),  # 2.6x: $324.64/mo x 12
    (410_000, 750_000, 5_356.56),  # 3.2x: $446.38/mo x 12
    (750_000, float("inf"), 5_843.52),  # 3.4x: $486.96/mo x 12
]

# Part D IRMAA surcharges (same income brackets, different amounts)
IRMAA_PART_D_MONTHLY_2026_MFJ = [
    (0, 218_000, 0),  # Standard
    (218_000, 274_000, 13.70),  # +$13.70/mo
    (274_000, 342_000, 35.30),  # +$35.30/mo
    (342_000, 410_000, 56.90),  # +$56.90/mo
    (410_000, 750_000, 78.50),  # +$78.50/mo
    (750_000, float("inf"), 85.80),  # +$85.80/mo
]

# Single filer IRMAA thresholds (exactly half of MFJ)
IRMAA_THRESHOLDS_2026_SINGLE = [
    (0, 109_000, 0),
    (109_000, 137_000, 973.92),
    (137_000, 171_000, 2_434.80),
    (171_000, 205_000, 3_895.68),
    (205_000, 500_000, 5_356.56),
    (500_000, float("inf"), 5_843.52),
]


# =============================================================================
# HEALTHCARE COSTS 2026
# =============================================================================
# Annual healthcare costs by age bracket (in today's dollars)
# Updated Jan 2026: ACA enhanced credits EXPIRED Dec 31, 2025

HEALTHCARE_COSTS_2026 = {
    (0, 64): 23_000,  # Pre-Medicare ACA (no enhanced credits) for couple
    (65, 999): 12_000,  # Medicare: Part B + Part D + Medigap for couple
}

# Medicare Part B annual deductible 2026
MEDICARE_PART_B_DEDUCTIBLE_2026 = 283


# =============================================================================
# RMD (REQUIRED MINIMUM DISTRIBUTIONS) - SECURE Act 2.0
# =============================================================================
# IRS Uniform Lifetime Table (updated Jan 2022, per IRS Publication 590-B)
# Key: Age, Value: Distribution Period (life expectancy divisor)
# RMD = account_balance / divisor

RMD_UNIFORM_LIFETIME_TABLE = {
    72: 27.4,
    73: 26.5,
    74: 25.5,
    75: 24.6,
    76: 23.7,
    77: 22.9,
    78: 22.0,
    79: 21.1,
    80: 20.2,
    81: 19.4,
    82: 18.5,
    83: 17.7,
    84: 16.8,
    85: 16.0,
    86: 15.2,
    87: 14.4,
    88: 13.7,
    89: 12.9,
    90: 12.2,
    91: 11.5,
    92: 10.8,
    93: 10.1,
    94: 9.5,
    95: 8.9,
    96: 8.4,
    97: 7.8,
    98: 7.3,
    99: 6.8,
    100: 6.4,
    101: 6.0,
    102: 5.6,
    103: 5.2,
    104: 4.9,
    105: 4.6,
    106: 4.3,
    107: 4.1,
    108: 3.9,
    109: 3.7,
    110: 3.5,
    111: 3.4,
    112: 3.3,
    113: 3.1,
    114: 3.0,
    115: 2.9,
    116: 2.8,
    117: 2.7,
    118: 2.5,
    119: 2.3,
    120: 2.0,
}

# RMD start age by birth year (SECURE Act 2.0)
RMD_START_AGE_BY_BIRTH_YEAR = {
    "pre_1951": 72,
    "1951_1959": 73,
    "post_1959": 75,
}


# =============================================================================
# ACA THRESHOLDS 2026
# =============================================================================
# ACA subsidy cliff at 400% Federal Poverty Level

ACA_FPL_400_2026_SINGLE = 62_400  # Single person
ACA_FPL_400_2026_COUPLE = 84_640  # Family of 2 (couple)
ACA_FPL_400_2026_FAMILY_4 = 129_600  # Family of 4


# =============================================================================
# ALTERNATIVE MINIMUM TAX (AMT) 2026
# =============================================================================

AMT_EXEMPTION_2026_SINGLE = 90_100
AMT_EXEMPTION_2026_MFJ = 140_200
AMT_PHASEOUT_START_SINGLE = 500_000
AMT_PHASEOUT_START_MFJ = 1_000_000


# =============================================================================
# ESTATE TAX 2026
# =============================================================================

ESTATE_TAX_EXCLUSION_2026 = 15_000_000


# =============================================================================
# BACKWARDS COMPATIBILITY ALIASES
# =============================================================================

# Federal brackets
TAX_BRACKETS_FEDERAL_MFJ = TAX_BRACKETS_2026_FEDERAL_MFJ
TAX_BRACKETS_MFJ = TAX_BRACKETS_2026_FEDERAL_MFJ
STANDARD_DEDUCTION_MFJ = STANDARD_DEDUCTION_2026_FEDERAL_MFJ

# California brackets
TAX_BRACKETS_CA_MFJ = TAX_BRACKETS_2026_CA_MFJ
CA_STANDARD_DEDUCTION_MFJ = STANDARD_DEDUCTION_2026_CA_MFJ

# IRMAA
IRMAA_THRESHOLDS_MFJ = IRMAA_THRESHOLDS_2026_MFJ

# RMD (alias used by withdrawal.py)
UNIFORM_LIFETIME_TABLE = RMD_UNIFORM_LIFETIME_TABLE

# LTCG
LTCG_BRACKETS_MFJ = LTCG_BRACKETS_2026_MFJ

# Healthcare
HEALTHCARE_COSTS = HEALTHCARE_COSTS_2026
