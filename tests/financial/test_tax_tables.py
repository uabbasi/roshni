"""Tests for roshni.financial.calculators.tax_tables."""

from roshni.financial.calculators.tax_tables import (
    ESTATE_TAX_EXCLUSION_2026,
    HEALTHCARE_COSTS_2026,
    IRMAA_THRESHOLDS_2026_MFJ,
    LTCG_BRACKETS_2026_MFJ,
    NIIT_RATE,
    RMD_UNIFORM_LIFETIME_TABLE,
    STANDARD_DEDUCTION_2026_FEDERAL_MFJ,
    TAX_BRACKETS_2026_CA_MFJ,
    TAX_BRACKETS_2026_FEDERAL_MFJ,
    FilingStatus,
)


class TestFederalBrackets:
    def test_bracket_count(self):
        assert len(TAX_BRACKETS_2026_FEDERAL_MFJ) == 7

    def test_brackets_ascending(self):
        thresholds = [b[0] for b in TAX_BRACKETS_2026_FEDERAL_MFJ]
        assert thresholds == sorted(thresholds)

    def test_rates_ascending(self):
        rates = [b[1] for b in TAX_BRACKETS_2026_FEDERAL_MFJ]
        assert rates == sorted(rates)

    def test_top_rate_is_37_pct(self):
        assert TAX_BRACKETS_2026_FEDERAL_MFJ[-1][1] == 0.37

    def test_standard_deduction(self):
        assert STANDARD_DEDUCTION_2026_FEDERAL_MFJ == 32_200


class TestCaliforniaBrackets:
    def test_bracket_count(self):
        assert len(TAX_BRACKETS_2026_CA_MFJ) == 9

    def test_top_rate_is_12_3_pct(self):
        assert TAX_BRACKETS_2026_CA_MFJ[-1][1] == 0.123


class TestLTCG:
    def test_zero_rate_exists(self):
        assert LTCG_BRACKETS_2026_MFJ[0][1] == 0.00


class TestIRMAA:
    def test_standard_is_zero(self):
        assert IRMAA_THRESHOLDS_2026_MFJ[0][2] == 0

    def test_thresholds_ascending(self):
        lowers = [t[0] for t in IRMAA_THRESHOLDS_2026_MFJ]
        assert lowers == sorted(lowers)


class TestRMD:
    def test_age_72_exists(self):
        assert 72 in RMD_UNIFORM_LIFETIME_TABLE

    def test_age_120_exists(self):
        assert 120 in RMD_UNIFORM_LIFETIME_TABLE

    def test_divisor_decreases_with_age(self):
        assert RMD_UNIFORM_LIFETIME_TABLE[72] > RMD_UNIFORM_LIFETIME_TABLE[100]


class TestOtherConstants:
    def test_niit_rate(self):
        assert NIIT_RATE == 0.038

    def test_estate_exclusion(self):
        assert ESTATE_TAX_EXCLUSION_2026 == 15_000_000

    def test_healthcare_costs(self):
        assert (0, 64) in HEALTHCARE_COSTS_2026
        assert (65, 999) in HEALTHCARE_COSTS_2026

    def test_filing_status_enum(self):
        assert FilingStatus.MARRIED_FILING_JOINTLY.value == "married_filing_jointly"
