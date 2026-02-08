"""Tests for roshni.financial.calculators.mortgage."""

import pytest

from roshni.financial.calculators.mortgage import (
    MortgageTerms,
    calculate_interest_only_payment,
    calculate_lump_sum_payoff,
    calculate_monthly_payment,
    compare_scenarios,
    project_balance_with_prepay,
)


class TestMonthlyPayment:
    def test_standard_mortgage(self):
        # $500k, 7%, 30 years -> ~$3,327/mo
        payment = calculate_monthly_payment(500_000, 0.07, 30)
        assert payment == pytest.approx(3_327, rel=0.01)

    def test_zero_rate(self):
        assert calculate_monthly_payment(500_000, 0, 30) == 0

    def test_zero_principal(self):
        assert calculate_monthly_payment(0, 0.07, 30) == 0


class TestInterestOnlyPayment:
    def test_basic(self):
        # $500k at 3% = $15,000/year = $1,250/mo
        payment = calculate_interest_only_payment(500_000, 0.03)
        assert payment == pytest.approx(1_250, rel=0.01)


class TestProjectBalance:
    def test_no_prepay_interest_only(self):
        balance, interest = project_balance_with_prepay(
            starting_balance=500_000,
            annual_rate=0.03,
            months=12,
            monthly_prepay=0,
            is_interest_only=True,
        )
        # Interest-only: balance stays same
        assert balance == 500_000
        # Interest: ~$15,000/year
        assert interest == pytest.approx(15_000, rel=0.01)

    def test_with_prepay(self):
        balance, _ = project_balance_with_prepay(
            starting_balance=500_000,
            annual_rate=0.03,
            months=12,
            monthly_prepay=5_000,
            is_interest_only=True,
        )
        # $5k/mo * 12 = $60k paid off
        assert balance == pytest.approx(440_000, rel=0.01)

    def test_prepay_exceeds_balance(self):
        balance, _ = project_balance_with_prepay(
            starting_balance=10_000,
            annual_rate=0.03,
            months=12,
            monthly_prepay=5_000,
        )
        assert balance == 0


class TestCompareScenarios:
    def test_multiple_scenarios(self):
        terms = MortgageTerms(balance=500_000, current_rate=0.0245, reset_year=2030, reset_rate=0.07)
        comparison = compare_scenarios(terms, [0, 2_000, 5_000], current_year=2026)

        assert len(comparison.scenarios) == 3
        assert comparison.scenarios[0].name == "Do Nothing"
        # More prepayment = less total interest
        assert comparison.scenarios[2].total_interest < comparison.scenarios[0].total_interest

    def test_format_table(self):
        terms = MortgageTerms(balance=500_000, current_rate=0.03, reset_year=2030)
        comparison = compare_scenarios(terms, [0, 2_000], current_year=2026)
        table = comparison.format_table()
        assert "Mortgage Scenario Comparison" in table
        assert "Do Nothing" in table


class TestLumpSumPayoff:
    def test_basic_payoff(self):
        terms = MortgageTerms(balance=500_000, current_rate=0.03, reset_year=2030, reset_rate=0.07)
        result = calculate_lump_sum_payoff(terms, payoff_year=2028, current_year=2026)

        assert result["payoff_amount"] == 500_000
        assert result["monthly_savings"] > 0
        assert result["avoided_interest"] > 0
