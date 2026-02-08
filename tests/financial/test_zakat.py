"""Tests for roshni.financial.calculators.zakat."""

import pytest

from roshni.financial.calculators.zakat import (
    PORTFOLIO_CONFIGS,
    Asset,
    AssetCategory,
    AssetClassZakatCalculator,
    InvestmentAssetClass,
    PortfolioAllocation,
    RetirementZakatApproach,
    ZakatCalculator,
    ZakatConfig,
    ZakatStrategy,
    calculate_retirement_goal_with_zakat,
    calculate_zakat,
    calculate_zakat_for_simulation,
    check_nisab,
    classify_assets,
)


@pytest.fixture
def sample_assets():
    return [
        Asset("Checking", AssetCategory.CASH, 50_000),
        Asset("Savings", AssetCategory.CASH, 100_000),
        Asset("Brokerage", AssetCategory.STOCKS, 500_000),
        Asset("401(k)", AssetCategory.RETIREMENT_401K, 800_000),
        Asset("Roth IRA", AssetCategory.RETIREMENT_ROTH, 200_000),
        Asset("Home", AssetCategory.PRIMARY_RESIDENCE, 1_500_000),
        Asset("Credit Cards", AssetCategory.SHORT_TERM_DEBT, 5_000),
        Asset("Mortgage", AssetCategory.LONG_TERM_DEBT, 400_000),
    ]


class TestClassifyAssets:
    def test_basic_classification(self, sample_assets):
        config = ZakatConfig()
        result = classify_assets(sample_assets, config)

        assert result.zakatable_total == 650_000  # 50k + 100k + 500k
        assert result.retirement_total == 1_000_000  # 800k + 200k
        assert result.non_zakatable_total == 1_500_000  # home
        assert result.debt_total == 405_000  # 5k + 400k

    def test_on_withdrawal_excludes_retirement(self, sample_assets):
        config = ZakatConfig(retirement_approach=RetirementZakatApproach.ON_WITHDRAWAL)
        result = classify_assets(sample_assets, config)
        assert result.retirement_zakatable == 0.0

    def test_full_value_includes_retirement(self, sample_assets):
        config = ZakatConfig(retirement_approach=RetirementZakatApproach.FULL_VALUE)
        result = classify_assets(sample_assets, config)
        assert result.retirement_zakatable == 1_000_000

    def test_accessible_portion(self, sample_assets):
        config = ZakatConfig(
            retirement_approach=RetirementZakatApproach.ACCESSIBLE_PORTION,
            early_withdrawal_penalty=0.10,
        )
        result = classify_assets(sample_assets, config)
        assert result.retirement_zakatable == 900_000  # 1M * 0.90

    def test_debt_deduction(self, sample_assets):
        config = ZakatConfig(deduct_short_term_debt=True, deduct_long_term_debt=True)
        result = classify_assets(sample_assets, config)
        # Short-term: $5,000, Long-term: $400,000/30 = ~$13,333
        assert result.deductible_debt == pytest.approx(5_000 + 400_000 / 30, rel=1e-2)


class TestCheckNisab:
    def test_meets_nisab(self):
        config = ZakatConfig(gold_price_per_gram=100.0)
        # Nisab = 85 * 100 = 8,500
        meets, threshold = check_nisab(10_000, config)
        assert meets is True
        assert threshold == 8_500

    def test_below_nisab(self):
        config = ZakatConfig(gold_price_per_gram=100.0)
        meets, _threshold = check_nisab(5_000, config)
        assert meets is False


class TestCalculateZakat:
    def test_basic_calculation(self):
        config = ZakatConfig()
        result = calculate_zakat(100_000, config)
        assert result == 2_500.0  # 2.5% of 100k

    def test_zero_wealth(self):
        config = ZakatConfig()
        result = calculate_zakat(0, config)
        assert result == 0.0

    def test_negative_wealth(self):
        config = ZakatConfig()
        result = calculate_zakat(-1000, config)
        assert result == 0.0

    def test_decimal_precision(self):
        config = ZakatConfig()
        result = calculate_zakat(123_456.78, config)
        # 123,456.78 * 0.025 = 3,086.4195 -> rounds to 3,086.42
        assert result == 3_086.42


class TestZakatCalculator:
    def test_full_calculation(self, sample_assets):
        calc = ZakatCalculator(ZakatConfig(retirement_approach=RetirementZakatApproach.ON_WITHDRAWAL))
        result = calc.calculate(sample_assets)

        assert result.meets_nisab is True
        assert result.zakat_rate == 0.025
        assert result.zakat_due > 0

    def test_compare_approaches(self, sample_assets):
        calc = ZakatCalculator()
        results = calc.calculate_with_approaches(sample_assets)

        assert "full_value" in results
        assert "on_withdrawal" in results
        assert "accessible_portion" in results
        assert "liquidation_value" in results

        # Full value should yield highest zakat
        assert results["full_value"].zakat_due >= results["on_withdrawal"].zakat_due

    def test_estimate_from_ledger(self):
        calc = ZakatCalculator()
        result = calc.estimate_from_ledger(
            retirement_stash=1_000_000,
            cash_investments=500_000,
            short_term_debt=5_000,
            mortgage_balance=300_000,
        )
        assert result.meets_nisab is True
        assert result.zakat_due > 0

    def test_update_gold_price(self):
        calc = ZakatCalculator()
        calc.update_gold_price(200.0)
        assert calc.config.gold_price_per_gram == 200.0
        assert calc.config.nisab_threshold_usd == 85 * 200

    def test_update_gold_price_negative_raises(self):
        calc = ZakatCalculator()
        with pytest.raises(ValueError, match="positive"):
            calc.update_gold_price(-10)

    def test_to_dict(self, sample_assets):
        calc = ZakatCalculator()
        result = calc.calculate(sample_assets)
        d = result.to_dict()
        assert "calculation_date" in d
        assert "nisab" in d
        assert "zakat" in d


class TestAssetClassZakatCalculator:
    def test_yale_portfolio(self):
        calc = AssetClassZakatCalculator()
        result = calc.calculate(10_000_000, "yale")

        # Yale-style: ~1.73% effective rate
        assert result.total_zakat == pytest.approx(172_500, rel=0.01)
        assert result.effective_rate == pytest.approx(0.01725, rel=0.01)
        assert result.savings_vs_flat > 0

    def test_full_strategy_flat_rate(self):
        calc = AssetClassZakatCalculator(strategy=ZakatStrategy.FULL)
        result = calc.calculate(1_000_000, "60_40")

        # FULL strategy = 2.5% on everything
        assert result.total_zakat == pytest.approx(25_000, rel=0.01)

    def test_all_equity_lowest_rate(self):
        calc = AssetClassZakatCalculator()
        result = calc.calculate(1_000_000, "all_equity")

        # All equity = 0.75% rate
        assert result.effective_rate == pytest.approx(0.0075, rel=0.01)

    def test_unknown_portfolio_raises(self):
        calc = AssetClassZakatCalculator()
        with pytest.raises(ValueError, match="Unknown portfolio"):
            calc.calculate(1_000_000, "nonexistent")

    def test_from_actual_holdings(self):
        calc = AssetClassZakatCalculator()
        holdings = {"US Stocks": 600_000, "Bond Funds": 400_000}
        result = calc.calculate_from_actual_holdings(holdings)

        assert result.portfolio_value == 1_000_000
        assert result.total_zakat > 0

    def test_from_actual_holdings_empty(self):
        calc = AssetClassZakatCalculator()
        result = calc.calculate_from_actual_holdings({})
        assert result.total_zakat == 0


class TestPortfolioAllocation:
    def test_effective_rate(self):
        alloc = PORTFOLIO_CONFIGS["60_40"]
        rate = alloc.effective_zakat_rate()
        # 60% equity @ 0.75% + 40% bonds @ 2.5% = 1.45%
        assert rate == pytest.approx(0.0145, rel=0.01)

    def test_invalid_weights_raises(self):
        with pytest.raises(ValueError, match=r"sum to 1\.0"):
            PortfolioAllocation(
                name="Bad",
                allocations={InvestmentAssetClass.US_EQUITY: 0.5},
            )


class TestSimulationHelper:
    def test_standard_strategy(self):
        zakat = calculate_zakat_for_simulation(1_000_000, equity_ratio=0.70)
        # 700k * 0.0075 + 300k * 0.025 = 5,250 + 7,500 = 12,750
        assert zakat == pytest.approx(12_750, rel=0.01)

    def test_full_strategy(self):
        zakat = calculate_zakat_for_simulation(1_000_000, strategy=ZakatStrategy.FULL)
        assert zakat == pytest.approx(25_000, rel=0.01)

    def test_rate_override(self):
        zakat = calculate_zakat_for_simulation(1_000_000, zakat_rate_override=0.01)
        assert zakat == pytest.approx(10_000, rel=0.01)

    def test_zero_portfolio(self):
        assert calculate_zakat_for_simulation(0) == 0.0


class TestRetirementGoalWithZakat:
    def test_basic_goal(self):
        result = calculate_retirement_goal_with_zakat(100_000)
        assert result["goal_no_zakat"] == pytest.approx(2_000_000)
        assert result["goal_with_zakat"] > result["goal_no_zakat"]

    def test_with_allocation(self):
        result = calculate_retirement_goal_with_zakat(100_000, allocation="all_equity")
        # All equity = 0.75% zakat, so effective spending rate = 5% - 0.75% = 4.25%
        assert result["effective_zakat_rate"] == pytest.approx(0.0075, rel=0.01)
