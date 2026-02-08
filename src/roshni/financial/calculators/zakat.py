"""
Zakat Calculator — Islamic finance calculations for annual obligatory charity.

Implements:
- Asset classification (zakatable vs non-zakatable)
- Nisab threshold checking (gold standard)
- 2.5% zakat calculation on eligible wealth
- Retirement account handling per scholarly guidance
- Asset-class-aware portfolio zakat rates

Two zakat strategies are supported:
- FULL: 2.5% flat rate on all assets (most conservative interpretation)
- STANDARD: Asset-class-aware rates (0.75% for equities, 2.5% for others)

All arithmetic is done in Python — no LLM math.
"""

from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum

from loguru import logger

# === Zakat Strategy Constants ===

BASE_ZAKAT_RATE = 0.025  # 2.5% - standard zakat rate on zakatable wealth
ZAKATABLE_EQUITY_RATIO = 0.30  # 30% of company assets are zakatable (safety margin)
ZAKAT_RATE_EQUITY_STANDARD = BASE_ZAKAT_RATE * ZAKATABLE_EQUITY_RATIO  # 0.75% for STANDARD strategy


class ZakatStrategy(Enum):
    """Zakat calculation strategies for investment portfolios.

    FULL: 2.5% flat rate on all assets (most conservative).
    STANDARD: Asset-class-aware rates using 30% underlying assets ratio.
    """

    FULL = "full"
    STANDARD = "standard"


class AssetCategory(Enum):
    """Categories of assets for zakat classification."""

    # Zakatable assets (2.5% annual)
    CASH = "cash"
    STOCKS = "stocks"
    MUTUAL_FUNDS = "mutual_funds"
    BONDS = "bonds"
    GOLD = "gold"
    SILVER = "silver"
    BUSINESS_INVENTORY = "business_inventory"
    RECEIVABLES = "receivables"

    # Non-zakatable assets
    PRIMARY_RESIDENCE = "primary_residence"
    PERSONAL_VEHICLE = "personal_vehicle"
    PERSONAL_ITEMS = "personal_items"
    RETIREMENT_401K = "retirement_401k"
    RETIREMENT_IRA = "retirement_ira"
    RETIREMENT_ROTH = "retirement_roth"

    # Debts (reduce zakatable base)
    SHORT_TERM_DEBT = "short_term_debt"
    LONG_TERM_DEBT = "long_term_debt"


# Default classification sets
ZAKATABLE_CATEGORIES = {
    AssetCategory.CASH,
    AssetCategory.STOCKS,
    AssetCategory.MUTUAL_FUNDS,
    AssetCategory.BONDS,
    AssetCategory.GOLD,
    AssetCategory.SILVER,
    AssetCategory.BUSINESS_INVENTORY,
    AssetCategory.RECEIVABLES,
}

NON_ZAKATABLE_CATEGORIES = {
    AssetCategory.PRIMARY_RESIDENCE,
    AssetCategory.PERSONAL_VEHICLE,
    AssetCategory.PERSONAL_ITEMS,
}

RETIREMENT_CATEGORIES = {
    AssetCategory.RETIREMENT_401K,
    AssetCategory.RETIREMENT_IRA,
    AssetCategory.RETIREMENT_ROTH,
}

DEBT_CATEGORIES = {
    AssetCategory.SHORT_TERM_DEBT,
    AssetCategory.LONG_TERM_DEBT,
}


class RetirementZakatApproach(Enum):
    """Different scholarly approaches to retirement account zakat."""

    FULL_VALUE = "full_value"  # Pay on full value annually
    ON_WITHDRAWAL = "on_withdrawal"  # Pay only when withdrawn
    ACCESSIBLE_PORTION = "accessible_portion"  # Vested amount minus penalty
    LIQUIDATION_VALUE = "liquidation_value"  # Value minus taxes/penalties


@dataclass
class Asset:
    """Individual asset for zakat calculation."""

    name: str
    category: AssetCategory
    value: float  # Current market value
    acquisition_date: date | None = None
    notes: str = ""


@dataclass
class ZakatConfig:
    """Configuration for zakat calculations."""

    zakat_rate: float = 0.025
    nisab_gold_grams: float = 85.0  # ~3 oz, traditional standard
    gold_price_per_gram: float = 144.0  # Fallback: ~$144/gram as of Jan 2026
    retirement_approach: RetirementZakatApproach = RetirementZakatApproach.ON_WITHDRAWAL
    early_withdrawal_penalty: float = 0.10
    retirement_tax_rate: float = 0.30
    deduct_short_term_debt: bool = True
    deduct_long_term_debt: bool = True

    @property
    def nisab_threshold_usd(self) -> float:
        """Calculate nisab threshold in USD based on gold price."""
        return self.nisab_gold_grams * self.gold_price_per_gram


@dataclass
class ZakatClassification:
    """Result of classifying assets for zakat."""

    zakatable_assets: list[Asset] = field(default_factory=list)
    non_zakatable_assets: list[Asset] = field(default_factory=list)
    retirement_assets: list[Asset] = field(default_factory=list)
    debts: list[Asset] = field(default_factory=list)

    zakatable_total: float = 0.0
    non_zakatable_total: float = 0.0
    retirement_total: float = 0.0
    retirement_zakatable: float = 0.0
    debt_total: float = 0.0
    deductible_debt: float = 0.0

    @property
    def net_zakatable_wealth(self) -> float:
        """Total zakatable wealth after debt deductions."""
        return self.zakatable_total + self.retirement_zakatable - self.deductible_debt


@dataclass
class ZakatResult:
    """Complete result of zakat calculation."""

    classification: ZakatClassification
    nisab_threshold: float
    meets_nisab: bool
    zakatable_wealth: float
    zakat_rate: float
    zakat_due: float
    retirement_approach: str
    gold_price_used: float
    calculation_date: date

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "calculation_date": self.calculation_date.isoformat(),
            "nisab": {
                "threshold_usd": round(self.nisab_threshold, 2),
                "meets_nisab": self.meets_nisab,
            },
            "classification": {
                "zakatable_total": round(self.classification.zakatable_total, 2),
                "non_zakatable_total": round(self.classification.non_zakatable_total, 2),
                "retirement_total": round(self.classification.retirement_total, 2),
                "retirement_zakatable": round(self.classification.retirement_zakatable, 2),
                "debt_deduction": round(self.classification.deductible_debt, 2),
            },
            "zakat": {
                "zakatable_wealth": round(self.zakatable_wealth, 2),
                "rate": self.zakat_rate,
                "due": round(self.zakat_due, 2),
            },
            "config": {
                "retirement_approach": self.retirement_approach,
                "gold_price_per_gram": self.gold_price_used,
            },
        }


def classify_assets(
    assets: list[Asset],
    config: ZakatConfig,
) -> ZakatClassification:
    """Classify assets into zakatable, non-zakatable, retirement, and debt categories."""
    result = ZakatClassification()

    for asset in assets:
        if asset.category in ZAKATABLE_CATEGORIES:
            result.zakatable_assets.append(asset)
            result.zakatable_total += asset.value

        elif asset.category in NON_ZAKATABLE_CATEGORIES:
            result.non_zakatable_assets.append(asset)
            result.non_zakatable_total += asset.value

        elif asset.category in RETIREMENT_CATEGORIES:
            result.retirement_assets.append(asset)
            result.retirement_total += asset.value

        elif asset.category in DEBT_CATEGORIES:
            result.debts.append(asset)
            result.debt_total += asset.value

    result.retirement_zakatable = _calculate_retirement_zakatable(result.retirement_total, config)
    result.deductible_debt = _calculate_deductible_debt(result.debts, config)

    return result


def _calculate_retirement_zakatable(
    retirement_total: float,
    config: ZakatConfig,
) -> float:
    """Calculate zakatable portion of retirement accounts based on chosen approach."""
    match config.retirement_approach:
        case RetirementZakatApproach.FULL_VALUE:
            return retirement_total
        case RetirementZakatApproach.ON_WITHDRAWAL:
            return 0.0
        case RetirementZakatApproach.ACCESSIBLE_PORTION:
            return retirement_total * (1 - config.early_withdrawal_penalty)
        case RetirementZakatApproach.LIQUIDATION_VALUE:
            after_penalty = retirement_total * (1 - config.early_withdrawal_penalty)
            return after_penalty * (1 - config.retirement_tax_rate)
        case _:
            logger.warning(f"Unknown retirement approach: {config.retirement_approach}")
            return 0.0


def _calculate_deductible_debt(
    debts: list[Asset],
    config: ZakatConfig,
) -> float:
    """Calculate debt that can be deducted from zakatable wealth."""
    deductible = 0.0

    for debt in debts:
        if debt.category == AssetCategory.SHORT_TERM_DEBT:
            if config.deduct_short_term_debt:
                deductible += debt.value

        elif debt.category == AssetCategory.LONG_TERM_DEBT:
            if config.deduct_long_term_debt:
                annual_portion = debt.value / 30
                deductible += annual_portion

    return deductible


def check_nisab(
    zakatable_wealth: float,
    config: ZakatConfig,
) -> tuple[bool, float]:
    """Check if zakatable wealth meets the nisab threshold.

    Returns:
        Tuple of (meets_nisab, nisab_threshold_usd)
    """
    threshold = config.nisab_threshold_usd
    meets = zakatable_wealth >= threshold

    logger.debug(
        f"Nisab check: ${zakatable_wealth:,.2f} vs threshold ${threshold:,.2f} "
        f"({config.nisab_gold_grams}g gold @ ${config.gold_price_per_gram}/g)"
    )

    return (meets, threshold)


def calculate_zakat(
    zakatable_wealth: float,
    config: ZakatConfig,
) -> float:
    """Calculate zakat due on zakatable wealth.

    Uses Decimal for precise calculation, returns float.
    """
    if zakatable_wealth <= 0:
        return 0.0

    wealth = Decimal(str(zakatable_wealth))
    rate = Decimal(str(config.zakat_rate))
    zakat = (wealth * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    return float(zakat)


class ZakatCalculator:
    """Calculator for Islamic obligatory charity (zakat).

    Handles asset classification, nisab checking, and zakat calculation
    with configurable retirement account handling.
    """

    def __init__(self, config: ZakatConfig | None = None):
        self.config = config or ZakatConfig()

    def calculate(
        self,
        assets: list[Asset],
        config: ZakatConfig | None = None,
    ) -> ZakatResult:
        """Perform complete zakat calculation."""
        calc_config = config or self.config

        classification = classify_assets(assets, calc_config)
        zakatable_wealth = classification.net_zakatable_wealth
        meets_nisab, threshold = check_nisab(zakatable_wealth, calc_config)

        zakat_due = 0.0
        if meets_nisab:
            zakat_due = calculate_zakat(zakatable_wealth, calc_config)
        else:
            logger.info(f"Wealth ${zakatable_wealth:,.2f} below nisab ${threshold:,.2f}, no zakat due")

        return ZakatResult(
            classification=classification,
            nisab_threshold=threshold,
            meets_nisab=meets_nisab,
            zakatable_wealth=zakatable_wealth,
            zakat_rate=calc_config.zakat_rate,
            zakat_due=zakat_due,
            retirement_approach=calc_config.retirement_approach.value,
            gold_price_used=calc_config.gold_price_per_gram,
            calculation_date=date.today(),
        )

    def calculate_with_approaches(
        self,
        assets: list[Asset],
    ) -> dict[str, ZakatResult]:
        """Calculate zakat using all retirement approaches for comparison."""
        results = {}

        for approach in RetirementZakatApproach:
            config = ZakatConfig(
                gold_price_per_gram=self.config.gold_price_per_gram,
                retirement_approach=approach,
                early_withdrawal_penalty=self.config.early_withdrawal_penalty,
                retirement_tax_rate=self.config.retirement_tax_rate,
                deduct_short_term_debt=self.config.deduct_short_term_debt,
                deduct_long_term_debt=self.config.deduct_long_term_debt,
            )
            results[approach.value] = self.calculate(assets, config)

        return results

    def update_gold_price(self, price_per_gram: float) -> None:
        """Update gold price used for nisab calculation."""
        if price_per_gram <= 0:
            raise ValueError("Gold price must be positive")

        old_price = self.config.gold_price_per_gram
        self.config.gold_price_per_gram = price_per_gram

        logger.info(
            f"Gold price updated: ${old_price:.2f}/g -> ${price_per_gram:.2f}/g "
            f"(Nisab: ${self.config.nisab_threshold_usd:,.2f})"
        )

    def estimate_from_ledger(
        self,
        retirement_stash: float,
        cash_investments: float,
        short_term_debt: float = 0,
        mortgage_balance: float = 0,
    ) -> ZakatResult:
        """Quick zakat estimate from aggregate values."""
        assets = [
            Asset(
                name="Retirement Accounts",
                category=AssetCategory.RETIREMENT_401K,
                value=retirement_stash,
            ),
            Asset(
                name="Cash & Investments",
                category=AssetCategory.STOCKS,
                value=cash_investments,
            ),
        ]

        if short_term_debt > 0:
            assets.append(
                Asset(
                    name="Short-term Debt",
                    category=AssetCategory.SHORT_TERM_DEBT,
                    value=short_term_debt,
                )
            )

        if mortgage_balance > 0:
            assets.append(
                Asset(
                    name="Mortgage",
                    category=AssetCategory.LONG_TERM_DEBT,
                    value=mortgage_balance,
                )
            )

        return self.calculate(assets)


# ============================================================================
# ASSET-CLASS-SPECIFIC ZAKAT RATES
# ============================================================================
# Based on AAOIFI guidelines and scholarly consensus:
# - Equity: 0.75% (30% x 2.5% - only zakatable portion of company assets)
# - Fixed Income/Cash/Crypto/Gold: 2.5%
# - Real Estate/QOZ: 0% (exempt - not held for trade)
# - Alternatives (NAV known): 2.5%
# - Alternatives (NAV unknown): 10% (conservative estimate)
# ============================================================================


class InvestmentAssetClass(Enum):
    """Investment asset classes with their zakat treatment."""

    US_EQUITY = "us_equity"
    INTL_EQUITY = "intl_equity"
    FIXED_INCOME = "fixed_income"
    CASH = "cash"
    REAL_ESTATE = "real_estate"
    ALTERNATIVES_NAV_KNOWN = "alternatives_nav_known"
    ALTERNATIVES_NAV_UNKNOWN = "alternatives_nav_unknown"
    GOLD = "gold"
    CRYPTO = "crypto"


# Default zakat rates by investment asset class
INVESTMENT_ZAKAT_RATES = {
    InvestmentAssetClass.US_EQUITY: 0.0075,  # 0.75% (30% x 2.5%)
    InvestmentAssetClass.INTL_EQUITY: 0.0075,
    InvestmentAssetClass.FIXED_INCOME: 0.025,
    InvestmentAssetClass.CASH: 0.025,
    InvestmentAssetClass.REAL_ESTATE: 0.0,  # Exempt
    InvestmentAssetClass.ALTERNATIVES_NAV_KNOWN: 0.025,
    InvestmentAssetClass.ALTERNATIVES_NAV_UNKNOWN: 0.10,  # Conservative
    InvestmentAssetClass.GOLD: 0.025,
    InvestmentAssetClass.CRYPTO: 0.025,
}


@dataclass
class PortfolioAllocation:
    """Portfolio allocation with asset class breakdown for zakat calculations."""

    name: str
    allocations: dict[InvestmentAssetClass, float]  # Asset class -> weight (0-1)
    description: str = ""

    def __post_init__(self):
        total = sum(self.allocations.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Allocations must sum to 1.0, got {total}")

    def effective_zakat_rate(self, zakat_rates: dict[InvestmentAssetClass, float] | None = None) -> float:
        """Calculate blended effective zakat rate for this allocation."""
        rates = zakat_rates or INVESTMENT_ZAKAT_RATES
        return sum(weight * rates.get(asset_class, 0.025) for asset_class, weight in self.allocations.items())

    def calculate_zakat(
        self, portfolio_value: float, zakat_rates: dict[InvestmentAssetClass, float] | None = None
    ) -> dict[str, float]:
        """Calculate zakat by asset class and total."""
        rates = zakat_rates or INVESTMENT_ZAKAT_RATES
        breakdown = {}
        total = 0.0

        for asset_class, weight in self.allocations.items():
            asset_value = portfolio_value * weight
            zakat_rate = rates.get(asset_class, 0.025)
            zakat_amount = asset_value * zakat_rate
            breakdown[asset_class.value] = {
                "value": asset_value,
                "rate": zakat_rate,
                "zakat": zakat_amount,
            }
            total += zakat_amount

        return {
            "breakdown": breakdown,
            "total_zakat": total,
            "effective_rate": total / portfolio_value if portfolio_value > 0 else 0,
            "portfolio_value": portfolio_value,
        }


# Pre-defined portfolio configurations for retirement planning
PORTFOLIO_CONFIGS = {
    "60_40": PortfolioAllocation(
        name="Traditional 60/40",
        description="Classic balanced portfolio - 1.45% effective zakat rate",
        allocations={
            InvestmentAssetClass.US_EQUITY: 0.60,
            InvestmentAssetClass.FIXED_INCOME: 0.40,
        },
    ),
    "80_20": PortfolioAllocation(
        name="80/20 Growth",
        description="Growth-oriented with global equity - 1.10% effective zakat rate",
        allocations={
            InvestmentAssetClass.US_EQUITY: 0.50,
            InvestmentAssetClass.INTL_EQUITY: 0.30,
            InvestmentAssetClass.FIXED_INCOME: 0.15,
            InvestmentAssetClass.CASH: 0.05,
        },
    ),
    "yale": PortfolioAllocation(
        name="Yale Endowment Style",
        description="Diversified with alternatives and real assets - ~1.73% effective zakat rate",
        allocations={
            InvestmentAssetClass.US_EQUITY: 0.35,
            InvestmentAssetClass.INTL_EQUITY: 0.20,
            InvestmentAssetClass.REAL_ESTATE: 0.15,
            InvestmentAssetClass.ALTERNATIVES_NAV_KNOWN: 0.075,
            InvestmentAssetClass.ALTERNATIVES_NAV_UNKNOWN: 0.075,
            InvestmentAssetClass.FIXED_INCOME: 0.10,
            InvestmentAssetClass.CASH: 0.05,
        },
    ),
    "zakat_optimized": PortfolioAllocation(
        name="Zakat-Optimized Yale",
        description="Maximize low-zakat assets while maintaining diversification - ~0.95% effective zakat rate",
        allocations={
            InvestmentAssetClass.US_EQUITY: 0.25,
            InvestmentAssetClass.INTL_EQUITY: 0.15,
            InvestmentAssetClass.REAL_ESTATE: 0.25,
            InvestmentAssetClass.ALTERNATIVES_NAV_KNOWN: 0.20,
            InvestmentAssetClass.FIXED_INCOME: 0.10,
            InvestmentAssetClass.CASH: 0.05,
        },
    ),
    "all_equity": PortfolioAllocation(
        name="All-Equity",
        description="100% global equity for lowest effective zakat rate - 0.75%",
        allocations={
            InvestmentAssetClass.US_EQUITY: 0.60,
            InvestmentAssetClass.INTL_EQUITY: 0.40,
        },
    ),
}


@dataclass
class AssetClassZakatResult:
    """Result of asset-class-specific zakat calculation."""

    portfolio_value: float
    allocation_name: str
    total_zakat: float
    effective_rate: float
    breakdown: dict[str, dict]
    flat_rate_comparison: float
    savings_vs_flat: float
    strategy: str = "standard"

    def to_dict(self) -> dict:
        return {
            "portfolio_value": self.portfolio_value,
            "allocation": self.allocation_name,
            "strategy": self.strategy,
            "total_zakat": round(self.total_zakat, 2),
            "effective_rate_pct": round(self.effective_rate * 100, 3),
            "flat_rate_zakat": round(self.flat_rate_comparison, 2),
            "savings": round(self.savings_vs_flat, 2),
            "breakdown": self.breakdown,
        }


class AssetClassZakatCalculator:
    """Calculator with strategy support.

    FULL: 2.5% flat rate on all assets (most conservative)
    STANDARD: Asset-class-aware rates (0.75% for equities, 2.5% for others)
    """

    def __init__(
        self,
        strategy: ZakatStrategy = ZakatStrategy.STANDARD,
        zakat_rates: dict[InvestmentAssetClass, float] | None = None,
    ):
        self.strategy = strategy

        if zakat_rates is not None:
            self.zakat_rates = zakat_rates
        elif strategy == ZakatStrategy.FULL:
            self.zakat_rates = {asset_class: BASE_ZAKAT_RATE for asset_class in InvestmentAssetClass}
        else:
            self.zakat_rates = INVESTMENT_ZAKAT_RATES.copy()

    def calculate(
        self,
        portfolio_value: float,
        allocation: PortfolioAllocation | str,
    ) -> AssetClassZakatResult:
        """Calculate zakat for a portfolio with given allocation."""
        if isinstance(allocation, str):
            if allocation not in PORTFOLIO_CONFIGS:
                raise ValueError(f"Unknown portfolio config: {allocation}. Available: {list(PORTFOLIO_CONFIGS.keys())}")
            allocation = PORTFOLIO_CONFIGS[allocation]

        result = allocation.calculate_zakat(portfolio_value, self.zakat_rates)
        flat_rate_zakat = portfolio_value * BASE_ZAKAT_RATE

        return AssetClassZakatResult(
            portfolio_value=portfolio_value,
            allocation_name=allocation.name,
            total_zakat=result["total_zakat"],
            effective_rate=result["effective_rate"],
            breakdown=result["breakdown"],
            flat_rate_comparison=flat_rate_zakat,
            savings_vs_flat=flat_rate_zakat - result["total_zakat"],
            strategy=self.strategy.value,
        )

    def calculate_from_actual_holdings(
        self,
        holdings: dict[str, float],
        category_mapping: dict[str, InvestmentAssetClass] | None = None,
    ) -> AssetClassZakatResult:
        """Calculate zakat from actual holdings with category mapping.

        Args:
            holdings: Dict of category -> value
            category_mapping: Dict mapping category names to InvestmentAssetClass.
                Defaults to common investment category names.
        """
        if category_mapping is None:
            category_mapping = DEFAULT_CATEGORY_MAPPING

        total_value = sum(holdings.values())
        if total_value <= 0:
            return AssetClassZakatResult(
                portfolio_value=0,
                allocation_name="Actual Holdings",
                total_zakat=0,
                effective_rate=0,
                breakdown={},
                flat_rate_comparison=0,
                savings_vs_flat=0,
                strategy=self.strategy.value,
            )

        breakdown = {}
        total_zakat = 0.0

        for category, value in holdings.items():
            asset_class = category_mapping.get(category, InvestmentAssetClass.FIXED_INCOME)
            zakat_rate = self.zakat_rates.get(asset_class, BASE_ZAKAT_RATE)
            zakat_amount = value * zakat_rate

            breakdown[category] = {
                "value": value,
                "asset_class": asset_class.value,
                "rate": zakat_rate,
                "zakat": zakat_amount,
            }
            total_zakat += zakat_amount

        flat_rate_zakat = total_value * BASE_ZAKAT_RATE

        return AssetClassZakatResult(
            portfolio_value=total_value,
            allocation_name="Actual Holdings",
            total_zakat=total_zakat,
            effective_rate=total_zakat / total_value,
            breakdown=breakdown,
            flat_rate_comparison=flat_rate_zakat,
            savings_vs_flat=flat_rate_zakat - total_zakat,
            strategy=self.strategy.value,
        )


# Default mapping for common investment category names to asset classes
DEFAULT_CATEGORY_MAPPING: dict[str, InvestmentAssetClass] = {
    "US Stocks": InvestmentAssetClass.US_EQUITY,
    "US Index Funds": InvestmentAssetClass.US_EQUITY,
    "International Stocks": InvestmentAssetClass.INTL_EQUITY,
    "International Funds": InvestmentAssetClass.INTL_EQUITY,
    "Bond Funds": InvestmentAssetClass.FIXED_INCOME,
    "US Bonds": InvestmentAssetClass.FIXED_INCOME,
    "Cash": InvestmentAssetClass.CASH,
    "Real Estate": InvestmentAssetClass.REAL_ESTATE,
    "Alternatives": InvestmentAssetClass.ALTERNATIVES_NAV_KNOWN,
    "Commodities": InvestmentAssetClass.GOLD,
    "Inflation Hedge": InvestmentAssetClass.CRYPTO,
    "Crypto": InvestmentAssetClass.CRYPTO,
    "Cryptocurrency": InvestmentAssetClass.CRYPTO,
}


def get_effective_zakat_rate_for_allocation(allocation_name: str) -> float:
    """Get the effective zakat rate for a predefined portfolio allocation."""
    if allocation_name not in PORTFOLIO_CONFIGS:
        return 0.025
    return PORTFOLIO_CONFIGS[allocation_name].effective_zakat_rate()


def calculate_retirement_goal_with_zakat(
    annual_spending: float,
    base_withdrawal_rate: float = 0.05,
    allocation: PortfolioAllocation | str | None = None,
    use_flat_zakat: bool = False,
) -> dict:
    """Calculate retirement goal considering zakat as an additional expense.

    Key insight: With zakat, effective sustainable spending = base_rate - zakat_rate.

    Args:
        annual_spending: Target annual spending
        base_withdrawal_rate: Base withdrawal rate (default 5% for endowment)
        allocation: Portfolio allocation for asset-class zakat calculation
        use_flat_zakat: If True, use flat 2.5% instead of asset-class rates

    Returns:
        Dict with goals for both with/without zakat scenarios
    """
    goal_no_zakat = annual_spending / base_withdrawal_rate

    if use_flat_zakat:
        effective_zakat_rate = 0.025
    elif allocation:
        if isinstance(allocation, str):
            effective_zakat_rate = (
                PORTFOLIO_CONFIGS[allocation].effective_zakat_rate() if allocation in PORTFOLIO_CONFIGS else 0.025
            )
        else:
            effective_zakat_rate = allocation.effective_zakat_rate()
    else:
        effective_zakat_rate = 0.025

    effective_spending_rate = base_withdrawal_rate - effective_zakat_rate

    if effective_spending_rate <= 0:
        goal_with_zakat = float("inf")
    else:
        goal_with_zakat = annual_spending / effective_spending_rate

    return {
        "annual_spending": annual_spending,
        "base_withdrawal_rate": base_withdrawal_rate,
        "effective_zakat_rate": effective_zakat_rate,
        "effective_spending_rate": effective_spending_rate,
        "goal_no_zakat": goal_no_zakat,
        "goal_with_zakat": goal_with_zakat,
        "goal_multiplier": goal_with_zakat / goal_no_zakat if goal_no_zakat > 0 else float("inf"),
    }


def calculate_zakat_for_simulation(
    portfolio_value: float,
    equity_ratio: float = 0.70,
    strategy: ZakatStrategy = ZakatStrategy.STANDARD,
    zakat_rate_override: float | None = None,
) -> float:
    """Calculate annual Zakat for Monte Carlo simulations.

    Simplified calculation using equity ratio as a proxy for asset allocation.

    Args:
        portfolio_value: Current portfolio value
        equity_ratio: Proportion in equities (0-1). Default 0.70.
        strategy: ZakatStrategy.FULL or ZakatStrategy.STANDARD
        zakat_rate_override: If provided, uses flat rate instead

    Returns:
        Annual Zakat amount
    """
    if portfolio_value <= 0:
        return 0.0

    if zakat_rate_override is not None:
        return portfolio_value * zakat_rate_override

    if strategy == ZakatStrategy.FULL:
        return portfolio_value * BASE_ZAKAT_RATE

    equity_value = portfolio_value * equity_ratio
    other_value = portfolio_value * (1 - equity_ratio)

    return (equity_value * ZAKAT_RATE_EQUITY_STANDARD) + (other_value * BASE_ZAKAT_RATE)
