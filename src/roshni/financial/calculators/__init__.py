"""Financial calculators — tax, mortgage, zakat, life events."""

from .life_events import EventType, LifeEvent, ScenarioConfig, spending_phases_to_events
from .mortgage import (
    MortgageComparison,
    MortgageTerms,
    PrepayScenario,
    calculate_lump_sum_payoff,
    calculate_monthly_payment,
    compare_scenarios,
)
from .tax_tables import (
    RMD_UNIFORM_LIFETIME_TABLE,
    TAX_BRACKETS_2026_CA_MFJ,
    TAX_BRACKETS_2026_FEDERAL_MFJ,
    FilingStatus,
)
from .zakat import (
    AssetClassZakatCalculator,
    ZakatCalculator,
    ZakatConfig,
    ZakatStrategy,
)

__all__ = [
    "RMD_UNIFORM_LIFETIME_TABLE",
    "TAX_BRACKETS_2026_CA_MFJ",
    "TAX_BRACKETS_2026_FEDERAL_MFJ",
    "AssetClassZakatCalculator",
    "EventType",
    "FilingStatus",
    "LifeEvent",
    "MortgageComparison",
    "MortgageTerms",
    "PrepayScenario",
    "ScenarioConfig",
    "ZakatCalculator",
    "ZakatConfig",
    "ZakatStrategy",
    "calculate_lump_sum_payoff",
    "calculate_monthly_payment",
    "compare_scenarios",
    "spending_phases_to_events",
]
