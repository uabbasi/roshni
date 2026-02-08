"""Core financial data models.

Generic representations for accounts, holdings, and portfolios.
These are framework-agnostic — any data source (Empower, Plaid,
manual CSV) can produce these models.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass
class Account:
    """Investment account.

    Attributes:
        account_id: Unique identifier (string for portability).
        name: Human-readable name.
        account_type: e.g., "Brokerage", "401k", "IRA", "Roth IRA".
        total_value: Current total value.
        last_updated: When data was last refreshed.
        institution: e.g., "Vanguard", "Fidelity" (optional).
    """

    account_id: str
    name: str
    account_type: str
    total_value: Decimal
    last_updated: datetime
    institution: str | None = None

    def __post_init__(self):
        if not isinstance(self.total_value, Decimal):
            self.total_value = Decimal(str(self.total_value))
        if self.total_value < 0:
            raise ValueError(f"Account {self.name} has negative value: {self.total_value}")
        if not self.name:
            raise ValueError("Account name cannot be empty")


@dataclass
class Holding:
    """Individual investment position.

    Attributes:
        account_id: Foreign key to Account.
        ticker: Stock/fund ticker symbol.
        description: Full name of security.
        quantity: Number of shares.
        price: Current price per share.
        value: Total value (quantity * price).
        cost_basis: Original purchase price.
        asset_class: e.g., "Stock", "Bond", "Fund" (optional).
    """

    account_id: str
    ticker: str
    description: str
    quantity: Decimal
    price: Decimal
    value: Decimal
    cost_basis: Decimal
    asset_class: str | None = None

    def __post_init__(self):
        for field_name in ["quantity", "price", "value", "cost_basis"]:
            val = getattr(self, field_name)
            if not isinstance(val, Decimal):
                setattr(self, field_name, Decimal(str(val)))

        if self.price < 0:
            raise ValueError(f"Negative price for {self.ticker}: {self.price}")

    @property
    def gain_loss(self) -> Decimal:
        """Unrealized gain/loss."""
        return self.value - self.cost_basis

    @property
    def gain_loss_pct(self) -> float:
        """Unrealized gain/loss as percentage."""
        if self.cost_basis == 0:
            return 0.0
        return float((self.value - self.cost_basis) / self.cost_basis)
