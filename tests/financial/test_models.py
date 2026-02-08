"""Tests for roshni.financial.models."""

from datetime import datetime
from decimal import Decimal

import pytest

from roshni.financial.models import Account, Holding


class TestAccount:
    def test_create(self):
        acct = Account(
            account_id="123",
            name="Brokerage",
            account_type="Taxable",
            total_value=Decimal("100000"),
            last_updated=datetime.now(),
        )
        assert acct.account_id == "123"
        assert acct.total_value == Decimal("100000")

    def test_auto_convert_to_decimal(self):
        acct = Account(
            account_id="1",
            name="Test",
            account_type="401k",
            total_value=50000.50,
            last_updated=datetime.now(),
        )
        assert isinstance(acct.total_value, Decimal)
        assert acct.total_value == Decimal("50000.5")

    def test_negative_value_raises(self):
        with pytest.raises(ValueError, match="negative value"):
            Account(
                account_id="1",
                name="Bad",
                account_type="X",
                total_value=Decimal("-100"),
                last_updated=datetime.now(),
            )

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            Account(
                account_id="1",
                name="",
                account_type="X",
                total_value=Decimal("100"),
                last_updated=datetime.now(),
            )


class TestHolding:
    def test_create(self):
        holding = Holding(
            account_id="123",
            ticker="VTSAX",
            description="Vanguard Total Stock Market",
            quantity=Decimal("100"),
            price=Decimal("50"),
            value=Decimal("5000"),
            cost_basis=Decimal("4000"),
        )
        assert holding.gain_loss == Decimal("1000")
        assert holding.gain_loss_pct == pytest.approx(0.25)

    def test_auto_convert_to_decimal(self):
        holding = Holding(
            account_id="1",
            ticker="VTI",
            description="Vanguard ETF",
            quantity=100,
            price=50.0,
            value=5000,
            cost_basis=4000,
        )
        assert isinstance(holding.quantity, Decimal)
        assert isinstance(holding.price, Decimal)

    def test_negative_price_raises(self):
        with pytest.raises(ValueError, match="Negative price"):
            Holding(
                account_id="1",
                ticker="BAD",
                description="Bad",
                quantity=Decimal("10"),
                price=Decimal("-5"),
                value=Decimal("-50"),
                cost_basis=Decimal("100"),
            )

    def test_zero_cost_basis_gain_pct(self):
        holding = Holding(
            account_id="1",
            ticker="FREE",
            description="Free shares",
            quantity=Decimal("10"),
            price=Decimal("50"),
            value=Decimal("500"),
            cost_basis=Decimal("0"),
        )
        assert holding.gain_loss_pct == 0.0
