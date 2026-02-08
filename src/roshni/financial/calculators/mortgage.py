"""Mortgage scenario analyzer for ARM prepayment and payoff decisions.

Compares different mortgage strategies:
- Prepaying extra each month
- Lump sum payoff
- Letting ARM reset
- Refinancing

Pure math — zero external dependencies.
"""

from dataclasses import dataclass, field


@dataclass
class MortgageTerms:
    """Current mortgage terms."""

    balance: float
    current_rate: float  # Annual rate (e.g., 0.0245 for 2.45%)
    is_interest_only: bool = True
    reset_year: int | None = None  # Year ARM resets
    reset_rate: float | None = None  # Projected rate at reset
    remaining_term_years: int = 20  # Years to amortize after reset


@dataclass
class PrepayScenario:
    """Results from a prepayment scenario."""

    name: str
    monthly_prepay: float
    total_prepaid: float  # Over period until reset
    balance_at_reset: float
    monthly_payment_after_reset: float
    total_interest_to_reset: float
    total_interest_after_reset: float
    total_interest: float
    total_paid: float  # Interest + principal


@dataclass
class MortgageComparison:
    """Comparison of multiple mortgage scenarios."""

    current_balance: float
    reset_year: int
    years_to_reset: int
    scenarios: list[PrepayScenario] = field(default_factory=list)

    def format_table(self) -> str:
        """Format comparison as text table."""
        lines = []
        lines.append("=" * 75)
        lines.append("  Mortgage Scenario Comparison")
        lines.append("=" * 75)
        lines.append(f"\nCurrent Balance: ${self.current_balance:,.0f}")
        lines.append(f"Reset Year: {self.reset_year} ({self.years_to_reset} years)")
        lines.append("")

        lines.append("-" * 75)
        lines.append(
            f"{'Scenario':<20} {'Prepay':<12} {'Balance@Reset':<15} {'Payment After':<15} {'Total Interest':<12}"
        )
        lines.append("-" * 75)

        for s in sorted(self.scenarios, key=lambda x: x.total_interest):
            prepay_str = f"${s.monthly_prepay:,.0f}/mo" if s.monthly_prepay > 0 else "None"
            lines.append(
                f"{s.name:<20} {prepay_str:<12} ${s.balance_at_reset:>12,.0f} "
                f"${s.monthly_payment_after_reset:>12,.0f}/mo ${s.total_interest:>10,.0f}"
            )

        lines.append("-" * 75)

        if len(self.scenarios) >= 2:
            baseline = self.scenarios[0]
            lines.append("\nSavings vs Do Nothing:")
            for s in self.scenarios[1:]:
                interest_saved = baseline.total_interest - s.total_interest
                payment_saved = baseline.monthly_payment_after_reset - s.monthly_payment_after_reset
                if interest_saved > 0:
                    lines.append(
                        f"  {s.name}: Save ${interest_saved:,.0f} interest, ${payment_saved:,.0f}/mo lower payment"
                    )

        return "\n".join(lines)


def calculate_monthly_payment(principal: float, annual_rate: float, years: int) -> float:
    """Calculate monthly payment for amortizing loan.

    Args:
        principal: Loan amount
        annual_rate: Annual interest rate (e.g., 0.07 for 7%)
        years: Loan term in years

    Returns:
        Monthly payment amount
    """
    if annual_rate <= 0 or years <= 0 or principal <= 0:
        return 0

    monthly_rate = annual_rate / 12
    num_payments = years * 12

    payment = principal * (monthly_rate * (1 + monthly_rate) ** num_payments) / ((1 + monthly_rate) ** num_payments - 1)
    return payment


def calculate_interest_only_payment(principal: float, annual_rate: float) -> float:
    """Calculate interest-only monthly payment."""
    return principal * annual_rate / 12


def project_balance_with_prepay(
    starting_balance: float,
    annual_rate: float,
    months: int,
    monthly_prepay: float = 0,
    is_interest_only: bool = True,
) -> tuple[float, float]:
    """Project loan balance after prepayments.

    Args:
        starting_balance: Current loan balance
        annual_rate: Annual interest rate
        months: Number of months to project
        monthly_prepay: Extra payment each month (goes to principal)
        is_interest_only: If True, regular payment is interest-only

    Returns:
        Tuple of (ending_balance, total_interest_paid)
    """
    balance = starting_balance
    total_interest = 0
    monthly_rate = annual_rate / 12

    for _ in range(months):
        interest = balance * monthly_rate
        total_interest += interest

        if monthly_prepay > 0:
            balance = max(0, balance - monthly_prepay)

        if balance <= 0:
            break

    return balance, total_interest


def analyze_prepay_scenario(
    terms: MortgageTerms,
    monthly_prepay: float,
    scenario_name: str,
    years_to_reset: int,
) -> PrepayScenario:
    """Analyze a prepayment scenario.

    Args:
        terms: Current mortgage terms
        monthly_prepay: Extra monthly payment
        scenario_name: Name for this scenario
        years_to_reset: Years until ARM resets

    Returns:
        PrepayScenario with all calculations
    """
    months_to_reset = years_to_reset * 12

    balance_at_reset, interest_to_reset = project_balance_with_prepay(
        starting_balance=terms.balance,
        annual_rate=terms.current_rate,
        months=months_to_reset,
        monthly_prepay=monthly_prepay,
        is_interest_only=terms.is_interest_only,
    )

    reset_rate = terms.reset_rate or 0.07
    remaining_years = terms.remaining_term_years

    if balance_at_reset > 0:
        monthly_after_reset = calculate_monthly_payment(balance_at_reset, reset_rate, remaining_years)
        total_payments_after = monthly_after_reset * remaining_years * 12
        interest_after_reset = total_payments_after - balance_at_reset
    else:
        monthly_after_reset = 0
        interest_after_reset = 0

    total_prepaid = monthly_prepay * months_to_reset

    return PrepayScenario(
        name=scenario_name,
        monthly_prepay=monthly_prepay,
        total_prepaid=total_prepaid,
        balance_at_reset=balance_at_reset,
        monthly_payment_after_reset=monthly_after_reset,
        total_interest_to_reset=interest_to_reset,
        total_interest_after_reset=max(0, interest_after_reset),
        total_interest=interest_to_reset + max(0, interest_after_reset),
        total_paid=interest_to_reset + max(0, interest_after_reset) + terms.balance,
    )


def compare_scenarios(
    terms: MortgageTerms,
    prepay_amounts: list[float],
    current_year: int,
) -> MortgageComparison:
    """Compare multiple prepayment scenarios.

    Args:
        terms: Current mortgage terms
        prepay_amounts: List of monthly prepayment amounts to compare (0 = do nothing)
        current_year: Current year for calculating years to reset

    Returns:
        MortgageComparison with all scenarios
    """
    reset_year = terms.reset_year or (current_year + 5)
    years_to_reset = max(1, reset_year - current_year)

    scenarios = []

    for prepay in prepay_amounts:
        name = "Do Nothing" if prepay == 0 else f"Prepay ${prepay:,.0f}/mo"

        scenario = analyze_prepay_scenario(
            terms=terms,
            monthly_prepay=prepay,
            scenario_name=name,
            years_to_reset=years_to_reset,
        )
        scenarios.append(scenario)

    return MortgageComparison(
        current_balance=terms.balance,
        reset_year=reset_year,
        years_to_reset=years_to_reset,
        scenarios=scenarios,
    )


def calculate_lump_sum_payoff(
    terms: MortgageTerms,
    payoff_year: int,
    current_year: int,
) -> dict:
    """Calculate impact of lump sum payoff.

    Args:
        terms: Current mortgage terms
        payoff_year: Year to pay off mortgage
        current_year: Current year

    Returns:
        Dict with payoff analysis
    """
    years_to_payoff = max(0, payoff_year - current_year)
    months_to_payoff = years_to_payoff * 12

    _, interest_to_payoff = project_balance_with_prepay(
        starting_balance=terms.balance,
        annual_rate=terms.current_rate,
        months=months_to_payoff,
        monthly_prepay=0,
        is_interest_only=terms.is_interest_only,
    )

    current_monthly = calculate_interest_only_payment(terms.balance, terms.current_rate)

    reset_year = terms.reset_year or (current_year + 5)
    if payoff_year < reset_year:
        years_to_reset = max(1, reset_year - current_year)
        months_to_reset = years_to_reset * 12

        _, interest_to_reset = project_balance_with_prepay(terms.balance, terms.current_rate, months_to_reset, 0, True)

        reset_rate = terms.reset_rate or 0.07
        monthly_after_reset = calculate_monthly_payment(terms.balance, reset_rate, terms.remaining_term_years)
        interest_after_reset = monthly_after_reset * terms.remaining_term_years * 12 - terms.balance
        avoided_interest = interest_to_reset + interest_after_reset - interest_to_payoff
    else:
        avoided_interest = 0
        monthly_after_reset = 0

    return {
        "payoff_amount": terms.balance,
        "payoff_year": payoff_year,
        "interest_to_payoff": interest_to_payoff,
        "current_monthly": current_monthly,
        "monthly_savings": current_monthly,
        "annual_savings": current_monthly * 12,
        "avoided_interest": avoided_interest,
        "monthly_after_reset_avoided": monthly_after_reset,
    }
