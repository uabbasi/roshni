"""Life event modeling for retirement scenario planning.

Data structures for modeling scheduled financial events
that affect retirement sustainability:
- Lump sum withdrawals (school tuition, home purchase)
- Spending changes (mortgage payoff, healthcare starting/stopping)

These events are processed by Monte Carlo simulators to show impact
on retirement success rates.
"""

from dataclasses import dataclass, field
from enum import Enum


class EventType(Enum):
    """Type of life event affecting retirement plan."""

    LUMP_WITHDRAWAL = "lump_withdrawal"  # One-time portfolio withdrawal
    SPENDING_CHANGE = "spending_change"  # Permanent spending adjustment


@dataclass
class LifeEvent:
    """A scheduled financial event affecting retirement plan.

    Attributes:
        year: Year of retirement when event occurs (0 = now/before retirement starts,
              1 = first year, etc.). For LUMP_WITHDRAWAL, the withdrawal happens
              at the start of that year.
        event_type: Type of event (lump withdrawal or spending change)
        amount: For LUMP_WITHDRAWAL: amount to withdraw (positive).
                For SPENDING_CHANGE: change in annual spending (negative = reduction,
                e.g., -30000 for mortgage payoff reducing expenses by $30k/year)
        description: Human-readable label for display
    """

    year: int
    event_type: EventType
    amount: float
    description: str = ""

    def __post_init__(self):
        """Validate event parameters."""
        if self.year < 0:
            raise ValueError(f"Event year must be >= 0, got {self.year}")
        if self.event_type == EventType.LUMP_WITHDRAWAL and self.amount <= 0:
            raise ValueError(f"Lump withdrawal amount must be positive, got {self.amount}")

    def format_amount(self) -> str:
        """Format amount for display."""
        if self.event_type == EventType.LUMP_WITHDRAWAL:
            return f"-${self.amount:,.0f}"
        else:
            if self.amount < 0:
                return f"-${abs(self.amount):,.0f}/year"
            else:
                return f"+${self.amount:,.0f}/year"

    def __str__(self) -> str:
        year_str = "now" if self.year == 0 else f"year {self.year}"
        if self.event_type == EventType.LUMP_WITHDRAWAL:
            return f"{self.description or 'Withdrawal'}: {self.format_amount()} ({year_str})"
        else:
            action = "reduced" if self.amount < 0 else "increased"
            return f"{self.description or 'Spending'} {action} by ${abs(self.amount):,.0f}/year (from {year_str})"


@dataclass
class ScenarioConfig:
    """Configuration for a life event scenario simulation.

    Bundles scenario parameters for easy passing to simulators.
    """

    name: str = "Untitled Scenario"
    events: list[LifeEvent] = field(default_factory=list)

    def add_lump_withdrawal(self, amount: float, year: int = 0, description: str = "") -> "ScenarioConfig":
        """Add a lump sum withdrawal event. Returns self for chaining."""
        self.events.append(
            LifeEvent(
                year=year,
                event_type=EventType.LUMP_WITHDRAWAL,
                amount=amount,
                description=description,
            )
        )
        return self

    def add_spending_change(self, amount: float, year: int, description: str = "") -> "ScenarioConfig":
        """Add a permanent spending change event. Returns self for chaining."""
        self.events.append(
            LifeEvent(
                year=year,
                event_type=EventType.SPENDING_CHANGE,
                amount=amount,
                description=description,
            )
        )
        return self

    def clear(self) -> "ScenarioConfig":
        """Clear all events."""
        self.events.clear()
        return self

    @property
    def total_lump_withdrawals(self) -> float:
        """Total of all lump sum withdrawals."""
        return sum(e.amount for e in self.events if e.event_type == EventType.LUMP_WITHDRAWAL)

    @property
    def final_spending_adjustment(self) -> float:
        """Net spending adjustment after all events."""
        return sum(e.amount for e in self.events if e.event_type == EventType.SPENDING_CHANGE)

    def events_for_year(self, year: int) -> list[LifeEvent]:
        """Get all events scheduled for a specific year."""
        return [e for e in self.events if e.year == year]

    def __str__(self) -> str:
        if not self.events:
            return f"{self.name}: No events"
        event_strs = [f"  - {e}" for e in sorted(self.events, key=lambda x: x.year)]
        return f"{self.name}:\n" + "\n".join(event_strs)


def spending_phases_to_events(
    spending_phases: list[dict],
    start_year: int | None = None,
) -> list[LifeEvent]:
    """Convert spending phases to LifeEvent objects for Monte Carlo.

    The spending phases represent absolute spending levels at different
    points in time. This function converts the *changes* between phases
    into LifeEvent spending adjustments.

    Also handles lump withdrawal entries (marked with 'lump_withdrawal' key).

    Args:
        spending_phases: List of dicts with 'year', 'spending', 'description' keys
            sorted by year (ascending). Entries with 'lump_withdrawal' key are
            one-time portfolio withdrawals.
        start_year: The retirement start year. If None, uses current year.

    Returns:
        List of LifeEvent objects representing spending changes and lump withdrawals.
    """
    from datetime import date

    if not spending_phases:
        return []

    if start_year is None:
        start_year = date.today().year

    events = []

    # Separate spending phases from lump withdrawals
    spending_only = [p for p in spending_phases if p.get("spending") is not None]
    lump_withdrawals = [p for p in spending_phases if p.get("lump_withdrawal")]

    # Process lump withdrawals
    for phase in lump_withdrawals:
        year_of_change = phase["year"]
        years_from_start = year_of_change - start_year

        if years_from_start >= 0:
            events.append(
                LifeEvent(
                    year=years_from_start,
                    event_type=EventType.LUMP_WITHDRAWAL,
                    amount=phase["lump_withdrawal"],
                    description=phase.get("description", f"Lump withdrawal year {year_of_change}"),
                )
            )

    # Process spending changes (need at least 2 phases to compare)
    if len(spending_only) >= 2:
        sorted_phases = sorted(spending_only, key=lambda p: p["year"])

        for i in range(1, len(sorted_phases)):
            prev_phase = sorted_phases[i - 1]
            curr_phase = sorted_phases[i]

            spending_change = curr_phase["spending"] - prev_phase["spending"]
            year_of_change = curr_phase["year"]
            years_from_start = year_of_change - start_year

            if years_from_start > 0 and spending_change != 0:
                events.append(
                    LifeEvent(
                        year=years_from_start,
                        event_type=EventType.SPENDING_CHANGE,
                        amount=spending_change,
                        description=curr_phase.get("description", f"Spending change year {year_of_change}"),
                    )
                )

    return events
