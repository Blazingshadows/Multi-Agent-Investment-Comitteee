"""Portfolio state, the Risk Management Layer (review_trade — approve /
reduce / reject before any capital moves), the Execution Layer (execute —
mechanical order sizing once a decision has passed risk review), and forced
square-off before market close.
"""

from dataclasses import dataclass, field

from core.config import BUYING_POWER, MAX_POSITION_FRACTION, MAX_RISK_SCORE
from core.cost_model import apply_costs
from core.schemas import Action, ConsensusResult, CostBreakdown


@dataclass
class Portfolio:
    cash: float = BUYING_POWER
    positions: dict[str, float] = field(default_factory=dict)  # symbol -> qty (positive = long)


@dataclass
class ExecutionResult:
    action_taken: Action
    qty: float
    price: float
    net_cash_flow: float
    cost_breakdown: CostBreakdown | None


def review_trade(
    portfolio: Portfolio, decision: ConsensusResult, price: float, current_prices: dict[str, float]
) -> tuple[ConsensusResult, str]:
    """Risk Management Layer — final approval authority before execution.
    Returns a (possibly-reduced-or-rejected) copy of `decision` plus a note
    explaining what changed and why; never mutates the input. `current_prices`
    is the cycle's full market snapshot, needed to value existing positions
    the trade itself doesn't touch.
    """
    if decision.consensus_verdict not in (Action.BUY, Action.SELL):
        return decision, "No position change proposed — risk review not applicable."

    existing_exposure = sum(
        abs(qty) * current_prices.get(symbol, price) for symbol, qty in portfolio.positions.items()
    )
    headroom = BUYING_POWER - existing_exposure

    if headroom <= 0:
        rejected = decision.model_copy(update={"allocation": 0.0, "consensus_verdict": Action.WAIT})
        return rejected, (
            f"Rejected: existing exposure ₹{existing_exposure:,.0f} already at/over the "
            f"₹{BUYING_POWER:,.0f} leverage cap."
        )

    proposed_notional = decision.allocation * BUYING_POWER
    capped_notional = min(proposed_notional, headroom, MAX_POSITION_FRACTION * BUYING_POWER)
    reduced_allocation = capped_notional / BUYING_POWER

    notes = []
    if capped_notional < proposed_notional - 1e-6:
        notes.append(f"allocation {decision.allocation:.2%} -> {reduced_allocation:.2%} (leverage/concentration limit)")

    if decision.expected_risk_return.risk_score > MAX_RISK_SCORE:
        reduced_allocation *= 0.5
        notes.append(f"halved further — risk score {decision.expected_risk_return.risk_score:.2f} > {MAX_RISK_SCORE}")

    if reduced_allocation <= 1e-6:
        rejected = decision.model_copy(update={"allocation": 0.0, "consensus_verdict": Action.WAIT})
        return rejected, "Rejected: no capital headroom after risk adjustments."

    note = "Approved as proposed." if not notes else "Reduced: " + "; ".join(notes) + "."
    return decision.model_copy(update={"allocation": reduced_allocation}), note


def execute(portfolio: Portfolio, decision: ConsensusResult, price: float) -> ExecutionResult:
    """Mechanical order sizing + cost application. Call only on a decision
    that has already been through review_trade().
    """
    if decision.consensus_verdict not in (Action.BUY, Action.SELL):
        return ExecutionResult(action_taken=decision.consensus_verdict, qty=0.0, price=price, net_cash_flow=0.0, cost_breakdown=None)

    notional = decision.allocation * BUYING_POWER
    qty = round(notional / price)
    if qty <= 0:
        return ExecutionResult(action_taken=Action.WAIT, qty=0.0, price=price, net_cash_flow=0.0, cost_breakdown=None)

    net_cash_flow, cost_breakdown = apply_costs(decision.consensus_verdict, qty, price)

    portfolio.cash += net_cash_flow
    signed_qty = qty if decision.consensus_verdict == Action.BUY else -qty
    portfolio.positions[decision.symbol] = portfolio.positions.get(decision.symbol, 0.0) + signed_qty

    return ExecutionResult(action_taken=decision.consensus_verdict, qty=qty, price=price, net_cash_flow=net_cash_flow, cost_breakdown=cost_breakdown)


def force_square_off(portfolio: Portfolio, current_prices: dict[str, float]) -> list[tuple[str, ExecutionResult]]:
    """Closes every open position — call once at SESSION_SQUARE_OFF. Trading
    Rules require all positions closed before market close."""
    results = []
    for symbol, qty in list(portfolio.positions.items()):
        if qty == 0:
            continue
        price = current_prices[symbol]
        action = Action.SELL if qty > 0 else Action.BUY
        net_cash_flow, cost_breakdown = apply_costs(action, abs(qty), price)

        portfolio.cash += net_cash_flow
        portfolio.positions[symbol] = 0.0

        results.append((
            symbol,
            ExecutionResult(action_taken=action, qty=abs(qty), price=price, net_cash_flow=net_cash_flow, cost_breakdown=cost_breakdown),
        ))
    return results


def portfolio_value(portfolio: Portfolio, current_prices: dict[str, float]) -> float:
    positions_value = sum(qty * current_prices[symbol] for symbol, qty in portfolio.positions.items() if qty != 0)
    return portfolio.cash + positions_value
