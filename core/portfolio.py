"""Person B owns this file. Minimal real implementation for the stub
pipeline — sizes an order from ConsensusResult.allocation and applies real
trading costs. Risk Management Layer approval/rejection (position limits,
leverage cap enforcement, forced square-off) is not implemented yet;
everything here executes exactly as the consensus proposed.
"""

from dataclasses import dataclass, field

from core.config import BUYING_POWER
from core.cost_model import apply_costs
from core.schemas import Action, ConsensusResult, CostBreakdown


@dataclass
class Portfolio:
    cash: float = BUYING_POWER
    positions: dict[str, float] = field(default_factory=dict)  # symbol -> qty


@dataclass
class ExecutionResult:
    action_taken: Action
    qty: float
    price: float
    net_cash_flow: float
    cost_breakdown: CostBreakdown | None


def execute(portfolio: Portfolio, decision: ConsensusResult, price: float) -> ExecutionResult:
    if decision.consensus_verdict not in (Action.BUY, Action.SELL):
        return ExecutionResult(
            action_taken=decision.consensus_verdict, qty=0.0, price=price,
            net_cash_flow=0.0, cost_breakdown=None,
        )

    notional = decision.allocation * BUYING_POWER
    qty = round(notional / price)
    net_cash_flow, cost_breakdown = apply_costs(decision.consensus_verdict, qty, price)

    portfolio.cash += net_cash_flow
    signed_qty = qty if decision.consensus_verdict == Action.BUY else -qty
    portfolio.positions[decision.symbol] = portfolio.positions.get(decision.symbol, 0.0) + signed_qty

    return ExecutionResult(
        action_taken=decision.consensus_verdict, qty=qty, price=price,
        net_cash_flow=net_cash_flow, cost_breakdown=cost_breakdown,
    )
