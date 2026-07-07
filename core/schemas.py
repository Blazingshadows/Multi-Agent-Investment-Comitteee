"""Frozen data contracts shared between Person A (agents/critics/consensus) and
Person B (data/execution/API/dashboard) — see PROJECT.md §7a. Do not change a
field name or type without syncing with the other person; add new optional
fields if you need to extend something mid-build.
"""

from datetime import datetime
from enum import Enum, IntEnum

from pydantic import BaseModel, Field, computed_field


class Direction(IntEnum):
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    WAIT = "WAIT"
    SWITCH = "SWITCH"


class AgentOutput(BaseModel):
    """What every specialist agent (Technical, Fundamental, Sentiment, Macro &
    Policy, Risk, Forecasting) returns each cycle. Owned/produced by Person A.
    """

    agent: str
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    evidence: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def signed_vote(self) -> float:
        return self.direction.value * self.confidence


class AgentWeight(BaseModel):
    """Every raw factor from the §3 influence formula, logged per agent per
    cycle — this is what makes the consensus explainable rather than a black
    box. Produced by Person A's consensus_engine.py.
    """

    agent: str
    expertise: float = Field(ge=0.0)
    historical_reliability: float = Field(ge=0.0, le=1.0)
    herding_penalty: float = Field(ge=0.0, le=1.0)
    trust: float = Field(ge=0.0)
    relevance: float = Field(ge=0.0)
    agreement_live: float = Field(ge=0.0, le=1.0)
    w_raw: float = Field(ge=0.0)
    w_normalized: float = Field(ge=0.0, le=1.0)
    signed_vote: float = Field(ge=-1.0, le=1.0)


class CriticFeedback(BaseModel):
    """One critic's verdict on the leading proposal. `critic` is a free string
    (not an enum) so this works unchanged whether the MVP's 2-critic setup or
    the full 4-critic split (README Layer 4) is running.
    """

    critic: str
    verdict: str
    alternative_stocks: list[str] = Field(default_factory=list)


class ExpectedRiskReturn(BaseModel):
    expected_return: float
    expected_drawdown: float = Field(ge=0.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    sharpe_est: float | None = None


class ConsensusResult(BaseModel):
    """The output of core/consensus_engine.py for one stock, one cycle.
    Person A produces this; Person B consumes it (Risk Layer, Execution,
    dashboard) without needing to know how it was computed.
    """

    symbol: str
    dcs: float = Field(ge=-1.0, le=1.0, description="Signed, weighted Directional Confidence Score (§3)")
    disagreement: float = Field(ge=0.0, description="Weighted variance of agent votes around dcs")
    consensus_verdict: Action
    allocation: float = Field(ge=0.0, le=2.0, description="Fraction of base capital; up to 2.0 under 1:2 leverage")
    agent_recommendations: list[AgentOutput]
    weight_breakdown: list[AgentWeight]
    consensus_reasoning: str
    alternative_stocks_considered: list[str] = Field(default_factory=list)
    critic_feedback: list[CriticFeedback] = Field(default_factory=list)
    expected_risk_return: ExpectedRiskReturn
    switch_target: str | None = Field(default=None, description="Set only when consensus_verdict == SWITCH")

    @computed_field
    @property
    def directional_confidence(self) -> float:
        return abs(self.dcs)


class CostBreakdown(BaseModel):
    """Every NSE intraday cost component from §4. Produced by Person B's
    cost_model.apply_costs().
    """

    brokerage: float = Field(ge=0.0)
    stt: float = Field(ge=0.0)
    exchange_txn_charges: float = Field(ge=0.0)
    sebi_charges: float = Field(ge=0.0)
    stamp_duty: float = Field(ge=0.0)
    gst: float = Field(ge=0.0)
    slippage: float = Field(ge=0.0)

    @computed_field
    @property
    def total_cost(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.exchange_txn_charges
            + self.sebi_charges
            + self.stamp_duty
            + self.gst
            + self.slippage
        )


class DecisionLogRow(BaseModel):
    """One row per cycle per stock evaluated — the §6 schema, and the single
    table that satisfies the PS's Per-Trade-Output and Final-Output sections.
    Person B persists this into db/schema.sql's decision_log table.
    """

    id: int | None = None
    cycle_ts: datetime
    stock: str
    agent_recommendations: list[AgentOutput]
    directional_confidence_score: float = Field(ge=0.0, le=1.0)
    weight_breakdown: list[AgentWeight]
    consensus_verdict: Action
    consensus_reasoning: str
    alternative_stocks_considered: list[str] = Field(default_factory=list)
    critic_feedback: list[CriticFeedback] = Field(default_factory=list)
    expected_risk_return: ExpectedRiskReturn
    action_taken: Action
    qty: float = 0.0
    price: float = 0.0
    cost_breakdown: CostBreakdown | None = None
    net_cash_flow: float = 0.0

    @classmethod
    def from_consensus(
        cls,
        cycle_ts: datetime,
        stock: str,
        result: ConsensusResult,
        action_taken: Action,
        qty: float,
        price: float,
        cost_breakdown: CostBreakdown | None,
        net_cash_flow: float,
    ) -> "DecisionLogRow":
        """Deterministic ConsensusResult -> DecisionLogRow mapping, so Person A's
        output and Person B's persisted row never drift out of sync by hand.
        """
        return cls(
            cycle_ts=cycle_ts,
            stock=stock,
            agent_recommendations=result.agent_recommendations,
            directional_confidence_score=result.directional_confidence,
            weight_breakdown=result.weight_breakdown,
            consensus_verdict=result.consensus_verdict,
            consensus_reasoning=result.consensus_reasoning,
            alternative_stocks_considered=result.alternative_stocks_considered,
            critic_feedback=result.critic_feedback,
            expected_risk_return=result.expected_risk_return,
            action_taken=action_taken,
            qty=qty,
            price=price,
            cost_breakdown=cost_breakdown,
            net_cash_flow=net_cash_flow,
        )
