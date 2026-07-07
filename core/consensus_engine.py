"""STUB — Person A owns this file. Replace run_consensus() with the real §3
Directional Confidence-Aware Consensus formula (expertise x trust x relevance
x agreement_live, DCS fusion, HOLD/WAIT/BUY/SELL thresholds). Every downstream
consumer (execution, persistence, dashboard) is built against the
ConsensusResult shape, not against this stub's logic — so swapping this
function out for the real thing never requires touching Person B's files.
"""

from core.schemas import (
    Action,
    AgentOutput,
    AgentWeight,
    ConsensusResult,
    CriticFeedback,
    ExpectedRiskReturn,
)


def run_consensus(symbol: str, agent_outputs: list[AgentOutput]) -> ConsensusResult:
    """STUB: ignores the real weighting math and returns a fixed verdict so
    the rest of the pipeline can be built and tested against a stable shape.
    """
    weight_breakdown = [
        AgentWeight(
            agent=a.agent,
            expertise=1.0,
            historical_reliability=0.5,
            herding_penalty=0.0,
            trust=0.5,
            relevance=1.0,
            agreement_live=1.0,
            w_raw=1.0,
            w_normalized=1.0 / len(agent_outputs),
            signed_vote=a.signed_vote,
        )
        for a in agent_outputs
    ]

    return ConsensusResult(
        symbol=symbol,
        dcs=0.42,
        disagreement=0.1,
        consensus_verdict=Action.BUY,
        allocation=0.25,
        agent_recommendations=agent_outputs,
        weight_breakdown=weight_breakdown,
        consensus_reasoning="STUB verdict — replace with the real §3 Directional Confidence Score computation.",
        alternative_stocks_considered=[],
        critic_feedback=[
            CriticFeedback(critic="Devil's Advocate", verdict="STUB — no real critic has run yet.")
        ],
        expected_risk_return=ExpectedRiskReturn(expected_return=0.0, expected_drawdown=0.0, risk_score=0.5),
    )
