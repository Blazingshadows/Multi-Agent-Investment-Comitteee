"""The Directional Confidence-Aware Consensus — the graded core of the whole
system. Pure math, no LLM call: expertise x trust x relevance x
agreement_live^gamma fused into a Directional Confidence Score, resolved into
BUY/SELL/HOLD/WAIT via §3's thresholds. SWITCH is a separate compositional
decision (evaluate_switch) since it requires comparing two stocks' DCS.
"""

import sqlite3

from core import trust_store
from core.config import (
    GAMMA_AGREEMENT,
    LAMBDA_HERDING,
    SWITCH_SAFETY_MARGIN,
    THETA_BUY,
    THETA_HOLD,
    THETA_SELL,
    THETA_VAR,
)
from core.cost_model import apply_costs
from core.schemas import (
    Action,
    AgentOutput,
    AgentWeight,
    ConsensusResult,
    CriticFeedback,
    ExpectedRiskReturn,
)

# Base skill prior per agent for short-horizon intraday calls — Technical and
# Forecasting are built for this regime; Fundamental/Macro are slower-moving
# disciplines with lower base relevance to a 5-15 min holding horizon.
BASE_EXPERTISE = {
    "Technical": 1.0,
    "Forecasting": 1.0,
    "Sentiment": 0.8,
    "Risk": 0.7,
    "Fundamental": 0.5,
    "Macro & Policy": 0.5,
}

# context flag -> {agent: multiplier}. Applied to both expertise (on top of
# the base skill prior) and relevance (regime-fit alone, base 1.0).
CONTEXT_BOOSTS = {
    "earnings_day": {"Fundamental": 1.8, "Sentiment": 1.5},
    "rbi_policy_day": {"Macro & Policy": 2.0},
    "high_volatility_day": {"Risk": 1.5, "Technical": 1.2},
}


def _context_multiplier(agent: str, context: dict) -> float:
    multiplier = 1.0
    for flag, boosts in CONTEXT_BOOSTS.items():
        if context.get(flag) and agent in boosts:
            multiplier *= boosts[agent]
    return multiplier


def _expertise(agent: str, context: dict) -> float:
    return BASE_EXPERTISE.get(agent, 1.0) * _context_multiplier(agent, context)


def _relevance(agent: str, context: dict) -> float:
    return _context_multiplier(agent, context)


def _agreement_live(signed_votes: list[float], index: int) -> float:
    """Leave-one-out peer corroboration — computed from raw votes, not from
    w_i or DCS, so there's no circular dependency (§3).
    """
    others = signed_votes[:index] + signed_votes[index + 1 :]
    if not others:
        return 1.0
    peer_mean = sum(others) / len(others)
    return 1 - abs(signed_votes[index] - peer_mean) / 2


def _build_reasoning(weight_breakdown: list[AgentWeight], verdict: Action, dcs: float, disagreement: float) -> str:
    ranked = sorted(weight_breakdown, key=lambda w: -w.w_normalized)
    top = ranked[:2]

    def _lean(vote: float) -> str:
        return "BUY-leaning" if vote > 0 else "SELL-leaning" if vote < 0 else "neutral"

    top_desc = ", ".join(f"{w.agent} (w={w.w_normalized:.2f}, {_lean(w.signed_vote)})" for w in top)

    dissenters = [w for w in weight_breakdown if (w.signed_vote > 0) != (dcs > 0) and abs(w.signed_vote) > 0.1]
    dissent_desc = ""
    if dissenters:
        d = max(dissenters, key=lambda w: w.w_normalized)
        dissent_desc = f" {d.agent} dissented but was discounted to w={d.w_normalized:.2f} (trust={d.trust:.2f})."

    return (
        f"{verdict.value} driven primarily by {top_desc}. "
        f"Directional Confidence Score={dcs:+.2f}, weighted disagreement={disagreement:.3f}.{dissent_desc}"
    )


def run_consensus(
    symbol: str,
    agent_outputs: list[AgentOutput],
    conn: sqlite3.Connection,
    context: dict | None = None,
    critic_feedback: list[CriticFeedback] | None = None,
    alternative_stocks_considered: list[str] | None = None,
    expected_risk_return: ExpectedRiskReturn | None = None,
) -> ConsensusResult:
    context = context or {}
    critic_feedback = critic_feedback or []
    alternative_stocks_considered = alternative_stocks_considered or []
    expected_risk_return = expected_risk_return or ExpectedRiskReturn(
        expected_return=0.0, expected_drawdown=0.0, risk_score=0.5
    )

    signed_votes = [a.signed_vote for a in agent_outputs]
    weights_raw: list[float] = []
    weight_breakdown: list[AgentWeight] = []

    for i, agent_output in enumerate(agent_outputs):
        expertise = _expertise(agent_output.agent, context)
        relevance = _relevance(agent_output.agent, context)
        reliability = trust_store.historical_reliability(conn, agent_output.agent)
        herding = trust_store.herding_penalty(conn, agent_output.agent)
        trust = reliability * (1 - LAMBDA_HERDING * herding)
        agreement_live = _agreement_live(signed_votes, i)

        w_raw = expertise * trust * relevance * (agreement_live**GAMMA_AGREEMENT)
        weights_raw.append(w_raw)
        weight_breakdown.append(
            AgentWeight(
                agent=agent_output.agent,
                expertise=expertise,
                historical_reliability=reliability,
                herding_penalty=herding,
                trust=trust,
                relevance=relevance,
                agreement_live=agreement_live,
                w_raw=w_raw,
                w_normalized=0.0,  # filled in below once the total is known
                signed_vote=agent_output.signed_vote,
            )
        )

    total_w = sum(weights_raw) or 1e-9
    weight_breakdown = [
        w.model_copy(update={"w_normalized": w_raw / total_w}) for w, w_raw in zip(weight_breakdown, weights_raw)
    ]

    dcs = sum(w.w_normalized * w.signed_vote for w in weight_breakdown)
    disagreement = sum(w.w_normalized * (w.signed_vote - dcs) ** 2 for w in weight_breakdown)

    if dcs >= THETA_BUY:
        verdict = Action.BUY
    elif dcs <= -THETA_SELL:
        verdict = Action.SELL
    elif abs(dcs) < THETA_HOLD and disagreement < THETA_VAR:
        verdict = Action.HOLD
    else:
        verdict = Action.WAIT

    # DCS is bounded to [-1, 1] (a weighted average of votes each in [-1, 1]),
    # so scale by how far past the BUY/SELL threshold the conviction sits —
    # otherwise allocation could never reach the 1:2 leverage cap at all.
    threshold = THETA_BUY if verdict == Action.BUY else THETA_SELL
    allocation = min(2.0, abs(dcs) / threshold) if verdict in (Action.BUY, Action.SELL) else 0.0
    reasoning = _build_reasoning(weight_breakdown, verdict, dcs, disagreement)

    return ConsensusResult(
        symbol=symbol,
        dcs=dcs,
        disagreement=disagreement,
        consensus_verdict=verdict,
        allocation=allocation,
        agent_recommendations=agent_outputs,
        weight_breakdown=weight_breakdown,
        consensus_reasoning=reasoning,
        alternative_stocks_considered=alternative_stocks_considered,
        critic_feedback=critic_feedback,
        expected_risk_return=expected_risk_return,
    )


def evaluate_switch(
    current_symbol: str,
    current_dcs: float,
    current_price: float,
    current_qty: float,
    alternative_symbol: str,
    alternative_dcs: float,
    alternative_price: float,
    safety_margin: float = SWITCH_SAFETY_MARGIN,
) -> bool:
    """True if rotating out of current_symbol into alternative_symbol clears
    the round-trip trading cost plus a safety margin — only meaningful when
    already holding a position (qty > 0). Prevents churning on noise (§3).
    """
    if current_qty <= 0:
        return False
    edge = alternative_dcs - current_dcs
    if edge <= 0:
        return False

    _, sell_cost = apply_costs(Action.SELL, current_qty, current_price)
    proceeds = current_qty * current_price - sell_cost.total_cost
    buy_qty = round(proceeds / alternative_price) if alternative_price > 0 else 0
    if buy_qty <= 0:
        return False
    _, buy_cost = apply_costs(Action.BUY, buy_qty, alternative_price)

    round_trip_cost_fraction = (sell_cost.total_cost + buy_cost.total_cost) / (current_qty * current_price)
    return edge > (round_trip_cost_fraction + safety_margin)
