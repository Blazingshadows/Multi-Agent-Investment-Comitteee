"""Shared helpers for the LLM-backed agents (fundamental, sentiment,
macro_policy, risk) and both critics. The deterministic agents (technical,
forecasting) don't need this — they never call an LLM.
"""

from typing import Literal

from pydantic import BaseModel, Field

from core.schemas import AgentOutput, Direction


class LLMDirectionalCall(BaseModel):
    """What we ask the LLM to return — direction as a string enum (LLMs are
    far more reliable with named values than with raw -1/0/1 integers).
    """

    direction: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    evidence: list[str] = Field(default_factory=list)


def to_agent_output(agent_name: str, call: LLMDirectionalCall) -> AgentOutput:
    return AgentOutput(
        agent=agent_name,
        direction=Direction[call.direction],
        confidence=call.confidence,
        reasoning=call.reasoning,
        evidence=call.evidence,
    )


def neutral_fallback(agent_name: str, reason: str) -> AgentOutput:
    """Used when every LLM provider fails (LLMRouterError) — a zero-
    confidence NEUTRAL call the consensus engine can still process without
    crashing the cycle, rather than propagating the exception.
    """
    return AgentOutput(agent=agent_name, direction=Direction.NEUTRAL, confidence=0.0, reasoning=reason, evidence=[])
