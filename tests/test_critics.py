"""Mocks core.llm_router.complete (patched per-module) so these run offline.
Live quality (is the verdict actually specific, not a one-word non-answer)
was verified manually against real agent outputs and a real provider.
"""

from backend.critics import devils_advocate, opportunity
from backend.critics._common import LLMCriticCall
from core.schemas import AgentOutput, Direction


def _agent_output(name: str, direction: Direction, confidence: float) -> AgentOutput:
    return AgentOutput(agent=name, direction=direction, confidence=confidence, reasoning="test reasoning", evidence=[])


def test_devils_advocate_converts_llm_call(monkeypatch):
    fake = LLMCriticCall(verdict="Challenging Technical's bullish RSI read given weak volume confirmation.", alternative_stocks=[])
    monkeypatch.setattr(devils_advocate, "complete", lambda *a, **kw: fake)

    agents = [_agent_output("Technical", Direction.BULLISH, 0.8)]
    result = devils_advocate.review("INFY", agents, "BUY", 0.5)

    assert result.critic == "Devil's Advocate"
    assert "Technical" in result.verdict


def test_devils_advocate_llm_failure_returns_readable_fallback(monkeypatch):
    from core.llm_router import LLMRouterError

    def always_fails(*a, **kw):
        raise LLMRouterError("no providers")

    monkeypatch.setattr(devils_advocate, "complete", always_fails)
    agents = [_agent_output("Technical", Direction.BULLISH, 0.8)]
    result = devils_advocate.review("INFY", agents, "BUY", 0.5)

    assert "unavailable" in result.verdict


def test_opportunity_no_other_candidates_short_circuits_without_llm(monkeypatch):
    def fail_if_called(*a, **kw):
        raise AssertionError("should not call LLM")

    monkeypatch.setattr(opportunity, "complete", fail_if_called)
    result = opportunity.review("INFY", 0.3, {})
    assert result.alternative_stocks == []


def test_opportunity_converts_llm_call_and_names_alternatives(monkeypatch):
    fake = LLMCriticCall(verdict="RELIANCE has meaningfully stronger conviction (DCS +0.62 vs +0.30).", alternative_stocks=["RELIANCE"])
    monkeypatch.setattr(opportunity, "complete", lambda *a, **kw: fake)

    result = opportunity.review("INFY", 0.30, {"RELIANCE": 0.62, "TCS": 0.10})
    assert result.alternative_stocks == ["RELIANCE"]
    assert "RELIANCE" in result.verdict


def test_opportunity_llm_failure_falls_back_to_ranked_alternatives(monkeypatch):
    from core.llm_router import LLMRouterError

    def always_fails(*a, **kw):
        raise LLMRouterError("no providers")

    monkeypatch.setattr(opportunity, "complete", always_fails)
    result = opportunity.review("INFY", 0.30, {"RELIANCE": 0.62, "TCS": 0.10, "SBIN": -0.50})
    # falls back to ranked-by-|DCS| when the LLM is unavailable, still a usable list
    assert result.alternative_stocks == ["RELIANCE", "SBIN", "TCS"]
