"""Mocks core.llm_router.complete (patched per-module, since each agent
imports it into its own namespace) so these run offline, no live API call.
Live behavior is covered by manual smoke tests against real data/providers.
"""

import numpy as np
import pandas as pd
import pytest

from backend.agents import fundamental, macro_policy, risk, sentiment
from backend.agents._common import LLMDirectionalCall
from core.llm_router import LLMRouterError
from core.schemas import Direction


def _fake_call(direction="BULLISH", confidence=0.7):
    return LLMDirectionalCall(direction=direction, confidence=confidence, reasoning="test reasoning", evidence=["e1"])


def _always_fails(*args, **kwargs):
    raise LLMRouterError("no providers configured")


def _ohlcv(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    timestamps = pd.date_range("2026-07-08 09:15", periods=n, freq="5min")
    close = 1000 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "open": close, "high": close + 1, "low": close - 1, "close": close,
            "volume": rng.integers(10_000, 100_000, n),
        },
        index=timestamps,
    )


# --- Fundamental ---


def test_fundamental_no_data_is_neutral_without_calling_llm(monkeypatch):
    monkeypatch.setattr(fundamental, "complete", lambda *a, **kw: pytest.fail("should not call LLM"))
    result = fundamental.analyze("INFY", {})
    assert result.direction == Direction.NEUTRAL
    assert result.confidence == 0.0


def test_fundamental_converts_llm_call(monkeypatch):
    monkeypatch.setattr(fundamental, "complete", lambda *a, **kw: _fake_call("BULLISH", 0.8))
    result = fundamental.analyze("INFY", {"fundamentals": {"trailingPE": 14.0}})
    assert result.agent == "Fundamental"
    assert result.direction == Direction.BULLISH
    assert result.confidence == 0.8


def test_fundamental_llm_failure_falls_back_to_neutral(monkeypatch):
    monkeypatch.setattr(fundamental, "complete", _always_fails)
    result = fundamental.analyze("INFY", {"fundamentals": {"trailingPE": 14.0}})
    assert result.direction == Direction.NEUTRAL
    assert result.confidence == 0.0
    assert "LLM unavailable" in result.reasoning


# --- Sentiment ---


def test_sentiment_no_headlines_is_neutral_without_calling_llm(monkeypatch):
    monkeypatch.setattr(sentiment, "complete", lambda *a, **kw: pytest.fail("should not call LLM"))
    result = sentiment.analyze("INFY", {"news": []})
    assert result.direction == Direction.NEUTRAL


def test_sentiment_converts_llm_call(monkeypatch):
    monkeypatch.setattr(sentiment, "complete", lambda *a, **kw: _fake_call("BEARISH", 0.6))
    result = sentiment.analyze("INFY", {"news": ["Infosys misses Q1 estimates"]})
    assert result.agent == "Sentiment"
    assert result.direction == Direction.BEARISH


# --- Macro & Policy ---


def test_macro_policy_no_context_is_neutral_without_calling_llm(monkeypatch):
    monkeypatch.setattr(macro_policy, "complete", lambda *a, **kw: pytest.fail("should not call LLM"))
    result = macro_policy.analyze("INFY", {})
    assert result.direction == Direction.NEUTRAL


def test_macro_policy_converts_llm_call(monkeypatch):
    monkeypatch.setattr(macro_policy, "complete", lambda *a, **kw: _fake_call("NEUTRAL", 0.3))
    result = macro_policy.analyze("INFY", {"macro_flags": {"rbi_policy_day": False}})
    assert result.agent == "Macro & Policy"
    assert result.direction == Direction.NEUTRAL


# --- Risk ---


def test_risk_compute_stats_none_when_too_little_history():
    assert risk.compute_risk_stats(_ohlcv(n=3)) is None


def test_risk_score_from_atr_scales_and_clips():
    assert risk.risk_score_from_atr(0.0) == 0.0
    assert risk.risk_score_from_atr(0.02) == 1.0
    assert risk.risk_score_from_atr(0.05) == 1.0  # clipped, not > 1
    assert 0.0 < risk.risk_score_from_atr(0.01) < 1.0


def test_risk_no_history_is_neutral_without_calling_llm(monkeypatch):
    monkeypatch.setattr(risk, "complete", lambda *a, **kw: pytest.fail("should not call LLM"))
    result = risk.analyze("INFY", {"ohlcv": _ohlcv(n=3)})
    assert result.direction == Direction.NEUTRAL


def test_risk_converts_llm_call(monkeypatch):
    monkeypatch.setattr(risk, "complete", lambda *a, **kw: _fake_call("NEUTRAL", 0.5))
    result = risk.analyze("INFY", {"ohlcv": _ohlcv(n=30)})
    assert result.agent == "Risk"
    assert result.direction == Direction.NEUTRAL
