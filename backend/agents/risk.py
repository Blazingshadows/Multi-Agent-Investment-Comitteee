"""Risk Prediction agent — computes real volatility/drawdown statistics from
OHLCV (deterministic, via core.indicators.atr), then has the LLM interpret
them into a directional risk lean. The LLM only narrates already-computed
numbers, it never invents a risk figure. risk_score_from_atr() is also used
directly by the orchestrator to populate ConsensusResult.expected_risk_return,
so the Risk Management Layer's gating uses the same number this agent reasons
over.
"""

import pandas as pd

from backend.agents._common import LLMDirectionalCall, neutral_fallback, to_agent_output
from core.indicators import atr
from core.llm_router import LLMRouterError, complete
from core.schemas import AgentOutput

AGENT_NAME = "Risk"

SYSTEM_PROMPT = (
    "You are the Risk Prediction Analyst on an intraday NSE/BSE investment committee. "
    "You are given real, already-computed volatility and drawdown statistics for a "
    "stock — never invent numbers of your own. Elevated volatility/drawdown should "
    "usually pull you toward BEARISH (caution, reduce conviction) or NEUTRAL; only "
    "call BULLISH if risk conditions are unusually calm AND that calm itself supports "
    "adding exposure. Cite the specific statistics you were given."
)


def risk_score_from_atr(atr_pct: float, scale: float = 0.02) -> float:
    """0 at zero volatility, 1.0 at atr_pct >= `scale` (2% of price per bar's
    true range is already a large, high-risk intraday move).
    """
    return min(1.0, max(0.0, atr_pct / scale))


def compute_risk_stats(df: pd.DataFrame) -> dict | None:
    if df is None or len(df) < 15:
        return None

    atr_val = atr(df["high"], df["low"], df["close"]).iloc[-1]
    price = df["close"].iloc[-1]
    if pd.isna(atr_val):
        return None

    atr_pct = float(atr_val / price)
    rolling_high = df["close"].cummax()
    drawdown_pct = float((df["close"] / rolling_high - 1).min())

    return {
        "atr_pct_of_price": round(atr_pct, 4),
        "risk_score": round(risk_score_from_atr(atr_pct), 4),
        "max_drawdown_in_window": round(drawdown_pct, 4),
        "bars_in_window": len(df),
    }


def analyze(symbol: str, context: dict) -> AgentOutput:
    stats = compute_risk_stats(context.get("ohlcv"))
    if stats is None:
        return neutral_fallback(AGENT_NAME, "Not enough OHLCV history to compute risk statistics yet.")

    user_prompt = f"Stock: {symbol}\nRisk statistics: {stats}"
    try:
        call = complete(SYSTEM_PROMPT, user_prompt, LLMDirectionalCall)
    except LLMRouterError as exc:
        return neutral_fallback(AGENT_NAME, f"LLM unavailable: {exc}")

    return to_agent_output(AGENT_NAME, call)
