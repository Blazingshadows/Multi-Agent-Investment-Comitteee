"""Fundamental agent — LLM reasoning over valuation/earnings-quality data
pulled from yfinance .info (backend.data.market_data.fetch_fundamentals).
Falls back to a neutral, zero-confidence call if every LLM provider fails
rather than crashing the cycle.
"""

from backend.agents._common import LLMDirectionalCall, neutral_fallback, to_agent_output
from core.llm_router import LLMRouterError, complete
from core.schemas import AgentOutput

AGENT_NAME = "Fundamental"

SYSTEM_PROMPT = (
    "You are the Fundamental Analyst on an intraday NSE/BSE investment committee. "
    "You know your discipline (valuation, earnings quality, balance-sheet strength) is "
    "inherently slow-moving and has low base relevance to a 15-30 minute intraday "
    "holding horizon — say so explicitly if the data doesn't give you a real edge for "
    "THIS timeframe, and keep your confidence low in that case. Only raise confidence "
    "when the fundamentals plausibly explain today's price action (e.g. today is an "
    "earnings/guidance day). Call BULLISH, BEARISH, or NEUTRAL and cite the specific "
    "numbers you used."
)


def analyze(symbol: str, context: dict) -> AgentOutput:
    fundamentals = context.get("fundamentals", {})
    if not fundamentals:
        return neutral_fallback(AGENT_NAME, "No fundamentals data available for this cycle.")

    user_prompt = f"Stock: {symbol}\nFundamentals: {fundamentals}"
    try:
        call = complete(SYSTEM_PROMPT, user_prompt, LLMDirectionalCall)
    except LLMRouterError as exc:
        return neutral_fallback(AGENT_NAME, f"LLM unavailable: {exc}")

    return to_agent_output(AGENT_NAME, call)
