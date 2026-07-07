"""Macro & Policy agent — LLM reasoning over the static macro calendar
(backend.data.macro_calendar.get_macro_context) merged with any live
context flags the orchestrator adds. Covers both PS categories "Policy &
Geopolitical Impact" and Government Policy under one voice.
"""

from backend.agents._common import LLMDirectionalCall, neutral_fallback, to_agent_output
from core.llm_router import LLMRouterError, complete
from core.schemas import AgentOutput

AGENT_NAME = "Macro & Policy"

SYSTEM_PROMPT = (
    "You are the Macro & Policy Analyst on an intraday NSE/BSE investment committee, "
    "covering RBI policy, government policy, and geopolitical events. Your relevance "
    "spikes sharply on an RBI policy day or during budget week — on an ordinary day "
    "with no macro catalyst, say so explicitly and keep confidence low rather than "
    "inventing a signal. Call BULLISH, BEARISH, or NEUTRAL for the stock given the "
    "current macro backdrop, and cite the specific calendar facts you used."
)


def analyze(symbol: str, context: dict) -> AgentOutput:
    macro_context = context.get("macro_flags", {})
    if not macro_context:
        return neutral_fallback(AGENT_NAME, "No macro calendar context available for this cycle.")

    user_prompt = f"Stock: {symbol}\nMacro calendar context: {macro_context}"
    try:
        call = complete(SYSTEM_PROMPT, user_prompt, LLMDirectionalCall)
    except LLMRouterError as exc:
        return neutral_fallback(AGENT_NAME, f"LLM unavailable: {exc}")

    return to_agent_output(AGENT_NAME, call)
