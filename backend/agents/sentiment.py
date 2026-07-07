"""News & Sentiment agent — LLM reasoning over recent headlines
(backend.data.news_feed.fetch_stock_news). Falls back to a neutral,
zero-confidence call if every LLM provider fails or no headlines were found.
"""

from backend.agents._common import LLMDirectionalCall, neutral_fallback, to_agent_output
from core.llm_router import LLMRouterError, complete
from core.schemas import AgentOutput

AGENT_NAME = "Sentiment"

SYSTEM_PROMPT = (
    "You are the News & Sentiment Analyst on an intraday NSE/BSE investment committee. "
    "Given recent headlines mentioning a stock, judge whether the net tone is "
    "BULLISH, BEARISH, or NEUTRAL for the next 15-30 minutes of intraday price action. "
    "Weight company-specific news (earnings, guidance, corporate actions) far more than "
    "generic sector/market commentary. If headlines are old, generic, or contradictory, "
    "say so and keep confidence low. Cite the specific headlines that drove your call."
)


def analyze(symbol: str, context: dict) -> AgentOutput:
    headlines = context.get("news", [])
    if not headlines:
        return neutral_fallback(AGENT_NAME, "No recent headlines found for this stock.")

    user_prompt = f"Stock: {symbol}\nRecent headlines:\n" + "\n".join(f"- {h}" for h in headlines)
    try:
        call = complete(SYSTEM_PROMPT, user_prompt, LLMDirectionalCall)
    except LLMRouterError as exc:
        return neutral_fallback(AGENT_NAME, f"LLM unavailable: {exc}")

    return to_agent_output(AGENT_NAME, call)
