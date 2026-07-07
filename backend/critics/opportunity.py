"""Opportunity Critic — scans the other stocks evaluated this cycle for a
stronger alternative than the current proposal. This critic only identifies
and narrates the best alternative; the actual cost-aware SWITCH decision is
core.consensus_engine.evaluate_switch(), called by the orchestrator.
"""

from backend.critics._common import LLMCriticCall
from core.llm_router import LLMRouterError, complete
from core.schemas import CriticFeedback

CRITIC_NAME = "Opportunity Critic"

SYSTEM_PROMPT = (
    "You are the Opportunity Critic on an intraday NSE/BSE investment committee. "
    "You are shown the current stock's consensus and the consensus for every other "
    "stock evaluated this cycle. Identify whether any alternative has a meaningfully "
    "stronger, same-direction conviction than the current proposal."
    "\n\n"
    "Your `verdict` field is READ DIRECTLY by the trading committee's audit log — a "
    "short phrase like 'stronger alternative exists' is USELESS and will be rejected. "
    "Write 2-3 full sentences: name the specific alternative stock(s) with their DCS "
    "values, quantify how much stronger the conviction is, and state whether that gap "
    "is large enough to be worth considering once switching costs are factored in. If "
    "the current proposal is still the best of the set, say so in a full sentence "
    "explaining the comparison, not just 'no'."
)


def review(current_symbol: str, current_dcs: float, other_results: dict[str, float]) -> CriticFeedback:
    """other_results: {symbol: dcs} for every other stock evaluated this cycle."""
    if not other_results:
        return CriticFeedback(critic=CRITIC_NAME, verdict="No other candidates evaluated this cycle to compare against.")

    ranked = sorted(other_results.items(), key=lambda kv: -abs(kv[1]))
    fallback_alternatives = [symbol for symbol, _ in ranked[:3]]

    candidates_desc = "\n".join(f"- {symbol}: DCS={dcs:+.2f}" for symbol, dcs in ranked)
    user_prompt = f"Current stock: {current_symbol} (DCS={current_dcs:+.2f})\n\nOther candidates this cycle:\n{candidates_desc}"

    try:
        call = complete(SYSTEM_PROMPT, user_prompt, LLMCriticCall)
    except LLMRouterError as exc:
        return CriticFeedback(critic=CRITIC_NAME, verdict=f"Critic unavailable: {exc}", alternative_stocks=fallback_alternatives)

    return CriticFeedback(
        critic=CRITIC_NAME, verdict=call.verdict, alternative_stocks=call.alternative_stocks or fallback_alternatives
    )
