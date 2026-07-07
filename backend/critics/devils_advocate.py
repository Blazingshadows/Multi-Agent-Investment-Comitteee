"""Devil's-Advocate Critic — MVP unified critic covering Risk + Profit +
Macro challenge angles in one LLM call (PROJECT.md §2 scope decision;
README's full 4-critic split is a stretch goal). Reviews the emerging
consensus and argues against it from whichever angle is most credible,
rather than inventing a generic objection.
"""

from backend.critics._common import LLMCriticCall
from core.llm_router import LLMRouterError, complete
from core.schemas import AgentOutput, CriticFeedback

CRITIC_NAME = "Devil's Advocate"

SYSTEM_PROMPT = (
    "You are the Devil's Advocate critic on an intraday NSE/BSE investment committee. "
    "You will be shown the committee's individual agent votes and the emerging "
    "consensus verdict for a stock. Argue AGAINST the proposal from whichever of these "
    "angles is most credible given the evidence: "
    "(1) Risk — tail risk/drawdown/volatility the specialists may have underweighted; "
    "(2) Profit — whether this is actually the most profit-maximizing allocation once "
    "realistic trading costs (brokerage, STT, slippage) are netted out on a 15-30 "
    "minute intraday hold; "
    "(3) Macro — whether the proposal is consistent with the current macro/policy regime. "
    "\n\n"
    "Your `verdict` field is READ DIRECTLY by the trading committee's audit log — a "
    "one-word or one-phrase answer like 'REJECT' or 'AGAINST' is USELESS and will be "
    "rejected. Write 2-4 full sentences: name the specific agent(s) whose reasoning "
    "you're challenging, state the concrete angle (risk/profit/macro), and explain the "
    "mechanism in plain terms a trader could act on. If you genuinely find no credible "
    "objection, say so in a full sentence explaining why, don't just write 'none' or "
    "'approve'."
)


def review(symbol: str, agent_outputs: list[AgentOutput], preliminary_verdict: str, preliminary_dcs: float) -> CriticFeedback:
    votes_desc = "\n".join(
        f"- {a.agent}: {a.direction.name} (confidence {a.confidence:.2f}) — {a.reasoning}" for a in agent_outputs
    )
    user_prompt = (
        f"Stock: {symbol}\n"
        f"Emerging consensus: {preliminary_verdict} (DCS={preliminary_dcs:+.2f})\n\n"
        f"Agent votes:\n{votes_desc}"
    )
    try:
        call = complete(SYSTEM_PROMPT, user_prompt, LLMCriticCall)
    except LLMRouterError as exc:
        return CriticFeedback(critic=CRITIC_NAME, verdict=f"Critic unavailable: {exc}")

    return CriticFeedback(critic=CRITIC_NAME, verdict=call.verdict, alternative_stocks=call.alternative_stocks)
