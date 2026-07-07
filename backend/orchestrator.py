"""Ties every layer together: gather data -> run all 6 specialist agents ->
resolve/record trust_store predictions -> consensus per stock -> both
critics -> SWITCH evaluation across the watchlist -> risk review -> execute
-> persist. run_cycle() is the single testable unit; call it repeatedly
(live polling or replay mode) from a thin scheduling loop.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from backend.agents import forecasting, fundamental, macro_policy, risk, sentiment, technical
from backend.critics import devils_advocate, opportunity
from backend.data.macro_calendar import get_macro_context
from backend.data.market_data import fetch_fundamentals, fetch_ohlcv
from backend.data.news_feed import fetch_stock_news
from core.config import WATCHLIST
from core.consensus_engine import evaluate_switch, run_consensus
from core.portfolio import Portfolio, execute, force_square_off, portfolio_value, review_trade
from core.schemas import Action, ConsensusResult, DecisionLogRow, ExpectedRiskReturn
from core.trust_store import record_prediction, resolve_pending_predictions
from db.persistence import insert_decision_log, insert_portfolio_snapshot, insert_trade

SPECIALIST_AGENTS = [technical, forecasting, fundamental, sentiment, macro_policy, risk]

_fundamentals_cache: dict[str, dict] = {}
_news_cache: dict[str, tuple[datetime, list[str]]] = {}
NEWS_CACHE_TTL = timedelta(minutes=10)
_last_price: dict[str, float] = {}


def _get_fundamentals(symbol: str) -> dict:
    """Fundamentals don't move intraday — fetch once per session, not every
    cycle, so we're not re-hitting yfinance .info on every 5-min tick.
    """
    if symbol not in _fundamentals_cache:
        try:
            _fundamentals_cache[symbol] = fetch_fundamentals(symbol)
        except Exception:
            _fundamentals_cache[symbol] = {}
    return _fundamentals_cache[symbol]


def _get_news(symbol: str, company_name: str | None) -> list[str]:
    cached = _news_cache.get(symbol)
    now = datetime.now()
    if cached and now - cached[0] < NEWS_CACHE_TTL:
        return cached[1]
    try:
        headlines = fetch_stock_news(symbol, company_name)
    except Exception:
        headlines = cached[1] if cached else []
    _news_cache[symbol] = (now, headlines)
    return headlines


def build_context(symbol: str) -> dict:
    df = fetch_ohlcv(symbol, period="5d", interval="5m")
    fundamentals = _get_fundamentals(symbol)
    news = _get_news(symbol, fundamentals.get("longName"))
    macro_flags = get_macro_context()

    risk_stats = risk.compute_risk_stats(df)
    consensus_context = dict(macro_flags)
    if risk_stats and risk_stats["risk_score"] > 0.6:
        consensus_context["high_volatility_day"] = True

    return {
        "ohlcv": df,
        "fundamentals": fundamentals,
        "news": news,
        "macro_flags": macro_flags,
        "consensus_context": consensus_context,
        "risk_stats": risk_stats,
    }


def run_all_agents(symbol: str, context: dict) -> list:
    """Runs all 6 specialists concurrently — the 4 LLM-backed ones are
    network-bound, so this is a meaningful speedup over sequential calls.
    """
    with ThreadPoolExecutor(max_workers=len(SPECIALIST_AGENTS)) as pool:
        futures = [pool.submit(module.analyze, symbol, context) for module in SPECIALIST_AGENTS]
        return [f.result() for f in futures]


def _expected_risk_return(context: dict, predicted_return: float | None) -> ExpectedRiskReturn:
    risk_stats = context.get("risk_stats")
    return ExpectedRiskReturn(
        expected_return=predicted_return or 0.0,
        expected_drawdown=abs(risk_stats["max_drawdown_in_window"]) if risk_stats else 0.0,
        risk_score=risk_stats["risk_score"] if risk_stats else 0.5,
    )


def run_cycle(conn, portfolio: Portfolio, cycle_ts: datetime, watchlist: list[str] = WATCHLIST) -> list[DecisionLogRow]:
    contexts: dict[str, dict] = {}
    all_agent_outputs: dict[str, list] = {}
    current_prices: dict[str, float] = {}
    results: dict[str, ConsensusResult] = {}

    # Pass 1: gather data, run specialists, record/resolve trust predictions,
    # compute a preliminary consensus per stock. A single symbol's data fetch
    # failing (transient yfinance/network hiccup) skips that symbol for this
    # cycle rather than crashing the whole cycle for every other stock.
    for symbol in watchlist:
        try:
            context = build_context(symbol)
            price = float(context["ohlcv"]["close"].iloc[-1])
        except Exception as exc:
            print(f"[orchestrator] skipping {symbol} this cycle — data fetch failed: {exc}")
            continue

        contexts[symbol] = context
        current_prices[symbol] = price

        if symbol in _last_price:
            resolve_pending_predictions(conn, symbol, _last_price[symbol], current_prices[symbol])
        _last_price[symbol] = current_prices[symbol]

        agent_outputs = run_all_agents(symbol, context)
        all_agent_outputs[symbol] = agent_outputs

        # Consensus must be weighted using only *prior* cycles' history —
        # run it before recording this cycle's own predictions, otherwise
        # herding_penalty()/historical_reliability() would see this cycle's
        # own not-yet-resolved votes and self-referentially weight itself.
        predicted_return = forecasting.predict_return(symbol, context)
        results[symbol] = run_consensus(
            symbol,
            agent_outputs,
            conn,
            context=context["consensus_context"],
            expected_risk_return=_expected_risk_return(context, predicted_return),
        )

        for output in agent_outputs:
            record_prediction(conn, cycle_ts, symbol, output.agent, output.direction, output.confidence)

    # Pass 2: critics — each needs the full picture from pass 1. Iterates
    # over `results` (symbols that survived pass 1), not the raw watchlist.
    final_results: dict[str, ConsensusResult] = {}
    for symbol in results:
        prelim = results[symbol]
        da_feedback = devils_advocate.review(symbol, all_agent_outputs[symbol], prelim.consensus_verdict.value, prelim.dcs)
        other_dcs = {s: r.dcs for s, r in results.items() if s != symbol}
        opp_feedback = opportunity.review(symbol, prelim.dcs, other_dcs)

        final_results[symbol] = prelim.model_copy(
            update={
                "critic_feedback": [da_feedback, opp_feedback],
                "alternative_stocks_considered": opp_feedback.alternative_stocks,
            }
        )

    # Pass 3: SWITCH — only meaningful for symbols we currently hold.
    for held_symbol, qty in list(portfolio.positions.items()):
        if qty <= 0 or held_symbol not in final_results:
            continue
        candidates = {s: r.dcs for s, r in final_results.items() if s != held_symbol}
        if not candidates:
            continue
        best_symbol, best_dcs = max(candidates.items(), key=lambda kv: kv[1])
        if evaluate_switch(
            held_symbol, final_results[held_symbol].dcs, current_prices[held_symbol], qty,
            best_symbol, best_dcs, current_prices[best_symbol],
        ):
            final_results[held_symbol] = final_results[held_symbol].model_copy(
                update={
                    "consensus_verdict": Action.SWITCH,
                    "switch_target": best_symbol,
                    "consensus_reasoning": (
                        f"SWITCH out of {held_symbol} into {best_symbol}: DCS edge "
                        f"{best_dcs - final_results[held_symbol].dcs:+.2f} clears round-trip costs. "
                        + final_results[held_symbol].consensus_reasoning
                    ),
                }
            )

    # Pass 4: risk review, execution, persistence. SWITCH executes as a SELL
    # of the held leg here; the target symbol's own row this same cycle
    # carries whatever BUY conviction it already earned on its own merits.
    decision_rows = []
    for symbol, result in final_results.items():
        execution_decision = result
        if result.consensus_verdict == Action.SWITCH:
            execution_decision = result.model_copy(update={"consensus_verdict": Action.SELL})

        reviewed, risk_note = review_trade(portfolio, execution_decision, current_prices[symbol], current_prices)
        reviewed = reviewed.model_copy(
            update={"consensus_reasoning": reviewed.consensus_reasoning + f" [Risk Layer: {risk_note}]"}
        )
        execution = execute(portfolio, reviewed, current_prices[symbol])

        row = DecisionLogRow.from_consensus(
            cycle_ts=cycle_ts, stock=symbol, result=result if result.consensus_verdict == Action.SWITCH else reviewed,
            action_taken=execution.action_taken, qty=execution.qty, price=execution.price,
            cost_breakdown=execution.cost_breakdown, net_cash_flow=execution.net_cash_flow,
        )
        decision_log_id = insert_decision_log(conn, row)

        if execution.action_taken in (Action.BUY, Action.SELL):
            insert_trade(
                conn, cycle_ts, symbol, execution.action_taken, execution.qty, execution.price,
                execution.cost_breakdown, execution.net_cash_flow, decision_log_id,
            )
        decision_rows.append(row)

    value = portfolio_value(portfolio, current_prices)
    insert_portfolio_snapshot(conn, cycle_ts, portfolio.cash, dict(portfolio.positions), value, value - portfolio.cash)

    return decision_rows


def run_square_off(conn, portfolio: Portfolio, cycle_ts: datetime) -> None:
    """Call once at SESSION_SQUARE_OFF — closes every open position and logs
    each leg as a trade."""
    current_prices = {symbol: fetch_ohlcv(symbol, period="1d", interval="5m")["close"].iloc[-1] for symbol in portfolio.positions if portfolio.positions[symbol] != 0}
    for symbol, execution in force_square_off(portfolio, current_prices):
        insert_trade(conn, cycle_ts, symbol, execution.action_taken, execution.qty, execution.price, execution.cost_breakdown, execution.net_cash_flow)
