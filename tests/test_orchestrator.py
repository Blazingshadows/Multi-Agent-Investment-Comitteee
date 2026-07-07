"""Integration test for the full orchestrator cycle — mocks every network
boundary (market data, fundamentals, news, all LLM-backed agents/critics via
their .analyze/.review) but exercises real consensus math, real risk review,
real execution, and real persistence.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from backend import orchestrator
from backend.critics import devils_advocate, opportunity
from core.config import BUYING_POWER, MAX_POSITION_FRACTION
from core.portfolio import Portfolio
from core.schemas import Action, AgentOutput, CriticFeedback, Direction
from db.persistence import get_decision_log, get_portfolio_curve, init_db


def _synthetic_ohlcv(n: int = 40, seed: int = 0, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-07-08 09:15", periods=n, freq="5min")
    close = 1000 + np.cumsum(rng.normal(drift, 1.0, n))
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": rng.integers(10_000, 100_000, n)},
        index=timestamps,
    )


def _patch_data_layer(monkeypatch, ohlcv_by_symbol: dict[str, pd.DataFrame]):
    monkeypatch.setattr(orchestrator, "fetch_ohlcv", lambda symbol, period="5d", interval="5m": ohlcv_by_symbol[symbol])
    monkeypatch.setattr(orchestrator, "_get_fundamentals", lambda symbol: {})
    monkeypatch.setattr(orchestrator, "_get_news", lambda symbol, company_name: [])


def _patch_agents(monkeypatch, direction: Direction, confidence: float):
    for i, module in enumerate(orchestrator.SPECIALIST_AGENTS):
        agent_name = f"Agent{i}"

        def fake_analyze(symbol, context, name=agent_name):
            return AgentOutput(agent=name, direction=direction, confidence=confidence, reasoning="test", evidence=[])

        monkeypatch.setattr(module, "analyze", fake_analyze)


def _patch_critics(monkeypatch):
    monkeypatch.setattr(devils_advocate, "review", lambda *a, **kw: CriticFeedback(critic="Devil's Advocate", verdict="test"))
    monkeypatch.setattr(opportunity, "review", lambda *a, **kw: CriticFeedback(critic="Opportunity Critic", verdict="test"))


def test_run_cycle_unanimous_bullish_buys_both_within_position_cap(tmp_path, monkeypatch):
    watchlist = ["SYM_A", "SYM_B"]
    ohlcv = {"SYM_A": _synthetic_ohlcv(seed=1), "SYM_B": _synthetic_ohlcv(seed=2)}
    _patch_data_layer(monkeypatch, ohlcv)
    _patch_agents(monkeypatch, Direction.BULLISH, 0.9)
    _patch_critics(monkeypatch)

    conn = init_db(str(tmp_path / "cycle.db"))
    portfolio = Portfolio()

    rows = orchestrator.run_cycle(conn, portfolio, datetime(2026, 7, 8, 9, 20), watchlist=watchlist)

    assert len(rows) == 2
    assert all(row.action_taken == Action.BUY for row in rows)
    # both capped at MAX_POSITION_FRACTION by the Risk Management Layer
    for symbol in watchlist:
        assert portfolio.positions[symbol] > 0
    total_exposure = sum(portfolio.positions[s] * ohlcv[s]["close"].iloc[-1] for s in watchlist)
    # whole-share rounding can push each leg up to ~1 share's value over the
    # exact cap; use the actual per-symbol price as the slack bound
    rounding_slack = sum(ohlcv[s]["close"].iloc[-1] for s in watchlist)
    assert total_exposure <= 2 * MAX_POSITION_FRACTION * BUYING_POWER + rounding_slack

    decision_log = get_decision_log(conn)
    assert len(decision_log) == 2
    assert len(get_portfolio_curve(conn)) == 1


def test_run_cycle_unanimous_neutral_holds_and_persists_no_trade(tmp_path, monkeypatch):
    watchlist = ["SYM_A"]
    ohlcv = {"SYM_A": _synthetic_ohlcv(seed=3)}
    _patch_data_layer(monkeypatch, ohlcv)
    _patch_agents(monkeypatch, Direction.NEUTRAL, 0.5)
    _patch_critics(monkeypatch)

    conn = init_db(str(tmp_path / "cycle.db"))
    portfolio = Portfolio()

    rows = orchestrator.run_cycle(conn, portfolio, datetime(2026, 7, 8, 9, 20), watchlist=watchlist)

    assert len(rows) == 1
    assert rows[0].action_taken == Action.HOLD
    assert portfolio.positions == {}
    # a no-trade cycle still gets logged — required for every "no-trade" explanation
    assert len(get_decision_log(conn)) == 1


def test_run_cycle_skips_symbol_on_data_fetch_failure(tmp_path, monkeypatch):
    watchlist = ["BROKEN", "SYM_B"]
    ohlcv = {"SYM_B": _synthetic_ohlcv(seed=6)}

    def flaky_fetch(symbol, period="5d", interval="5m"):
        if symbol == "BROKEN":
            raise RuntimeError("possibly delisted; no price data found")
        return ohlcv[symbol]

    monkeypatch.setattr(orchestrator, "fetch_ohlcv", flaky_fetch)
    monkeypatch.setattr(orchestrator, "_get_fundamentals", lambda symbol: {})
    monkeypatch.setattr(orchestrator, "_get_news", lambda symbol, company_name: [])
    _patch_agents(monkeypatch, Direction.NEUTRAL, 0.5)
    _patch_critics(monkeypatch)

    conn = init_db(str(tmp_path / "cycle.db"))
    portfolio = Portfolio()

    rows = orchestrator.run_cycle(conn, portfolio, datetime(2026, 7, 8, 9, 20), watchlist=watchlist)

    assert len(rows) == 1
    assert rows[0].stock == "SYM_B"


def test_run_cycle_switches_out_of_a_weak_held_position(tmp_path, monkeypatch):
    watchlist = ["WEAK", "STRONG"]
    ohlcv = {"WEAK": _synthetic_ohlcv(seed=4), "STRONG": _synthetic_ohlcv(seed=5)}
    _patch_data_layer(monkeypatch, ohlcv)
    _patch_critics(monkeypatch)

    # WEAK's agents mildly bearish/neutral, STRONG's agents strongly bullish
    def fake_analyze_factory(direction, confidence):
        def fake_analyze(symbol, context):
            lean = direction if symbol == "STRONG" else Direction.BEARISH
            conf = confidence if symbol == "STRONG" else 0.2
            return AgentOutput(agent="Agent", direction=lean, confidence=conf, reasoning="test", evidence=[])
        return fake_analyze

    for module in orchestrator.SPECIALIST_AGENTS:
        monkeypatch.setattr(module, "analyze", fake_analyze_factory(Direction.BULLISH, 0.95))

    conn = init_db(str(tmp_path / "cycle.db"))
    portfolio = Portfolio(positions={"WEAK": 10})  # already holding a weak position

    rows = orchestrator.run_cycle(conn, portfolio, datetime(2026, 7, 8, 9, 20), watchlist=watchlist)

    weak_row = next(r for r in rows if r.stock == "WEAK")
    assert weak_row.consensus_verdict == "SWITCH"
    assert weak_row.action_taken == "SELL"  # mechanically executes as closing the weak leg
    assert portfolio.positions["WEAK"] == 0.0
