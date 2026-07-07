"""Tests the REST handlers directly against manually-seeded state, bypassing
`lifespan` (which would otherwise immediately fire a real orchestrator cycle
— real network/LLM calls — the instant the app starts).
"""

from datetime import datetime

from fastapi.testclient import TestClient

import api.main as api_module
from core.config import CAPITAL
from core.portfolio import Portfolio
from core.schemas import Action, ConsensusResult, DecisionLogRow, ExpectedRiskReturn
from db.persistence import init_db, insert_decision_log


def _seed_state(tmp_path):
    db_path = str(tmp_path / "api_test.db")
    conn = init_db(db_path)
    portfolio = Portfolio(cash=15000.0, positions={"INFY": 5})
    api_module._state["conn"] = conn
    api_module._state["portfolio"] = portfolio
    api_module.DB_PATH = db_path  # _read_conn() opens fresh connections against this
    api_module._state["last_cycle_ts"] = "2026-07-08T09:20:00"
    api_module._state["squared_off"] = False
    return conn, portfolio


def _seed_decision(conn):
    result = ConsensusResult(
        symbol="INFY", dcs=0.5, disagreement=0.05, consensus_verdict=Action.BUY, allocation=0.2,
        agent_recommendations=[], weight_breakdown=[], consensus_reasoning="test",
        expected_risk_return=ExpectedRiskReturn(expected_return=0.02, expected_drawdown=0.05, risk_score=0.3),
    )
    row = DecisionLogRow.from_consensus(
        cycle_ts=datetime(2026, 7, 8, 9, 20), stock="INFY", result=result,
        action_taken=Action.BUY, qty=5, price=1500.0, cost_breakdown=None, net_cash_flow=-7500.0,
    )
    insert_decision_log(conn, row)


def test_status_endpoint(tmp_path):
    _seed_state(tmp_path)
    client = TestClient(api_module.app)
    response = client.get("/api/status")
    assert response.status_code == 200
    body = response.json()
    assert body["last_cycle_ts"] == "2026-07-08T09:20:00"
    assert "watchlist" in body


def test_decisions_endpoint(tmp_path):
    conn, _ = _seed_state(tmp_path)
    _seed_decision(conn)
    client = TestClient(api_module.app)
    response = client.get("/api/decisions")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["stock"] == "INFY"


def test_portfolio_endpoint(tmp_path):
    _seed_state(tmp_path)
    client = TestClient(api_module.app)
    response = client.get("/api/portfolio")
    assert response.status_code == 200
    body = response.json()
    assert body["cash"] == 15000.0
    assert body["positions"] == {"INFY": 5}


def test_summary_endpoint_computes_growth(tmp_path):
    conn, _ = _seed_state(tmp_path)
    from db.persistence import insert_portfolio_snapshot

    insert_portfolio_snapshot(conn, datetime(2026, 7, 8, 9, 20), cash=15000.0, positions={"INFY": 5}, value=CAPITAL * 1.1, net_pnl=1000.0)
    client = TestClient(api_module.app)
    response = client.get("/api/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["final_portfolio_value"] == CAPITAL * 1.1
    assert abs(body["portfolio_growth_pct"] - 10.0) < 1e-6


def test_trigger_cycle_calls_run_cycle_and_broadcasts(tmp_path, monkeypatch):
    _seed_state(tmp_path)

    called = {}

    def fake_run_cycle(conn, portfolio, cycle_ts, watchlist=None):
        called["ran"] = True
        return []

    monkeypatch.setattr(api_module, "run_cycle", fake_run_cycle)
    client = TestClient(api_module.app)
    response = client.post("/api/cycle/run")
    assert response.status_code == 200
    assert response.json() == {"decisions": 0}
    assert called.get("ran") is True
