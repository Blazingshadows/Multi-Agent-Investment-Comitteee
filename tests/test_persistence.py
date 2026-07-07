from datetime import datetime

from core.schemas import Action, ConsensusResult, CostBreakdown, DecisionLogRow, ExpectedRiskReturn
from db.persistence import (
    get_decision_log,
    get_portfolio_curve,
    get_trade_history,
    init_db,
    insert_decision_log,
    insert_portfolio_snapshot,
    insert_trade,
)


def _cost() -> CostBreakdown:
    return CostBreakdown(brokerage=15.0, stt=6.25, exchange_txn_charges=0.75, sebi_charges=0.03, stamp_duty=0.9, gst=2.84, slippage=5.0)


def test_trade_round_trip(tmp_path):
    conn = init_db(str(tmp_path / "t.db"))
    insert_trade(conn, datetime(2026, 7, 8, 9, 20), "INFY", Action.BUY, qty=10, price=1500.0, cost_breakdown=_cost(), net_cash_flow=-15030.77)

    history = get_trade_history(conn)
    assert len(history) == 1
    assert history[0]["stock"] == "INFY"
    assert history[0]["action"] == "BUY"
    assert history[0]["cost_breakdown"]["brokerage"] == 15.0


def test_portfolio_snapshot_round_trip(tmp_path):
    conn = init_db(str(tmp_path / "t.db"))
    insert_portfolio_snapshot(conn, datetime(2026, 7, 8, 9, 20), cash=5000.0, positions={"INFY": 10}, value=20000.0, net_pnl=500.0)

    curve = get_portfolio_curve(conn)
    assert len(curve) == 1
    assert curve[0]["portfolio_value"] == 20000.0
    assert curve[0]["positions"] == {"INFY": 10}


def test_decision_log_round_trip_and_limit(tmp_path):
    conn = init_db(str(tmp_path / "t.db"))
    result = ConsensusResult(
        symbol="INFY", dcs=0.5, disagreement=0.05, consensus_verdict=Action.BUY, allocation=0.2,
        agent_recommendations=[], weight_breakdown=[], consensus_reasoning="test",
        expected_risk_return=ExpectedRiskReturn(expected_return=0.02, expected_drawdown=0.05, risk_score=0.3),
    )
    for i in range(3):
        row = DecisionLogRow.from_consensus(
            cycle_ts=datetime(2026, 7, 8, 9, 15 + i), stock="INFY", result=result,
            action_taken=Action.BUY, qty=10, price=1500.0, cost_breakdown=_cost(), net_cash_flow=-15030.77,
        )
        insert_decision_log(conn, row)

    all_rows = get_decision_log(conn)
    assert len(all_rows) == 3
    assert all_rows[0]["expected_risk_return"]["risk_score"] == 0.3  # parsed from JSON, not a raw string
    assert all_rows[0]["consensus_verdict"] == "BUY"

    limited = get_decision_log(conn, limit=1)
    assert len(limited) == 1
