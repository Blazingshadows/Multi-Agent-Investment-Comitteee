"""Proves the full pipeline shape end-to-end: fixture -> real consensus
engine -> execution (real cost model) -> SQLite persistence -> read-back.
Rerun after any change to core/schemas.py or db/schema.sql — if it still
passes, the contract held.
"""

import json
from datetime import datetime
from pathlib import Path

from core.consensus_engine import run_consensus
from core.portfolio import Portfolio, execute
from core.schemas import Action, AgentOutput, DecisionLogRow
from db.persistence import init_db, insert_decision_log

FIXTURE = Path(__file__).parent / "fixtures" / "agent_outputs.json"


def test_stub_pipeline_end_to_end(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    agents = [AgentOutput(**a) for a in json.loads(FIXTURE.read_text())]

    result = run_consensus("INFY", agents, conn)
    assert result.consensus_verdict in (Action.BUY, Action.SELL, Action.HOLD, Action.WAIT)

    portfolio = Portfolio()
    execution = execute(portfolio, result, price=1500.0)

    row = DecisionLogRow.from_consensus(
        cycle_ts=datetime.now(),
        stock="INFY",
        result=result,
        action_taken=execution.action_taken,
        qty=execution.qty,
        price=execution.price,
        cost_breakdown=execution.cost_breakdown,
        net_cash_flow=execution.net_cash_flow,
    )

    row_id = insert_decision_log(conn, row)

    stored = conn.execute(
        "SELECT stock, consensus_verdict, action_taken FROM decision_log WHERE id = ?",
        (row_id,),
    ).fetchone()

    assert stored == ("INFY", result.consensus_verdict.value, execution.action_taken.value)
