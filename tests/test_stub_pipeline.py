"""Proves the hour-0-2 pipeline shape: fixture -> stub consensus -> execution
(real cost model) -> SQLite persistence -> read-back. Rerun this anytime
either person swaps a stub for real logic — if it still passes, the contract
held and nothing downstream needs to change.
"""

import json
from datetime import datetime
from pathlib import Path

from core.consensus_engine import run_consensus
from core.portfolio import Portfolio, execute
from core.schemas import AgentOutput, DecisionLogRow
from db.persistence import init_db, insert_decision_log

FIXTURE = Path(__file__).parent / "fixtures" / "agent_outputs.json"


def test_stub_pipeline_end_to_end(tmp_path):
    agents = [AgentOutput(**a) for a in json.loads(FIXTURE.read_text())]

    result = run_consensus("INFY", agents)

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

    conn = init_db(str(tmp_path / "test.db"))
    row_id = insert_decision_log(conn, row)

    stored = conn.execute(
        "SELECT stock, consensus_verdict, action_taken, qty > 0, net_cash_flow < 0 "
        "FROM decision_log WHERE id = ?",
        (row_id,),
    ).fetchone()

    assert stored == ("INFY", "BUY", "BUY", 1, 1)  # BUY sizes a positive qty and pays cash out
