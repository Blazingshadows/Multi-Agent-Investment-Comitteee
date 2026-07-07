"""Person B owns this file. Minimal SQLite persistence against db/schema.sql
— init + one insert for now. Extend with query helpers for the API/dashboard
(trade history, portfolio curve, decision log export) as the real build
progresses.
"""

import json
import sqlite3
from pathlib import Path

from core.schemas import DecisionLogRow

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_PATH.read_text())
    return conn


def insert_decision_log(conn: sqlite3.Connection, row: DecisionLogRow) -> int:
    cur = conn.execute(
        """
        INSERT INTO decision_log (
            cycle_ts, stock, agent_recommendations, directional_confidence_score,
            weight_breakdown, consensus_verdict, consensus_reasoning,
            alternative_stocks_considered, critic_feedback, expected_risk_return,
            action_taken, qty, price, cost_breakdown, net_cash_flow
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row.cycle_ts.isoformat(),
            row.stock,
            _dump_list(row.agent_recommendations),
            row.directional_confidence_score,
            _dump_list(row.weight_breakdown),
            row.consensus_verdict.value,
            row.consensus_reasoning,
            json.dumps(row.alternative_stocks_considered),
            _dump_list(row.critic_feedback),
            row.expected_risk_return.model_dump_json(),
            row.action_taken.value,
            row.qty,
            row.price,
            row.cost_breakdown.model_dump_json() if row.cost_breakdown else None,
            row.net_cash_flow,
        ),
    )
    conn.commit()
    return cur.lastrowid


def _dump_list(models) -> str:
    return "[" + ",".join(m.model_dump_json() for m in models) + "]"
