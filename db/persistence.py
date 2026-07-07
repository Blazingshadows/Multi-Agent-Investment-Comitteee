"""SQLite persistence against db/schema.sql — writers for all four tables,
plus read helpers the API/dashboard need (trade history, portfolio curve,
decision log export for the end-of-session report).
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from core.schemas import Action, CostBreakdown, DecisionLogRow

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


def insert_trade(
    conn: sqlite3.Connection,
    ts: datetime,
    stock: str,
    action: Action,
    qty: float,
    price: float,
    cost_breakdown: CostBreakdown | None,
    net_cash_flow: float,
    decision_log_id: int | None = None,
) -> int:
    """Only meaningful for BUY/SELL — call once per executed order, including
    each leg of a forced square-off.
    """
    cur = conn.execute(
        """
        INSERT INTO trades (ts, stock, action, qty, price, cost_breakdown, net_cash_flow, decision_log_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ts.isoformat(),
            stock,
            action.value,
            qty,
            price,
            cost_breakdown.model_dump_json() if cost_breakdown else None,
            net_cash_flow,
            decision_log_id,
        ),
    )
    conn.commit()
    return cur.lastrowid


def insert_portfolio_snapshot(
    conn: sqlite3.Connection, ts: datetime, cash: float, positions: dict[str, float], value: float, net_pnl: float
) -> int:
    cur = conn.execute(
        "INSERT INTO portfolio_snapshots (ts, cash, positions, portfolio_value, net_pnl) VALUES (?, ?, ?, ?, ?)",
        (ts.isoformat(), cash, json.dumps(positions), value, net_pnl),
    )
    conn.commit()
    return cur.lastrowid


def get_trade_history(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute(
        "SELECT ts, stock, action, qty, price, cost_breakdown, net_cash_flow, decision_log_id "
        "FROM trades ORDER BY ts"
    )
    rows = _rows_to_dicts(cur)
    for row in rows:
        row["cost_breakdown"] = json.loads(row["cost_breakdown"]) if row["cost_breakdown"] else None
    return rows


def get_portfolio_curve(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute("SELECT ts, cash, positions, portfolio_value, net_pnl FROM portfolio_snapshots ORDER BY ts")
    rows = _rows_to_dicts(cur)
    for row in rows:
        row["positions"] = json.loads(row["positions"]) if row["positions"] else {}
    return rows


def get_decision_log(conn: sqlite3.Connection, limit: int | None = None) -> list[dict]:
    """Newest first — the raw material for the end-of-session "Explainable
    Reasoning for every trade & no-trade" and "Complete Decision Log" fields.
    """
    query = "SELECT * FROM decision_log ORDER BY cycle_ts DESC"
    if limit is not None:
        query += " LIMIT ?"
        cur = conn.execute(query, (limit,))
    else:
        cur = conn.execute(query)

    rows = _rows_to_dicts(cur)
    json_fields = (
        "agent_recommendations",
        "weight_breakdown",
        "alternative_stocks_considered",
        "critic_feedback",
        "expected_risk_return",
        "cost_breakdown",
    )
    for row in rows:
        for field in json_fields:
            if row.get(field):
                row[field] = json.loads(row[field])
    return rows


def _rows_to_dicts(cursor: sqlite3.Cursor) -> list[dict]:
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _dump_list(models) -> str:
    return "[" + ",".join(m.model_dump_json() for m in models) + "]"
