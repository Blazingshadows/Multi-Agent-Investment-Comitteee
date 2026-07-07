-- Frozen alongside core/schemas.py — DecisionLogRow maps 1:1 onto decision_log.
-- json-typed columns store the corresponding pydantic model's .model_dump_json().

CREATE TABLE IF NOT EXISTS decision_log (
    id                              INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_ts                        TEXT NOT NULL,
    stock                           TEXT NOT NULL,
    agent_recommendations           TEXT NOT NULL,  -- json: list[AgentOutput]
    directional_confidence_score    REAL NOT NULL,
    weight_breakdown                TEXT NOT NULL,  -- json: list[AgentWeight]
    consensus_verdict               TEXT NOT NULL,
    consensus_reasoning             TEXT NOT NULL,
    alternative_stocks_considered   TEXT,            -- json: list[str]
    critic_feedback                 TEXT,            -- json: list[CriticFeedback]
    expected_risk_return            TEXT NOT NULL,  -- json: ExpectedRiskReturn
    action_taken                    TEXT NOT NULL,
    qty                             REAL NOT NULL DEFAULT 0,
    price                           REAL NOT NULL DEFAULT 0,
    cost_breakdown                  TEXT,            -- json: CostBreakdown
    net_cash_flow                   REAL NOT NULL DEFAULT 0
);

-- One row per agent per cycle per stock. `correct` is filled in retroactively
-- on the *next* cycle once the actual price move is known — this table is the
-- source of historical_reliability_i in §3.
CREATE TABLE IF NOT EXISTS agent_predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_ts            TEXT NOT NULL,
    stock                TEXT NOT NULL,
    agent               TEXT NOT NULL,
    direction           INTEGER NOT NULL,  -- -1 / 0 / 1
    confidence          REAL NOT NULL,
    outcome_direction   INTEGER,           -- filled in next cycle
    correct             INTEGER            -- 0/1, filled in next cycle
);

-- One row per executed order (BUY/SELL). References the decision_log row that
-- triggered it, so the full reasoning trail is always one join away.
CREATE TABLE IF NOT EXISTS trades (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT NOT NULL,
    stock             TEXT NOT NULL,
    action            TEXT NOT NULL,  -- BUY / SELL
    qty               REAL NOT NULL,
    price             REAL NOT NULL,
    cost_breakdown    TEXT,            -- json: CostBreakdown
    net_cash_flow     REAL NOT NULL,
    decision_log_id   INTEGER REFERENCES decision_log(id)
);

-- Periodic portfolio marks, used to render the dashboard's portfolio curve
-- and to compute Sharpe / max drawdown at session end.
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT NOT NULL,
    cash              REAL NOT NULL,
    positions         TEXT,            -- json: {stock: {qty, avg_price}}
    portfolio_value   REAL NOT NULL,
    net_pnl           REAL NOT NULL
);
