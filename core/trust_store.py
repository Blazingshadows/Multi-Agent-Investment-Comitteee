"""Reads/writes the agent_predictions table (db/schema.sql) — the source of
historical_reliability_i and herding_penalty_i from §3. Both are session-
scoped by design: the DB starts fresh each session (per PROJECT.md §8, cross-
session persistence is a stretch goal), so every query here naturally covers
"this session" only.
"""

import sqlite3
from collections import Counter, defaultdict
from datetime import datetime

from core.config import HERDING_PENALTY_MIN_AGREEMENT_RATE, LAMBDA_HERDING
from core.schemas import Direction


def record_prediction(
    conn: sqlite3.Connection, cycle_ts: datetime, stock: str, agent: str, direction: Direction, confidence: float
) -> int:
    """Insert a new, not-yet-resolved prediction. Call once per agent per
    stock per cycle, right after that agent's analyze() returns.
    """
    cur = conn.execute(
        "INSERT INTO agent_predictions (cycle_ts, stock, agent, direction, confidence) VALUES (?, ?, ?, ?, ?)",
        (cycle_ts.isoformat(), stock, agent, direction.value, confidence),
    )
    conn.commit()
    return cur.lastrowid


def resolve_pending_predictions(conn: sqlite3.Connection, stock: str, price_before: float, price_now: float) -> int:
    """Call once per stock per cycle, before recording that cycle's new
    predictions. Resolves every still-unresolved prediction for `stock`
    against the price move realized since it was made — this is what "checked
    at the next cycle" means in the §3 definition of historical_reliability_i.
    Returns the number of rows resolved.
    """
    if price_now > price_before:
        actual = Direction.BULLISH
    elif price_now < price_before:
        actual = Direction.BEARISH
    else:
        actual = Direction.NEUTRAL

    rows = conn.execute(
        "SELECT id, direction FROM agent_predictions WHERE stock = ? AND outcome_direction IS NULL",
        (stock,),
    ).fetchall()

    for row_id, predicted_direction in rows:
        correct = 1 if predicted_direction == actual.value else 0
        conn.execute(
            "UPDATE agent_predictions SET outcome_direction = ?, correct = ? WHERE id = ?",
            (actual.value, correct, row_id),
        )
    conn.commit()
    return len(rows)


def historical_reliability(conn: sqlite3.Connection, agent: str, prior: float = 0.5, alpha: float = 1.0) -> float:
    """Laplace-smoothed hit rate — starts at `prior` (0.5) when an agent has
    no resolved predictions yet, so a cold start never zeroes out an agent's
    influence.
    """
    total, correct_sum = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(correct), 0) FROM agent_predictions WHERE agent = ? AND correct IS NOT NULL",
        (agent,),
    ).fetchone()
    return (correct_sum + alpha * prior) / (total + alpha)


def herding_penalty(
    conn: sqlite3.Connection, agent: str, min_agreement_rate: float = HERDING_PENALTY_MIN_AGREEMENT_RATE
) -> float:
    """Fraction of this agent's calls that matched the cycle-level majority
    direction, scaled to 0 below `min_agreement_rate` and to (0, 1] above it
    — per §3, only penalize agents that agree with the crowd almost always.
    """
    all_rows = conn.execute("SELECT cycle_ts, stock, direction FROM agent_predictions").fetchall()
    if not all_rows:
        return 0.0

    majority_by_cycle: dict[tuple[str, str], int] = {}
    votes_by_cycle: dict[tuple[str, str], list[int]] = defaultdict(list)
    for cycle_ts, stock, direction in all_rows:
        votes_by_cycle[(cycle_ts, stock)].append(direction)
    for key, votes in votes_by_cycle.items():
        majority_by_cycle[key] = Counter(votes).most_common(1)[0][0]

    agent_rows = conn.execute(
        "SELECT cycle_ts, stock, direction FROM agent_predictions WHERE agent = ?", (agent,)
    ).fetchall()
    if not agent_rows:
        return 0.0

    agree_count = sum(
        1 for cycle_ts, stock, direction in agent_rows if direction == majority_by_cycle[(cycle_ts, stock)]
    )
    agreement_rate = agree_count / len(agent_rows)

    if agreement_rate <= min_agreement_rate:
        return 0.0
    return (agreement_rate - min_agreement_rate) / (1 - min_agreement_rate)


def trust_score(conn: sqlite3.Connection, agent: str, lambda_herding: float = LAMBDA_HERDING) -> float:
    """trust_i = historical_reliability_i * (1 - λ * herding_penalty_i), §3."""
    reliability = historical_reliability(conn, agent)
    penalty = herding_penalty(conn, agent)
    return reliability * (1 - lambda_herding * penalty)
