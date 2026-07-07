"""Validates the trust_store math against the PS's own worked example:
"Agent A: high confidence but always agrees -> lower influence" vs
"Agent B: moderate confidence but historically right when it disagrees ->
higher influence." Not just a schema-shape test — asserts the actual numbers.
"""

from datetime import datetime, timedelta

from core.schemas import Direction
from core.trust_store import herding_penalty, historical_reliability, record_prediction, resolve_pending_predictions, trust_score
from db.persistence import init_db

BASE = datetime(2026, 7, 8, 9, 15)
BULLISH, BEARISH = Direction.BULLISH, Direction.BEARISH

# Agent A always sides with the crowd (agrees every cycle) but is only right
# when the crowd happens to be right (2/5).
# Agent B breaks from the crowd in cycles 2-4 and is right every single time.
# Agent C is the "crowd" tiebreaker, always bullish.
DIRECTIONS_A = [BULLISH, BULLISH, BULLISH, BULLISH, BEARISH]
DIRECTIONS_B = [BULLISH, BEARISH, BEARISH, BEARISH, BEARISH]
DIRECTIONS_C = [BULLISH, BULLISH, BULLISH, BULLISH, BULLISH]
ACTUALS = [BULLISH, BEARISH, BEARISH, BEARISH, BEARISH]
PRICE_PAIRS = [(100, 101), (101, 100), (100, 99), (99, 98), (98, 97)]


def _seeded_db(tmp_path):
    conn = init_db(str(tmp_path / "trust.db"))
    for i in range(5):
        cycle_ts = BASE + timedelta(minutes=5 * i)
        record_prediction(conn, cycle_ts, "TEST", "A", DIRECTIONS_A[i], 0.7)
        record_prediction(conn, cycle_ts, "TEST", "B", DIRECTIONS_B[i], 0.6)
        record_prediction(conn, cycle_ts, "TEST", "C", DIRECTIONS_C[i], 0.5)
        price_before, price_now = PRICE_PAIRS[i]
        resolve_pending_predictions(conn, "TEST", price_before, price_now)
    return conn


def test_always_agrees_gets_penalized(tmp_path):
    conn = _seeded_db(tmp_path)
    # A matches the cycle majority in all 5 cycles -> 100% agreement rate,
    # above the 0.8 threshold -> maximal penalty.
    assert herding_penalty(conn, "A") == 1.0


def test_frequent_dissenter_gets_no_penalty(tmp_path):
    conn = _seeded_db(tmp_path)
    # B matches the majority in only 2/5 cycles -> below the 0.8 threshold.
    assert herding_penalty(conn, "B") == 0.0


def test_reliability_reflects_actual_correctness(tmp_path):
    conn = _seeded_db(tmp_path)
    # A correct in cycles 1 and 5 only (2/5); B correct every cycle (5/5).
    reliability_a = historical_reliability(conn, "A")
    reliability_b = historical_reliability(conn, "B")
    assert reliability_a == (2 + 0.5) / (5 + 1)
    assert reliability_b == (5 + 0.5) / (5 + 1)
    assert reliability_b > reliability_a


def test_trust_favors_the_profitable_dissenter_over_the_herder(tmp_path):
    conn = _seeded_db(tmp_path)
    trust_a = trust_score(conn, "A")
    trust_b = trust_score(conn, "B")
    # This is the PS's own example, reproduced numerically: the always-agrees
    # agent ends up with roughly half the influence of the agent that broke
    # from the crowd and was right every time it did.
    assert trust_b > trust_a
    assert abs(trust_a - 0.2083333333333333) < 1e-9
    assert abs(trust_b - 0.9166666666666666) < 1e-9


def test_cold_start_defaults_to_prior(tmp_path):
    conn = init_db(str(tmp_path / "empty.db"))
    assert historical_reliability(conn, "NewAgent") == 0.5
    assert herding_penalty(conn, "NewAgent") == 0.0
    assert trust_score(conn, "NewAgent") == 0.5
