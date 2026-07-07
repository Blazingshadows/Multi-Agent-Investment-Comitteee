"""Validates the pure §3 math in isolation from the domain-specific
expertise/relevance calibration (that's exercised via the PROJECT.md worked
example in test_stub_pipeline.py instead). Uses generic agent names not in
BASE_EXPERTISE so every agent gets the same default expertise=1.0.
"""

from core.consensus_engine import evaluate_switch, run_consensus
from core.schemas import Action, AgentOutput, Direction
from db.persistence import init_db


def _agent(name: str, direction: Direction, confidence: float) -> AgentOutput:
    return AgentOutput(agent=name, direction=direction, confidence=confidence, reasoning="test", evidence=[])


def test_unanimous_strong_bullish_is_buy(tmp_path):
    conn = init_db(str(tmp_path / "t.db"))
    agents = [_agent(f"Agent{i}", Direction.BULLISH, 0.9) for i in range(4)]
    result = run_consensus("TEST", agents, conn)
    assert result.consensus_verdict == Action.BUY
    assert result.dcs > 0.8
    assert result.disagreement < 0.01


def test_unanimous_neutral_is_hold(tmp_path):
    conn = init_db(str(tmp_path / "t.db"))
    agents = [_agent(f"Agent{i}", Direction.NEUTRAL, 0.5) for i in range(4)]
    result = run_consensus("TEST", agents, conn)
    assert result.consensus_verdict == Action.HOLD
    assert result.dcs == 0.0


def test_evenly_split_high_confidence_is_wait_not_hold(tmp_path):
    conn = init_db(str(tmp_path / "t.db"))
    agents = [
        _agent("Agent0", Direction.BULLISH, 1.0),
        _agent("Agent1", Direction.BEARISH, 1.0),
        _agent("Agent2", Direction.BULLISH, 1.0),
        _agent("Agent3", Direction.BEARISH, 1.0),
    ]
    result = run_consensus("TEST", agents, conn)
    # DCS nets to ~0 (symmetric split) but disagreement is high -> WAIT
    # (insufficient clarity), not HOLD (genuine neutral agreement).
    assert abs(result.dcs) < 0.15
    assert result.disagreement > 0.05
    assert result.consensus_verdict == Action.WAIT


def test_agreement_live_rewards_corroboration_over_lone_dissent(tmp_path):
    conn = init_db(str(tmp_path / "t.db"))
    agents = [
        _agent("Agent0", Direction.BULLISH, 0.8),
        _agent("Agent1", Direction.BULLISH, 0.8),
        _agent("Agent2", Direction.BULLISH, 0.8),
        _agent("Lone", Direction.BEARISH, 0.8),
    ]
    result = run_consensus("TEST", agents, conn)
    by_agent = {w.agent: w for w in result.weight_breakdown}
    assert by_agent["Agent0"].agreement_live > by_agent["Lone"].agreement_live


def test_evaluate_switch_fires_when_edge_beats_costs():
    assert evaluate_switch(
        current_symbol="A", current_dcs=0.2, current_price=1000.0, current_qty=10,
        alternative_symbol="B", alternative_dcs=0.9, alternative_price=1000.0,
    ) is True


def test_evaluate_switch_does_not_fire_with_no_position():
    assert evaluate_switch(
        current_symbol="A", current_dcs=0.2, current_price=1000.0, current_qty=0,
        alternative_symbol="B", alternative_dcs=0.9, alternative_price=1000.0,
    ) is False


def test_evaluate_switch_does_not_fire_on_tiny_edge():
    assert evaluate_switch(
        current_symbol="A", current_dcs=0.50, current_price=1000.0, current_qty=10,
        alternative_symbol="B", alternative_dcs=0.501, alternative_price=1000.0,
    ) is False
