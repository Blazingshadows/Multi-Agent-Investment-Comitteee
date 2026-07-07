from core.config import BUYING_POWER, MAX_POSITION_FRACTION, MAX_RISK_SCORE
from core.portfolio import ExecutionResult, Portfolio, execute, force_square_off, portfolio_value, review_trade
from core.schemas import Action, ConsensusResult, ExpectedRiskReturn


def _decision(symbol: str, verdict: Action, allocation: float, risk_score: float = 0.3) -> ConsensusResult:
    return ConsensusResult(
        symbol=symbol,
        dcs=0.5,
        disagreement=0.05,
        consensus_verdict=verdict,
        allocation=allocation,
        agent_recommendations=[],
        weight_breakdown=[],
        consensus_reasoning="test",
        expected_risk_return=ExpectedRiskReturn(expected_return=0.02, expected_drawdown=0.05, risk_score=risk_score),
    )


def test_review_trade_approves_within_limits():
    portfolio = Portfolio()
    decision = _decision("INFY", Action.BUY, allocation=0.2)
    reviewed, note = review_trade(portfolio, decision, price=1500.0, current_prices={})
    assert reviewed.allocation == 0.2
    assert "Approved" in note


def test_review_trade_caps_at_max_position_fraction():
    portfolio = Portfolio()
    decision = _decision("INFY", Action.BUY, allocation=0.9)  # above MAX_POSITION_FRACTION
    reviewed, note = review_trade(portfolio, decision, price=1500.0, current_prices={})
    assert reviewed.allocation == MAX_POSITION_FRACTION
    assert "Reduced" in note


def test_review_trade_reduces_for_existing_exposure():
    portfolio = Portfolio(positions={"TCS": 10})
    current_prices = {"TCS": BUYING_POWER * 0.9 / 10}  # TCS position already worth 90% of buying power
    decision = _decision("INFY", Action.BUY, allocation=0.3)
    reviewed, note = review_trade(portfolio, decision, price=1500.0, current_prices=current_prices)
    assert reviewed.allocation < 0.3
    assert reviewed.allocation == round(BUYING_POWER * 0.1, 6) / BUYING_POWER


def test_review_trade_rejects_when_no_headroom():
    portfolio = Portfolio(positions={"TCS": 100})
    current_prices = {"TCS": BUYING_POWER / 100}  # TCS alone already consumes all buying power
    decision = _decision("INFY", Action.BUY, allocation=0.2)
    reviewed, note = review_trade(portfolio, decision, price=1500.0, current_prices=current_prices)
    assert reviewed.allocation == 0.0
    assert reviewed.consensus_verdict == Action.WAIT
    assert "Rejected" in note


def test_review_trade_halves_for_high_risk_score():
    portfolio = Portfolio()
    decision = _decision("INFY", Action.BUY, allocation=0.2, risk_score=MAX_RISK_SCORE + 0.1)
    reviewed, note = review_trade(portfolio, decision, price=1500.0, current_prices={})
    assert reviewed.allocation == 0.1
    assert "risk score" in note


def test_review_trade_noop_for_hold_and_wait():
    portfolio = Portfolio()
    decision = _decision("INFY", Action.HOLD, allocation=0.0)
    reviewed, note = review_trade(portfolio, decision, price=1500.0, current_prices={})
    assert reviewed is decision
    assert "not applicable" in note


def test_review_trade_approves_sell_even_with_no_headroom():
    # SELL closes an existing position (frees exposure), so it should never
    # be capped by the same leverage logic that gates opening a new BUY.
    portfolio = Portfolio(positions={"INFY": 10, "TCS": 100})
    current_prices = {"INFY": 1500.0, "TCS": BUYING_POWER / 100}  # TCS alone consumes all buying power
    decision = _decision("INFY", Action.SELL, allocation=0.2)
    reviewed, note = review_trade(portfolio, decision, price=1500.0, current_prices=current_prices)
    assert reviewed is decision
    assert "Approved" in note


def test_execute_buy_updates_portfolio():
    portfolio = Portfolio()
    decision = _decision("INFY", Action.BUY, allocation=0.1)
    result = execute(portfolio, decision, price=1500.0)
    assert result.qty > 0
    assert portfolio.positions["INFY"] == result.qty
    assert portfolio.cash < BUYING_POWER  # cash paid out


def test_execute_tiny_allocation_becomes_wait():
    portfolio = Portfolio()
    decision = _decision("INFY", Action.BUY, allocation=0.00001)  # rounds to 0 shares at any real price
    result = execute(portfolio, decision, price=1500.0)
    assert result.action_taken == Action.WAIT
    assert result.qty == 0.0
    assert "INFY" not in portfolio.positions


def test_execute_sell_closes_the_full_held_position_not_an_allocation_based_size():
    portfolio = Portfolio(positions={"INFY": 10})
    # allocation is deliberately tiny/unrelated — SELL must ignore it and
    # close the actual held quantity, not recompute a fresh short size.
    decision = _decision("INFY", Action.SELL, allocation=0.01)
    result = execute(portfolio, decision, price=1500.0)
    assert result.qty == 10
    assert portfolio.positions["INFY"] == 0.0
    assert portfolio.cash > BUYING_POWER  # proceeds received


def test_execute_sell_with_no_position_becomes_wait():
    portfolio = Portfolio()
    decision = _decision("INFY", Action.SELL, allocation=0.2)
    result = execute(portfolio, decision, price=1500.0)
    assert result.action_taken == Action.WAIT
    assert result.qty == 0.0


def test_force_square_off_closes_all_positions():
    portfolio = Portfolio(cash=1000.0, positions={"INFY": 10, "TCS": -5})
    current_prices = {"INFY": 1500.0, "TCS": 3500.0}
    results = force_square_off(portfolio, current_prices)

    assert portfolio.positions["INFY"] == 0.0
    assert portfolio.positions["TCS"] == 0.0
    assert len(results) == 2
    symbols_closed = {symbol for symbol, _ in results}
    assert symbols_closed == {"INFY", "TCS"}


def test_portfolio_value_sums_cash_and_positions():
    portfolio = Portfolio(cash=1000.0, positions={"INFY": 10})
    value = portfolio_value(portfolio, {"INFY": 1500.0})
    assert value == 1000.0 + 10 * 1500.0
