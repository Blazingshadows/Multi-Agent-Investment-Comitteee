"""Runs the actual dashboard.py script via Streamlit's AppTest harness,
mocking httpx.get so this runs offline — no live API server, no real
network/LLM calls, no dependency on rate limits. Live rendering against a
real server was verified manually (see PROJECT.md).
"""

from unittest.mock import MagicMock, patch

import httpx
from streamlit.testing.v1 import AppTest

FAKE_STATUS = {
    "last_cycle_ts": "2026-07-08T09:20:00",
    "squared_off": False,
    "watchlist": ["INFY", "TCS"],
    "cycle_interval_seconds": 300,
}
FAKE_DECISION = {
    "cycle_ts": "2026-07-08T09:20:00",
    "stock": "INFY",
    "consensus_verdict": "BUY",
    "action_taken": "BUY",
    "directional_confidence_score": 0.42,
    "consensus_reasoning": "BUY driven by Technical and Forecasting.",
    "agent_recommendations": [
        {"agent": "Technical", "direction": 1, "confidence": 0.8, "reasoning": "RSI bullish", "evidence": []}
    ],
    "weight_breakdown": [
        {
            "agent": "Technical", "expertise": 1.0, "historical_reliability": 0.5, "herding_penalty": 0.0,
            "trust": 0.5, "relevance": 1.0, "agreement_live": 0.8, "w_raw": 0.4, "w_normalized": 1.0,
            "signed_vote": 0.8,
        }
    ],
    "alternative_stocks_considered": ["TCS"],
    "critic_feedback": [{"critic": "Devil's Advocate", "verdict": "No credible objection.", "alternative_stocks": []}],
    "expected_risk_return": {"expected_return": 0.02, "expected_drawdown": 0.05, "risk_score": 0.3, "sharpe_est": None},
    "cost_breakdown": None,
    "qty": 5,
    "price": 1500.0,
}
FAKE_SUMMARY = {
    "final_portfolio_value": 10500.0,
    "net_profit": 500.0,
    "portfolio_growth_pct": 5.0,
    "trade_history": [{"ts": "2026-07-08T09:20:00", "stock": "INFY", "action": "BUY", "qty": 5, "price": 1500.0, "net_cash_flow": -7530.0}],
    "decision_log": [FAKE_DECISION],
}
FAKE_PORTFOLIO = {
    "cash": 2500.0,
    "positions": {"INFY": 5},
    "latest_value": 10500.0,
    "curve": [{"ts": "2026-07-08T09:20:00", "cash": 2500.0, "positions": {"INFY": 5}, "portfolio_value": 10500.0, "net_pnl": 500.0}],
}

ROUTES = {
    "/api/status": FAKE_STATUS,
    "/api/summary": FAKE_SUMMARY,
    "/api/portfolio": FAKE_PORTFOLIO,
    "/api/decisions": [FAKE_DECISION],
}


def _fake_get(url, params=None, timeout=None):
    for path, body in ROUTES.items():
        if url.endswith(path):
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = body
            response.raise_for_status.return_value = None
            return response
    raise AssertionError(f"unexpected URL in test: {url}")


def test_dashboard_renders_without_exceptions():
    with patch("httpx.get", side_effect=_fake_get):
        at = AppTest.from_file("dashboard.py")
        at.run(timeout=30)

    assert not at.exception
    assert len(at.metric) == 4
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Final Portfolio Value"] == "₹10,500"
    assert metrics["Net Profit"] == "₹500"
    assert metrics["Total Trades"] == "1"


def test_dashboard_renders_decision_log_expander():
    with patch("httpx.get", side_effect=_fake_get):
        at = AppTest.from_file("dashboard.py")
        at.run(timeout=30)

    assert not at.exception
    assert len(at.expander) == 1
    assert "INFY" in at.expander[0].label


def test_dashboard_survives_api_unreachable_including_fragment_rerun():
    """Regression test: _get() used to call st.sidebar.error(...) on
    failure, which raised StreamlitAPIException when hit from inside the
    @st.fragment(run_every=...) block on an auto-refresh rerun (a fragment
    can't safely write to a container established outside itself). Caught
    live when the API was genuinely unreachable during a demo.
    """

    def _always_fails(url, params=None, timeout=None):
        raise httpx.ConnectError("connection refused")

    with patch("httpx.get", side_effect=_always_fails):
        at = AppTest.from_file("dashboard.py")
        at.run(timeout=30)
        assert not at.exception

        # Force the fragment to rerun on its own, the way run_every does —
        # this is exactly the code path that used to crash.
        at.run(timeout=30)
        assert not at.exception
