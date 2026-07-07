"""Streamlit dashboard — polls api/main.py over HTTP. Run the API first
(`uvicorn api.main:app`), then `streamlit run dashboard.py`.
"""

import httpx
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Autonomous Multi-Agent Investment Committee", layout="wide")

API_BASE = st.sidebar.text_input("API base URL", "http://127.0.0.1:8000")


def _get(path: str, **params):
    try:
        response = httpx.get(f"{API_BASE}{path}", params=params, timeout=10.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        st.sidebar.error(f"API unreachable: {exc}")
        return None


st.title("Autonomous Multi-Agent Investment Committee")
st.caption("Directional Confidence-Aware Consensus — intraday NSE/BSE paper trading")

with st.sidebar:
    st.header("Session")
    if st.button("Run cycle now", type="primary"):
        try:
            result = httpx.post(f"{API_BASE}/api/cycle/run", timeout=180.0).json()
            st.success(f"Cycle complete — {result['decisions']} decisions logged.")
        except httpx.HTTPError as exc:
            st.error(f"Cycle failed: {exc}")

    status = _get("/api/status")
    if status:
        st.write("**Last cycle:**", status["last_cycle_ts"] or "not run yet")
        st.write("**Squared off:**", status["squared_off"])
        st.write("**Cycle interval:**", f"{status['cycle_interval_seconds']}s")
        st.write("**Watchlist:**")
        st.write(", ".join(status["watchlist"]))


@st.fragment(run_every="10s")
def render_summary():
    summary = _get("/api/summary")
    if not summary:
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Portfolio Value", f"₹{summary['final_portfolio_value']:,.0f}")
    col2.metric("Net Profit", f"₹{summary['net_profit']:,.0f}")
    col3.metric("Portfolio Growth", f"{summary['portfolio_growth_pct']:+.2f}%")
    col4.metric("Total Trades", len(summary["trade_history"]))

    st.subheader("Portfolio Curve")
    portfolio = _get("/api/portfolio")
    if portfolio and portfolio["curve"]:
        curve_df = pd.DataFrame(portfolio["curve"])
        curve_df["ts"] = pd.to_datetime(curve_df["ts"])
        st.line_chart(curve_df.set_index("ts")["portfolio_value"])
    else:
        st.info("No portfolio snapshots yet — run a cycle to populate the curve.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Current Positions")
        if portfolio and portfolio["positions"]:
            st.dataframe(pd.DataFrame(portfolio["positions"].items(), columns=["Symbol", "Qty"]))
        else:
            st.info("No open positions.")

    with col_b:
        st.subheader("Trade History")
        if summary["trade_history"]:
            trades_df = pd.DataFrame(summary["trade_history"])[["ts", "stock", "action", "qty", "price", "net_cash_flow"]]
            st.dataframe(trades_df, hide_index=True)
        else:
            st.info("No trades executed yet.")


render_summary()

st.subheader("Decision Log — Agent Votes, Consensus Reasoning, Critic Feedback")
st.caption("Every trade AND every no-trade decision is logged here, per the PS's explainability requirement.")

decisions = _get("/api/decisions", limit=30)
if not decisions:
    st.info("No decisions logged yet — run a cycle.")
else:
    for decision in decisions:
        verdict = decision["consensus_verdict"]
        action = decision["action_taken"]
        badge = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "🔵" if action == "SWITCH" else "⚪"
        with st.expander(
            f"{badge} {decision['stock']} — {verdict} (action: {action}) — "
            f"DCS={decision['directional_confidence_score']:.2f} — {decision['cycle_ts']}"
        ):
            st.markdown(f"**Consensus reasoning:** {decision['consensus_reasoning']}")

            st.markdown("**Agent-wise recommendations:**")
            agents_df = pd.DataFrame(decision["agent_recommendations"])[["agent", "direction", "confidence", "reasoning"]]
            st.dataframe(agents_df, hide_index=True)

            st.markdown("**Weight breakdown (why each agent was weighted as it was):**")
            weights_df = pd.DataFrame(decision["weight_breakdown"])[
                ["agent", "expertise", "trust", "relevance", "agreement_live", "w_normalized"]
            ]
            st.dataframe(weights_df, hide_index=True)

            if decision["critic_feedback"]:
                st.markdown("**Critic feedback:**")
                for critic in decision["critic_feedback"]:
                    st.markdown(f"- *{critic['critic']}*: {critic['verdict']}")

            if decision["alternative_stocks_considered"]:
                st.markdown("**Alternative stocks considered:** " + ", ".join(decision["alternative_stocks_considered"]))

            risk_return = decision["expected_risk_return"]
            st.markdown(
                f"**Expected risk/return:** return={risk_return['expected_return']:+.3%}, "
                f"drawdown={risk_return['expected_drawdown']:.2%}, risk score={risk_return['risk_score']:.2f}"
            )

            if decision["cost_breakdown"]:
                st.markdown(f"**Cost breakdown:** {decision['cost_breakdown']}")
