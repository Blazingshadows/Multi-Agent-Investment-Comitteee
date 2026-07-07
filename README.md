# Autonomous Multi-Agent Investment Committee
### PS #10 — Directional Confidence-Aware Consensus for Intraday Paper Trading (NSE/BSE)

## Problem Statement

Retail and institutional investors face an overwhelming amount of market information every day:

- Price movements
- Technical indicators
- News events
- Macroeconomic developments
- Risk factors

Traditional trading systems often rely on a single model or strategy, making decisions based on a limited perspective. Human investment committees solve this problem by bringing together specialists with different viewpoints who debate, challenge assumptions, and collectively arrive at a decision.

The challenge is to build an Autonomous Multi-Agent Investment Committee capable of:

- Analyzing market data
- Forming independent opinions
- Debating conflicting viewpoints
- Reaching a **Directional Confidence-Aware Consensus** (not majority vote, not confidence averaging)
- Managing risk before capital deployment
- Executing paper trades autonomously, intraday, on the Indian market

## Objective

Build an Autonomous Multi-Agent Investment Committee that performs **real-time intraday paper trading on the Indian stock market (NSE/BSE)**, starting with **₹10,000 virtual capital** and **1:2 leverage**. The system must autonomously decide to **BUY, SELL, HOLD, WAIT, or SWITCH** stocks to **maximize end-of-day net profit (after all trading costs)** through explainable multi-agent reasoning.

Explainability is a hard requirement, not a substitute for the profit objective: the committee is judged on realized, cost-adjusted returns, and every one of those decisions must be traceable to the reasoning that produced it.

### Mandatory: Directional Confidence-Aware Consensus

**Simple majority voting or confidence averaging is not allowed.**

The final decision must use a Directional Confidence-Aware Consensus, where each agent's influence dynamically depends on all of the following:

1. **Confidence** — the agent's own certainty in its call
2. **Expertise** — how qualified the agent is for the current situation (e.g. a Technical agent's expertise is discounted on an earnings-driven move)
3. **Historical reliability** — the agent's track record of being directionally correct
4. **Trust** — a slower-moving, calibration-adjusted score built from reliability over time
5. **Context relevance** — how applicable the agent's specialty is to the current market regime
6. **Agreement / disagreement with other agents** — whether the agent's call is corroborated or contradicted by the rest of the committee

Every trade must clearly explain **why** the consensus reached its decision, citing which agents drove it and why they were weighted as they were.

---

## Trading Rules

| Parameter | Value |
|---|---|
| Market | NSE/BSE (Intraday) |
| Virtual Capital | ₹10,000 |
| Leverage | 1:2 |
| Session Length | 4–6 hours (single trading day) |
| Position Closure | All positions must be closed before market close |
| Cost Accounting | Profit calculated after realistic trading costs (brokerage, taxes, charges) |

### Decision Space

Every cycle, the committee must output one of five actions per stock under consideration:

- **BUY** — open/increase a long position
- **SELL** — close/reduce a long position
- **HOLD** — maintain the current position, no change
- **WAIT** — take no position; insufficient conviction or unfavorable setup
- **SWITCH** — exit the current holding and rotate capital into a higher-conviction alternative

---

# Proposed Solution

We propose an autonomous committee of specialized AI investment analysts, each responsible for evaluating the market from a unique perspective, operating on a compressed intraday clock (4–6 hours, NSE/BSE).

Instead of relying on a single predictive model, the system creates a structured decision-making process:

1. Planner agent sets the session's strategic goals and dynamically selects which specialist agents are relevant to consult for the current stock/regime.
2. Gather market information (NSE/BSE live/delayed feeds) for research and data collection.
3. Generate independent recommendations from the selected specialist agents, each backed by a custom-built AI/ML tool.
4. Conduct an argumentation and critic challenge phase (Risk, Profit, Macro, and Opportunity critics).
5. Aggregate recommendations through a **Directional Confidence-Aware Consensus orchestrator**, with agreement/disagreement weighted by a historical alignment graph across agents (never majority vote or plain averaging).
6. Validate decisions through a dedicated risk manager, respecting the ₹10,000 / 1:2 leverage constraints.
7. Execute intraday paper trades.
8. Track performance and continuously update trust, reliability, and calibration scores in shared memory.
9. Close all open positions before market close and report final, cost-adjusted results.

The final output is not simply BUY/SELL/HOLD/WAIT/SWITCH, but a capital allocation recommendation with supporting evidence, agent-level attribution, and risk justification.

---

# Core Design Principles

## Independent Reasoning
Each agent must reason independently before seeing the opinions of other agents.

## Constructive Disagreement
Disagreement is encouraged rather than avoided. Agreement/disagreement between agents is itself a signal that feeds the consensus weighting.

## Dynamic, Multi-Factor Trust
Agent influence is never static and never a single number — it is the product of confidence, expertise, historical reliability, trust, context relevance, and inter-agent agreement.

## Explainability
Every recommendation, every HOLD/WAIT, and every no-trade decision must be traceable to supporting evidence.

## Risk First
No trade can be executed without risk review and approval against the ₹10,000 capital base and 1:2 leverage limit.

## Cost-Aware Profit Maximization
The committee is optimizing for end-of-day net profit after brokerage, taxes, and charges — not gross directional accuracy.

---

# System Architecture

## 1. Investment Planner Agent (Orchestration Layer)

Responsible for strategic asset allocation and goal-setting at the start of, and during, each session. This is the entry point of the workflow and the only agent that controls *which* other agents run.

### Responsibilities
- Sets the session's strategic goals (e.g. capital preservation vs. opportunistic allocation) given the ₹10,000 base and 1:2 leverage limit
- Decides which specialist agents are relevant to consult for the current stock/regime (e.g. skips the Forecasting agent when intraday volume history is too thin; escalates the Macro & Policy agent around a rate decision)
- Manages and re-sequences the workflow (`reassess`) when new information (e.g. a news spike, an unusual price/volume anomaly) invalidates the current plan mid-session
- Triggers a fresh planning cycle when the Debate & Critic Loop (Layer 4) or Risk Management Layer (Layer 6) rejects a proposal outright

### Output
A per-cycle workflow plan: which stocks to evaluate, which specialist agents to activate, and the strategic objective the rest of the committee is optimizing against this cycle.

---

## 2. Market Data Layer

Responsible for collecting and normalizing NSE/BSE data.

### Inputs
- Live / delayed NSE/BSE stock prices
- OHLCV data
- Market indices
- News headlines
- Sector information
- Corporate/policy/geopolitical events

### Output
Unified market context object.

---

## 3. Specialist Agent Layer

Each specialist agent is backed by a **custom-built** AI/ML tool (APIs may be used as inputs, but the analytical logic must be built by the team, not a thin wrapper over a third-party signal). Together, the agents must cover all eight mandatory tool categories from the problem statement.

| Agent | Mandatory Tool Category Covered | Focus | Output |
|---|---|---|---|
| Technical Analyst Agent | Technical Indicator Engine | RSI, MACD, moving averages, momentum | BUY/SELL/HOLD/WAIT/SWITCH, confidence, evidence |
| News & Sentiment Agent | News & Sentiment Analysis | Financial news, earnings, corporate announcements | BUY/SELL/HOLD/WAIT/SWITCH, confidence, evidence |
| Forecasting Agent | Time-Series / DL Forecasting | Short-horizon price/volatility forecasts (intraday) | Directional forecast, confidence, evidence |
| Fundamental Analyst Agent | Fundamental Analysis | Valuation, earnings quality, balance-sheet signals | BUY/SELL/HOLD/WAIT/SWITCH, confidence, evidence |
| Macro & Policy Agent | Policy & Geopolitical Impact | Rate decisions, government policy, geopolitical shocks | BUY/SELL/HOLD/WAIT/SWITCH, confidence, evidence |
| Sector Intelligence Agent | Sector Intelligence | Sector rotation, peer-relative strength | BUY/SELL/HOLD/WAIT/SWITCH, confidence, evidence |
| Opportunity Discovery Agent | Opportunity Discovery | Screens NSE/BSE universe for SWITCH candidates | Ranked alternative stock list, confidence |
| Risk Prediction Agent | Risk Prediction | Forward-looking volatility/drawdown/tail-risk estimate | Risk score, confidence, evidence |

All specialist agents output a directional call from the full five-action space, a confidence score, and supporting evidence. Challenge and critique is not a specialist role here — it is handled by the four dedicated critic agents in the Debate & Critic Loop (Layer 4) below.

---

## 4. Debate & Critic Loop

The debate layer enables structured interaction between agents. This is the crucial disagreement zone: rather than a single generic contrarian, four dedicated critic agents attack the emerging proposal from different angles.

### Critic Agents

| Critic | Attacks the proposal on |
|---|---|
| Risk Critic | Tail risk, drawdown, and volatility assumptions the specialists may have underweighted |
| Profit Critic | Whether the proposal is actually the most profit-maximizing allocation once costs are netted out |
| Macro Critic | Consistency with the current macro/policy regime and sector cycle |
| Opportunity Critic | Actively searches the NSE/BSE universe for a **better** alternative than the current proposal, feeding SWITCH candidates |

### Flow

**Step 1** — Independent recommendations from all specialist agents selected by the Planner.
**Step 2** — Agents review opposing opinions and note agreement/disagreement with each peer.
**Step 3** — Risk, Profit, Macro, and Opportunity critics each challenge the emerging proposal from their respective angle, flagging blind spots and better alternatives.
**Step 4** — Specialist agents may revise confidence scores based on the critic challenges.

### Output
- Final committee recommendations (one of BUY/SELL/HOLD/WAIT/SWITCH)
- Updated confidence levels
- Per-pair agreement/disagreement matrix (feeds the consensus orchestrator)
- Per-critic verdicts and alternative stocks considered (feeds `critic_feedback` and `alternative_stocks_considered` in the per-trade report)

---

## 5. Consensus Orchestrator — Directional Confidence-Aware Consensus

Responsible for synthesizing committee opinions using the mandatory multi-factor formula. **Never** a majority vote or plain confidence average. Implemented as a Directional Trust-Aware Consensus & Evidence Fusion Agent.

### Inputs
- Agent recommendations (BUY/SELL/HOLD/WAIT/SWITCH)
- Confidence scores
- Expertise weighting for current context
- Historical reliability scores
- Trust scores
- Context relevance scores
- Agreement/disagreement matrix from the Debate & Critic Loop
- Pairwise historical alignment graph (Shared Knowledge & Memory Layer, Layer 8)

### Historical Alignment Graph

The Agreement Factor is not just this cycle's agreement — it is informed by a pairwise trust graph built from how agents have historically aligned with each other *and* with the eventual correct call:

- An agent with high confidence that always agrees with the rest of the committee carries **lower** informational value — it adds confirmation, not new signal.
- An agent with moderate confidence that has historically diverged from the crowd, but has been correct when it did, carries **higher** influence — its disagreement is informative.

This graph is persisted and updated after every session in the Shared Knowledge & Memory Layer, so the weighting adapts over time rather than resetting each cycle.

### Agent Influence Formula

```
Agent Influence =
    Confidence
  × Expertise (context-weighted)
  × Historical Reliability
  × Trust Score
  × Context Relevance
  × Agreement Factor (peer corroboration / contradiction)
```

The consensus verdict is the confidence-weighted resolution of all agent influences into a single directional call, with a committee-level confidence score.

### Output — Per-Trade Report

Every BUY / SELL / HOLD / WAIT / SWITCH decision must report:

```json
{
  "symbol": "INFY",
  "decision": "BUY",
  "allocation": 0.25,
  "directional_confidence": 0.74,
  "agent_recommendations": [
    {"agent": "Technical", "call": "BUY", "confidence": 0.82},
    {"agent": "News", "call": "BUY", "confidence": 0.77},
    {"agent": "Macro & Policy", "call": "WAIT", "confidence": 0.65}
  ],
  "consensus_verdict": "BUY",
  "reasoning_and_evidence": "Momentum + earnings beat outweigh sector-level caution; Macro Critic flagged rally sustainability risk.",
  "alternative_stocks_considered": ["TCS", "WIPRO"],
  "critic_feedback": "Macro Critic questioned sustainability of rally given sector weakness; Opportunity Critic confirmed no stronger alternative.",
  "expected_risk_return": {"expected_return": 0.03, "risk_score": 0.41}
}
```

---

## 6. Risk Management Layer

Final approval authority, enforcing the ₹10,000 capital base and 1:2 leverage limit.

### Responsibilities
- Position size control within leveraged capital
- Exposure limits
- Volatility checks (informed by the Risk Prediction Agent)
- Portfolio diversification
- Capital preservation
- Enforcing forced position closure before market close

### Actions
- Approve trade
- Reduce allocation
- Reject trade

---

## 7. Execution Layer

Implemented as the Portfolio Management & Order Execution Agent — the final decision-making step. Responsible for:
- Intraday paper trade execution on NSE/BSE
- Portfolio updates
- Transaction logging
- Closing all open positions before market close
- Deducting brokerage, taxes, and charges from realized P&L
- Writing the realized outcome of each decision back to the Shared Knowledge & Memory Layer so reliability/trust scores reflect it

### Outputs
- Trade history
- Portfolio state
- Performance statistics

---

## 8. Shared Knowledge & Memory Layer

A persistent store that the rest of the committee reads from and writes to across cycles and sessions, so the system's behavior compounds rather than resetting every run.

### Components
- **Financial Data Lake** — raw datasets, research, and reports collected by the Market Data Layer
- **Historical Agent Reliability DB** — per-agent track record, the source of the Historical Reliability score
- **Sector Cycle Analysis** — domain guidelines and best practices reused across sessions by the Sector Intelligence Agent
- **Market Regime Patterns** — past market decisions and outcomes, the source of the Context Relevance score
- **Consensus Log Archive** — the full decision log, used for research benchmarking and computing the Explainability Index

### Read/write pattern
- **Read** by the Planner (Layer 1) to decide which agents to consult, and by the Consensus Orchestrator (Layer 5) to compute Historical Reliability, Trust, Context Relevance, and the historical alignment graph
- **Written** by the Execution Layer (Layer 7) after each trade closes and its outcome is known

---

# Build Scope: Full Architecture vs. 24-Hour MVP

The architecture above is the full target design (matches the reference diagram). The 24-hour hackathon build ships a deliberately trimmed subset of it — see [PROJECT.md](PROJECT.md) for the complete build plan, formula derivation, and rationale for every cut. This table is the single source of truth for what's MVP vs. stretch, so this README and PROJECT.md don't contradict each other under judge questioning.

| Component | Full architecture (this README) | 24h MVP ([PROJECT.md](PROJECT.md)) | Status |
|---|---|---|---|
| Layer 1 — Investment Planner Agent | Dynamically selects which specialists to consult each cycle | Fixed asyncio scheduling loop; the full agent roster runs every cycle | Stretch goal (PROJECT.md §8) |
| Layer 3 — Specialist roster | 8 agents, one per PS tool category | 6 agents — Sector Intelligence is folded into the Opportunity Critic as an input signal, not an independent voting agent | Reduced scope (PROJECT.md §2) |
| Layer 4 — Debate & Critic Loop | 4 named critics: Risk, Profit, Macro, Opportunity | 2 critics: one unified "Devil's-Advocate" (covers risk + profit + macro in one pass) + Opportunity Critic | Reduced scope; 3-way split is a stretch goal (PROJECT.md §8) |
| Layer 5 — Agent Influence Formula | 6 flat factors: Confidence, Expertise, Historical Reliability, Trust, Context Relevance, Agreement | Confidence, Expertise, Relevance, and a live per-cycle Agreement term (`agreement_live_i`, leave-one-out peer corroboration) all stand alone; Historical Reliability is nested inside Trust (`trust_i = reliability_i * (1 − λ·herding_penalty_i)`) alongside a separate historical-agreement (herding) term | Simplified nesting of Reliability into Trust; all 6 dependencies present, including live agreement (PROJECT.md §3) |
| Layer 8 — Shared Knowledge & Memory | Trust/reliability compounds across sessions | Historical reliability resets each session (Laplace prior 0.5); persistence is within-session only, via SQLite | Reduced scope; cross-session warm-start is a stretch goal (PROJECT.md §8) |

Everything else — trading rules, capital/leverage, the 5-action decision space, cost model, per-trade report schema, Risk Management Layer, Execution Layer — is implemented as-is in the MVP; no gap between the two documents.

---

# Dynamic Trust Framework

Each agent maintains independently tracked scores, persisted in the Shared Knowledge & Memory Layer:

- **Historical reliability** — hit rate of directionally correct calls
- **Trust score** — slower-moving composite of reliability and calibration quality over time
- **Context relevance** — how applicable the agent's specialty is right now (e.g. News agent relevance spikes around earnings)
- **Expertise weighting** — situational competence for the current market regime
- **Agreement factor** — how corroborated or contradicted the agent's call is by the rest of the committee this cycle, adjusted by the historical alignment graph

This prevents static voting and enables adaptive, non-uniform committee behavior, in compliance with the "no majority vote / no confidence averaging" mandate.

---

# Evaluation Metrics

## Financial Metrics
- **Final Portfolio Value**
- **Net Profit** (after brokerage, taxes & charges)
- **Portfolio Growth** (% change from ₹10,000 base)
- **Sharpe Ratio**
- **Maximum Drawdown**
- **Win Rate**

## Agent Metrics
- **Agent Accuracy** — % of correct directional predictions
- **Confidence Calibration** — how well confidence aligns with outcomes
- **Trust Stability** — consistency of trust score updates
- **Debate Contribution** — impact of agent challenges on final decisions

## Consensus Metrics
- **Consensus Quality** — performance vs. individual agents
- **Decision Diversity** — measure of disagreement and viewpoint diversity
- **Allocation Efficiency** — capital deployed relative to confidence
- **Explainability Index** — how fully each decision's reasoning traces back to cited evidence and agent-level votes
- **Conflict Resolution Effectiveness** — how well disagreements flagged by the critic agents were resolved in the final consensus, measured against how they played out

## Risk Metrics
- **Risk Compliance** — % of trades approved under risk rules (₹10,000 / 1:2 leverage)
- **Exposure Control** — adherence to position limits
- **Portfolio Stability** — volatility of portfolio returns

---

# Success Criteria

## Minimum Viable Success
- All 8 mandatory AI/ML tool categories represented (custom-built, not solely API wrappers)
- Structured debate and critic workflow
- Directional Confidence-Aware Consensus generation (no majority vote / averaging)
- Risk manager approval layer enforcing ₹10,000 capital and 1:2 leverage
- Intraday paper trading execution on NSE/BSE
- Explainable trade logs, including full per-trade report fields

## Good Success
- Full 6-factor dynamic trust/influence scoring
- Planner-driven dynamic agent selection (not every agent runs every cycle)
- Historical performance tracking backed by a persistent Shared Knowledge & Memory Layer
- Portfolio allocation recommendations across the 5-action decision space (incl. SWITCH)
- Interactive committee dashboard
- All positions verifiably closed before market close each session

## Excellent Success
- Adaptive trust updates informed by a pairwise historical alignment graph across agents
- Multi-stock portfolio management with live SWITCH decisions
- Historical replay evaluation
- Fully explainable committee reasoning for every trade **and** every no-trade (WAIT/HOLD)
- Real-time paper trading demonstration with complete decision log and cost-adjusted P&L

---

# Demo Scenario

## Input
Stock: INFY (NSE)

Market Data:
- Positive earnings
- Rising momentum
- Sector weakness
- Trading session: intraday, 4–6 hours, ₹10,000 virtual capital, 1:2 leverage

## Planner Decision
Planner Agent selects Technical, News, Fundamental, Macro & Policy, Sector Intelligence, and Risk Prediction agents as relevant for INFY given the earnings-driven regime; skips the Forecasting agent this cycle due to insufficient intraday volume history.

## Committee Opinions
- Technical Agent: BUY (0.82)
- News Agent: BUY (0.77)
- Fundamental Agent: BUY (0.70)
- Macro & Policy Agent: WAIT (0.65)
- Sector Intelligence Agent: WAIT (0.60)
- Risk Prediction Agent: Moderate volatility flagged

## Critic Feedback
- Risk Critic: Flags moderate volatility, recommends capping allocation
- Profit Critic: Confirms 20–25% allocation is profit-maximizing after costs at current confidence
- Macro Critic: Notes sector weakness is consistent with Macro & Policy Agent's WAIT, tempers conviction
- Opportunity Critic: Compared against TCS, WIPRO — INFY still preferred given earnings momentum

## Consensus
Recommended Allocation: 25% of virtual capital
Directional Confidence: 74%
Decision: BUY

## Risk Review
Position approved. Allocation reduced to 20% due to volatility and leverage exposure limits.

## Execution
BUY INFY intraday → Portfolio updated → Position flagged for mandatory closure before market close → Decision, including agent-wise votes, critic feedback, and expected risk/return, stored in the audit log.

## End-of-Session Report
- Final Portfolio Value
- Net Profit (after brokerage, taxes & charges)
- Portfolio Growth %
- Trade History
- Explainable reasoning for every trade and every WAIT/HOLD decision
- Complete decision log

---

# Key Innovation

Most AI trading systems attempt to predict the market using a single model. Our system instead models the collaborative decision-making process of a real investment committee — where a Planner agent decides who needs to weigh in, multiple specialist experts debate, four dedicated critics challenge assumptions, agent trust compounds over time through a persistent memory layer and pairwise historical alignment graph, and the committee allocates leveraged capital through a Directional Confidence-Aware Consensus that is explainable down to the individual agent vote — while operating under the real constraints of intraday NSE/BSE paper trading: fixed capital, leverage, a hard session clock, and realistic trading costs.
