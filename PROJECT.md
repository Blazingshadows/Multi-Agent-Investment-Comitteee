# PS #10 — Autonomous Multi-Agent Investment Committee

**Team constraints:** 2–3 people, 24 hours, LLM access = Groq (free tier), OpenRouter, NVIDIA NIM, local Ollama/Qwen (M1 Pro 16GB + networked 24GB GPU box), market data = yfinance/NSE public (delayed), full web dashboard.

---

## 1. Problem, restated in build terms

Run a loop, all trading day, over a watchlist of NSE stocks. Each cycle:

1. Pull fresh market/news/fundamental data for each candidate stock.
2. A fixed set of specialist agents each independently score the stock: direction (BUY/SELL/HOLD-ish) + a confidence + reasoning.
3. A critic/debate step stress-tests the leading proposal and searches for a better alternative stock.
4. A **consensus engine** fuses agent opinions using dynamic, non-uniform weights (confidence × expertise × historical reliability × trust × context relevance × agreement/disagreement) into one decision: BUY / SELL / HOLD / WAIT / SWITCH.
5. Execute the decision against a virtual ₹10,000 (1:2 leverage → ₹20,000 buying power) portfolio, with realistic Indian intraday costs.
6. Log everything — every trade AND every no-trade needs an explanation.
7. At session end (or square-off time), report final portfolio value, net P&L, growth %, full trade history, and the complete decision log.

The **graded differentiator** is #4. Judges explicitly rule out majority voting / plain confidence averaging. This is where most of your engineering care should go — everything else (indicators, sentiment) is table stakes.

---

## 2. Scope decisions (what NOT to build, given 24h / 2-3 people)

Quality of 4–5 agents + a real consensus algorithm beats 8 shallow agents + hand-waved averaging. Cut list:

| Full spec (8 tools) | MVP decision |
|---|---|
| Technical Indicator Engine | **Build.** Core, cheap (no LLM needed for the math). |
| News & Sentiment Analysis | **Build.** |
| Time-Series / DL Forecasting | **Build.** A genuine trained model (not an LLM prompt) — this is the one tool that proves "custom AI/ML" rather than "LLM wrapper." Keep it deliberately small: a gradient-boosted regressor (LightGBM/XGBoost) or a small LSTM/GRU predicting next-15-min return direction + magnitude from lagged OHLCV + indicator features, trained once pre-hackathon-demo on a few weeks of 1-min/5-min history per watchlist stock. Output feeds in exactly like any other agent: `{direction, confidence, reasoning="model predicts +0.4% next 15min, feature importances: ...", evidence}`. |
| Fundamental Analysis | **Build.** |
| Policy & Geopolitical Impact | **Merge into one "Macro & Policy Agent"** with Government Policy — same data sources (RBI, budget, sector news), one prompt. *(Kept merged — see trade-off discussion below.)* |
| Sector Intelligence | **Fold into Opportunity Critic** (relative sector strength becomes one input to "is there a better stock right now") rather than a standalone agent. *(Kept folded — accepted compromise.)* |
| Opportunity Discovery | **Build** as the Opportunity Critic (matches the reference diagram). |
| Risk Prediction | **Build.** |
| 3 separate critics (Risk/Profit/Macro) | **Ship as 1 unified "Devil's Advocate" critic** for MVP (same LLM call, one prompt asks it to argue risk + profit + macro angles). **Stretch goal**: split into 3 prompts — trivial to add back once the pipeline works, don't build it first. |

Net MVP agent roster (**6 agents** + 2 critics):
**Technical · Fundamental · Sentiment/News · Macro&Policy · Risk · Time-Series Forecast** → **Devil's-Advocate Critic + Opportunity Critic** → **Consensus Engine** → **Execution/Portfolio Agent**.

Remaining accepted compromises (unchanged from the earlier review): Policy/Geopolitical still share one voice instead of two, Sector Intelligence is a critic input rather than a full voting member, and there's one unified critic instead of three. These stay cut to protect build time for the consensus engine and the dashboard — revisit only as stretch goals (§8) if hours remain.

Watchlist: pick ~10 liquid large-caps so data is reliable on yfinance and moves are meaningful intraday, e.g. `RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, SBIN, TATAMOTORS, ITC, LT, ADANIENT` (swap freely).

---

## 3. The Directional Confidence-Aware Consensus (the part that gets graded)

This needs to be a real, defensible formula in your code and your pitch — not vibes.

**Per agent *i*, per stock, per cycle, output a triple:**
- `direction_i ∈ {+1, -1, 0}` (bullish / bearish / neutral)
- `confidence_i ∈ [0,1]` (self-reported, the agent must justify this number in its reasoning)
- `signed_vote_i = direction_i * confidence_i` ∈ [-1, 1]

**Weight each agent's influence, not just its confidence:**

```
w_i_raw = expertise_i(context) * trust_i * relevance_i(context) * agreement_live_i ** γ
w_i     = w_i_raw / Σ_j w_j_raw          # normalize to sum to 1
```

- **expertise_i(context)** — static base prior per agent (e.g. Technical agent has high base expertise for short-horizon intraday calls; Fundamental agent has low base expertise intraday but high on earnings day). Store as a small lookup table, multiplied by a context flag (earnings-day, RBI-policy-day, high-volatility-day → boost the matching agent 1.5–2x).
- **trust_i** — `historical_reliability_i * (1 − λ * herding_penalty_i)`.
  - `historical_reliability_i` = Laplace-smoothed hit-rate of that agent's past directional calls *this session* (did price move the direction they called, checked at the next cycle). Start at 0.5 prior so cold-start isn't broken.
  - `herding_penalty_i` = how often this agent simply agrees with the eventual group majority, **across past cycles**. An agent that's always confident but never disagrees adds no information — this is explicitly called out in the reference material ("Agent A: high confidence but always agrees → lower influence"; "Agent B: moderate confidence but historically right when it disagrees → higher influence"). Penalize only when agreement rate is very high (>0.8) to avoid punishing agents for being correctly aligned.
- **relevance_i(context)** — same context table as expertise, kept as a separate factor so you can tune "how much this domain matters right now" independently from "how good this agent generally is."
- **agreement_live_i** — the missing piece: `herding_penalty_i` is entirely backward-looking (a slow trait about the agent), so on its own the formula never lets *this cycle's* corroboration/contradiction affect an agent's influence — which is exactly what the PS mandates ("Agreement/disagreement with other agents" as one of the six factors). Fix it with a **leave-one-out peer-corroboration score**, computed from the raw votes *before* any weighting, so it doesn't create a circular dependency on `w_i` or `DCS`:

  ```
  peer_mean_i      = mean( signed_vote_j for all j ≠ i )      # everyone else's raw vote, this cycle
  agreement_live_i = 1 − |signed_vote_i − peer_mean_i| / 2     # ∈ [0, 1]
  ```

  An agent whose call this cycle matches the rest of the committee gets `agreement_live_i ≈ 1` (corroborated); a lone dissenter gets `agreement_live_i` closer to `0` (contradicted). The exponent `γ` (start at `0.4`) keeps this a *modulating* factor rather than a dominant one — deliberately weak, so a historically-reliable contrarian (high `trust_i` from having been correct when disagreeing, per Agent B above) isn't cancelled out by a low `agreement_live_i` on any single cycle. Tune `γ` up only if the committee is herding on noise; tune it down (toward 0) if it's silencing genuinely informative dissent in your dry runs. This is a distinct signal from `herding_penalty_i`: one is "does this agent's history show independent thinking," the other is "does this agent's *current* call have peer corroboration" — the PS asks for both.

**Fuse into a Directional Confidence Score (DCS):**

```
DCS = Σ_i  w_i * signed_vote_i        # ∈ [-1, +1]
disagreement = Σ_i w_i * (signed_vote_i − DCS)²     # weighted variance
```

**Decision rule:**

```
if   |DCS| < θ_hold  and disagreement < θ_var   → HOLD   (agents genuinely agree it's neutral)
elif |DCS| < θ_hold  and disagreement >= θ_var  → WAIT   (agents disagree a lot — not enough clarity to act)
elif DCS >= θ_buy                                → BUY
elif DCS <= -θ_sell                               → SELL
```

**SWITCH**: if the Opportunity Critic proposes an alternative stock whose DCS beats the current position's DCS by more than the round-trip trading cost (see §4) + a safety margin, switch. Otherwise stay put — this stops the bot from churning on noise.

Suggested starting thresholds: `θ_hold = 0.15`, `θ_buy = θ_sell = 0.35`, tune once you see real numbers from a dry run. Log every raw factor (`expertise`, `trust`, `relevance`, `agreement_live`, `w_i`, `signed_vote_i`) per agent per trade — this *is* your "Consensus Reasoning" + "Directional Confidence Score" output field, and it's what makes the system explainable rather than a black box.

**Worked example (put this in your pitch deck, judges love a concrete trace):**
Technical=BUY(0.8), Fundamental=HOLD(0.4), Sentiment=BUY(0.6), Macro=SELL(0.3), Risk=HOLD(0.5).
Show the weight table (including each agent's `agreement_live_i` relative to the other four), show DCS compute by hand, show the final call. One slide, huge credibility gain — bonus points if one row visibly shows a lower-confidence, high-trust dissenter (Macro=SELL here) still pulling real weight despite low `agreement_live`, because `trust_i` carries it.

---

## 4. Trading cost model (needed for "profit after all trading costs")

Apply on every simulated fill (values are standard NSE intraday equity retail rates — treat as configurable constants):

- Brokerage: `min(₹20, 0.03% * turnover)` per executed order (flat-fee discount broker model)
- STT: `0.025%` of sell-side value only (intraday equity)
- Exchange transaction charges: `~0.00297%` of turnover (both sides)
- SEBI charges: `₹10 per crore` (negligible, include for completeness)
- Stamp duty: `0.003%` of buy-side value
- GST: `18%` on (brokerage + exchange transaction charges)
- Slippage: model `0.02–0.05%` between decision price and simulated fill price (you're not actually hitting an order book)

Wrap all of this in one `cost_model.py` function: `apply_costs(action, qty, price) -> (net_cash_flow, cost_breakdown_dict)`. The cost breakdown dict is a required log field, not just a number.

---

## 5. Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                          │
│  yfinance (1m/5m OHLCV, .NS tickers) · RSS/news feed                 │
│  (Moneycontrol/ET Markets) · yfinance .info (fundamentals) ·         │
│  static macro table (RBI repo rate, budget, sector calendar)         │
└──────────────────────────────────────┬────────────────────────────┘
                                        ▼
┌───────────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR (asyncio loop, one cycle every N min during            │
│  market hours; REPLAY MODE below when market is closed)              │
└──────────────────────────────────────┬────────────────────────────┘
                                        ▼
┌───────────────────────────────────────────────────────────────────┐
│  AGENT LAYER (parallel calls, each -> {direction, confidence,        │
│  reasoning, evidence})                                               │
│  Technical · Fundamental · Sentiment/News · Macro&Policy · Risk ·    │
│  Time-Series Forecast (trained model, not an LLM call)              │
└──────────────────────────────────────┬────────────────────────────┘
                                        ▼
┌───────────────────────────────────────────────────────────────────┐
│  DEBATE / CRITIC LAYER                                               │
│  Devil's-Advocate Critic (attacks the leading proposal)              │
│  Opportunity Critic (scans watchlist for a stronger alternative)     │
└──────────────────────────────────────┬────────────────────────────┘
                                        ▼
┌───────────────────────────────────────────────────────────────────┐
│  CONSENSUS ENGINE (§3 — pure Python math, NOT an LLM call)           │
│  → BUY / SELL / HOLD / WAIT / SWITCH + full factor breakdown          │
└──────────────────────────────────────┬────────────────────────────┘
                                        ▼
┌───────────────────────────────────────────────────────────────────┐
│  EXECUTION / PORTFOLIO AGENT                                         │
│  applies cost model, updates virtual portfolio, enforces leverage    │
│  cap & mandatory square-off before close                             │
└──────────────────────────────────────┬────────────────────────────┘
                                        ▼
┌───────────────────────────────────────────────────────────────────┐
│  PERSISTENCE (SQLite): trades, agent_predictions (for historical     │
│  reliability calc), portfolio_snapshots, decision_log (every cycle,  │
│  including no-trade cycles)                                          │
└──────────────────────────────────────┬────────────────────────────┘
                                        ▼
┌───────────────────────────────────────────────────────────────────┐
│  FASTAPI (REST + WebSocket) ─────────► DASHBOARD                    │
│  serves live cycle results, trade log, portfolio curve               │
└───────────────────────────────────────────────────────────────────┘
```

**LLM routing (matters — you have several free/cheap sources, don't let a rate limit kill your demo):**
Build one thin `llm_router.py` with a common `complete(system, user, json_schema)` interface and try providers in order with automatic fallback:
1. **Groq** (`llama-3.3-70b-versatile` or `llama-3.1-8b-instant`) — primary, very low latency, generous free tier. Use for the high-frequency agent calls (technical narration, sentiment scoring).
2. **Local Ollama on the 24GB GPU box** (Qwen2.5 14B/32B via network) — use for the consensus-adjacent reasoning that benefits from a stronger model (critic debate, final "Consensus Reasoning" paragraph) and as fallback when Groq rate-limits.
3. **OpenRouter free models** — secondary fallback.
4. **Local Ollama on the M1** — last-resort fallback so the demo never fully dies even offline.

Force every agent call to return **structured JSON** (`direction`, `confidence`, `reasoning`, `evidence`) — validate with pydantic, retry once on parse failure, never let a malformed LLM response crash a trading cycle.

**Replay/Simulation Mode (build this, don't skip it):** NSE hours are 9:15–15:30 IST. If your hackathon demo/judging slot falls outside market hours (very likely), you need a mode that replays a recent day's 1-minute bars at accelerated speed (e.g. 1 real second = 1 simulated minute) through the exact same pipeline, so the live dashboard still looks "live" during judging. Build this from day one as just an alternate data source behind the same interface — not a separate code path.

---

## 6. Data schema (also = your "Per Trade Output" requirement)

`decision_log` table — one row per cycle per stock evaluated, whether or not a trade happened:

```
cycle_ts, stock, agent_recommendations (json: [{agent, direction, confidence, reasoning, evidence}]),
directional_confidence_score, weight_breakdown (json), consensus_verdict,
consensus_reasoning (text), alternative_stocks_considered (json), critic_feedback (text),
expected_risk_return (json: {expected_return, expected_drawdown, sharpe_est}),
action_taken, qty, price, cost_breakdown (json), net_cash_flow
```

This one table directly satisfies every bullet in the problem statement's "Per Trade Output" and "Final Output" sections — build the schema first, then make every layer fill in its columns. Final report (portfolio value, net profit, growth %, trade history, decision log) is then just queries/exports over this table.

---

## 7. 24-hour plan for 2–3 people

**If 3 people**, split by layer (minimizes merge conflicts, clean interfaces):
- **P1 — Agents & Consensus (the graded core):** all 5 agents, both critics, consensus engine, historical-reliability/trust tracking.
- **P2 — Data & Execution:** yfinance/news ingestion, replay mode, cost model, portfolio/execution agent, SQLite schema, FastAPI, llm_router with fallback.
- **P3 — Dashboard & Demo:** React (Vite) or Streamlit dashboard, live polling/WebSocket, trade log + portfolio curve + agent-vote panel, plus owns the README/pitch deck and rehearses the demo.

**If 2 people:** merge P2+P3 into one "platform" person (start with Streamlit, not React — upgrade only if hours remain), the other owns Agents & Consensus full-time since that's what's explicitly graded. Both of you have Claude Code — the split below is designed so each of you drives your own Claude Code session against a frozen interface and never waits on the other.

### 7a. Two-person parallel/async plan (both on Claude Code)

**Directory ownership — near-zero file overlap, so there's almost nothing to merge-conflict on:**

| Owner | Files |
|---|---|
| **Person A — Committee** (the graded core) | `backend/agents/*` (6 specialists + Forecasting model), `backend/critics/*` (Devil's-Advocate, Opportunity), `core/consensus_engine.py`, `core/trust_store.py`, `core/llm_router.py` |
| **Person B — Platform** | `backend/data/*` (yfinance, news, replay mode), `core/cost_model.py`, `core/portfolio.py`, `db/schema.sql`, `api/main.py`, `frontend/` or `dashboard.py` (Streamlit) |
| **Shared, written together, then frozen** | `core/schemas.py` — pydantic models for `AgentOutput`, `ConsensusResult`, `DecisionLogRow` (§6), `CostBreakdown` (§4) |

**The only synchronous step:** spend the first 30–45 minutes together writing `core/schemas.py`. Once it's committed, it's the contract — Person A returns `ConsensusResult` objects, Person B's execution/API/dashboard consumes them, and neither needs to read the other's code to keep building.

**Stub-first so nobody blocks on the other:** immediately after `schemas.py`, each person commits a fixture instead of real logic — Person A writes `tests/fixtures/agent_outputs.json` (5–6 fake `AgentOutput` rows) plus a `consensus_engine.py` that returns one hardcoded `ConsensusResult`; Person B builds the `decision_log` writer, portfolio updater, and dashboard entirely against that hardcoded result. By hour 2 both sides have a full pipeline running end-to-end on fake data, in parallel. Real logic replaces the stubs later (Person A swaps in the real §3 formula around hour 6–10, Person B swaps fake JSON for live yfinance/news around the same time) — since the interface never changes, these swaps never touch the other person's files.

**Git workflow for two concurrent Claude Code sessions:**
- Each person works in their own `git worktree` off the same repo (`git worktree add ../committee-agents agents-and-consensus`, `git worktree add ../committee-platform platform-and-dashboard`) — separate working directories means neither Claude Code session ever sees the other's uncommitted edits or risks a mid-edit conflict.
- Two branches, merged to `main` only at the integration checkpoints below — not continuously. With a frozen interface there's usually nothing to resolve.
- `PROJECT.md`, `README.md`, and `core/schemas.py` live in both worktrees automatically (same repo) — that's what keeps two independent Claude Code sessions aligned on the contract without you having to explain it to each other out loud.
- Within your own half, you can run more than one Claude Code session in parallel too, since your files are also mostly independent (e.g. Person A: one session on Technical/Fundamental/Sentiment while a second trains the Forecasting model; Person B: one session on the data layer while a second builds the dashboard).

**Integration checkpoints — the only two moments you need to sync live:**
- **~Hour 6:** both stub pipelines run end-to-end; merge branches, confirm `schemas.py` didn't drift, watch one full fake cycle together.
- **~Hour 14:** both sides now have real logic; merge, run one real dry-run cycle in replay mode end-to-end, fix any schema mismatches that crept in.
- Everything else is async — commit small, push often, no need to narrate progress in real time.

| Hours | Milestone |
|---|---|
| 0–2 | Repo scaffold, watchlist picked, yfinance pull working end-to-end (real + replay mode stubbed) — **also pull 4-6 weeks of 5-min history per watchlist stock now**, forecasting agent needs it early |
| 2–6 | Skeleton pipeline runs with **stub agents** (fixed dummy outputs) all the way to a logged decision — get the full loop working before making any agent smart. In parallel (whoever owns forecasting): feature-engineer lagged OHLCV/indicators and get a first LightGBM/XGBoost fit running offline, even if crude |
| 6–10 | Agents become real (indicators, sentiment LLM calls, fundamentals, risk stats); forecasting model trained, validated on held-out days, wrapped in the same `{direction, confidence, reasoning, evidence}` interface as the LLM agents |
| 10–14 | Consensus engine (§3) + both critics + cost model + portfolio agent |
| 14–18 | Dashboard wired to live backend |
| 18–21 | Full dry run during/replaying market hours, tune thresholds, fix explainability gaps |
| 21–24 | README, deck (reuse your existing PS slide + worked example), rehearse demo, freeze code |

Forecasting-model note: don't reach for a deep LSTM under time pressure — a gradient-boosted tree on lagged returns/indicators trains in seconds on either machine, is trivial to explain to judges via feature importances, and is much less likely to blow up your 24h budget than debugging a neural net at 3am. Treat "deep learning" in the tool name as optional, not literal; a defensible trained model beats a fragile fancy one.

Have a **recorded/replayed fallback run** ready before judging — if the venue wifi or a free-tier LLM dies mid-demo, you switch to replay mode without missing a beat.

---

## 8. Stretch goals (only after MVP fully works end-to-end)

- Split the unified critic back into Risk/Profit/Macro critics (matches the reference diagram exactly).
- Add Sector Intelligence as its own agent instead of folded into Opportunity Critic.
- Add a Planner Agent that dynamically selects which specialist agents to consult per cycle, instead of the MVP's fixed-roster asyncio loop (matches the reference diagram's Investment Planner Agent box — see README's "Investment Planner Agent" layer).
- Persist historical reliability/trust across sessions (SQLite already has the data — the MVP formula just resets the prior each session; stretch is to warm-start `historical_reliability_i` from prior sessions' `agent_predictions` instead).
- Real-time social sentiment (StockTwits/Twitter) alongside news.
- Calibration plot of confidence vs. actual hit-rate per agent (nice slide, shows you actually measured trust rather than asserting it).
- Multi-day backtest mode to show the consensus weights adapting over time.

---

## 9. Immediate next step

Say the word and I'll scaffold the actual repo (`backend/agents/`, `core/consensus_engine.py`, `core/llm_router.py`, `db/schema.sql`, `api/main.py`, `frontend/`) with working skeletons so hour 0–2 is a git-clone instead of a blank folder.
