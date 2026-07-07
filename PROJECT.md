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

**Decision rule** (order matters — BUY/SELL checked first, WAIT is the catch-all for anything in between the hold-band and the buy/sell thresholds, which the original 4-branch sketch left undefined):

```
if   DCS >= θ_buy                                → BUY
elif DCS <= -θ_sell                              → SELL
elif |DCS| < θ_hold  and disagreement < θ_var    → HOLD   (agents genuinely agree it's neutral)
else                                             → WAIT   (weak conviction, or high disagreement, or the mid-zone between θ_hold and θ_buy/sell)
```

**SWITCH**: if the Opportunity Critic proposes an alternative stock whose DCS beats the current position's DCS by more than the round-trip trading cost (see §4) + a safety margin, switch. Otherwise stay put — this stops the bot from churning on noise. Implemented as `consensus_engine.evaluate_switch()`, separate from `run_consensus()` since it requires comparing two stocks' DCS.

**Allocation**: DCS is mathematically bounded to `[-1, 1]` (a weighted average of votes each in `[-1, 1]`), so allocating `min(2.0, |DCS|)` of capital can never reach the 1:2 leverage cap. Instead scale by how far past the BUY/SELL threshold the conviction sits: `allocation = min(2.0, |DCS| / θ_buy_or_sell)` — 1.0x capital right at the threshold, up to the full 2.0x leverage cap as conviction approaches `|DCS| = 1`.

Suggested starting thresholds: `θ_hold = 0.15`, `θ_buy = θ_sell = 0.35`, `θ_var = 0.05`, tune once you see real numbers from a dry run. Log every raw factor (`expertise`, `trust`, `relevance`, `agreement_live`, `w_i`, `signed_vote_i`) per agent per trade — this *is* your "Consensus Reasoning" + "Directional Confidence Score" output field, and it's what makes the system explainable rather than a black box.

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

## 7. Build status — MVP complete end-to-end

Everything below is built, tested (72 tests passing — `pytest tests/`), and verified against real data and real LLM calls, not just the fixture. All on `main`.

| Layer | Files | Status |
|---|---|---|
| Contract | `core/schemas.py`, `core/config.py`, `db/schema.sql` | Frozen, unchanged since hour 0 |
| Data | `backend/data/market_data.py` (+ `fetch_fundamentals`), `news_feed.py`, `macro_calendar.py` | Live-verified against yfinance + Google News RSS |
| Deterministic agents | `backend/agents/technical.py`, `forecasting.py` (+ `scripts/train_forecasting_model.py`) | Trained, 53.7% directional accuracy — honest edge over the 50% baseline |
| LLM agents | `fundamental.py`, `sentiment.py`, `macro_policy.py`, `risk.py`, `core/llm_router.py` | Live-verified against real Groq calls |
| Critics | `backend/critics/devils_advocate.py`, `opportunity.py` | Live-verified — prompt rewritten once after catching one-word non-answers in testing |
| Trust | `core/trust_store.py` | Numerically reproduces the PS's own "Agent A always agrees / Agent B profitably disagrees" example |
| Consensus | `core/consensus_engine.py` | The graded core — real §3 formula including `agreement_live_i`, `evaluate_switch()` |
| Risk + Execution | `core/portfolio.py` | `review_trade()` (approve/reduce/reject), `execute()`, `force_square_off()` |
| Persistence | `db/persistence.py` | All 4 tables, read helpers for the API/dashboard |
| Orchestrator | `backend/orchestrator.py` | Full cycle: agents → critics → consensus → SWITCH → risk → execution → persistence |
| API | `api/main.py` | FastAPI REST + WebSocket, background cycle loop |
| Dashboard | `dashboard.py` | Streamlit, live-verified against a real running API |

### Real bugs caught by testing before they reached a live run

Worth knowing about if you're touching this code — each was caught by either a test or a live smoke test, not by inspection:
- RSI computed `NaN` instead of `100` on a zero-loss window (`avg_loss.replace(0, np.nan)` masked a legitimate value).
- The §3 decision-rule pseudocode left a threshold gap undefined; `allocation = min(2.0, |DCS|)` could never reach the leverage cap since DCS is bounded to `[-1,1]`.
- SELL was sized as a fresh allocation-based short instead of closing the actual held quantity.
- `record_prediction()` ran before `run_consensus()`, so a symbol's own current-cycle votes polluted its own "historical" trust computation.
- The orchestrator's critic pass iterated the raw watchlist instead of symbols that survived data-fetch failures → `KeyError` the moment any one symbol's fetch failed.
- FastAPI's sync endpoints run in a worker threadpool; a shared SQLite connection without `check_same_thread=False` crashed on any read.
- A lock held for an entire orchestrator cycle (minutes, across a full watchlist) froze every dashboard read for that whole duration — fixed with WAL mode + per-request read connections.
- `Portfolio.cash` defaulted to `BUYING_POWER` (₹20,000) instead of `CAPITAL` (₹10,000) — a fresh portfolio reported "100% growth" before a single trade.

### What's not built — genuine gaps, not exaggerated completeness

- **Replay mode.** NSE hours are 9:15–15:30 IST; if your demo slot falls outside that window, `backend/data/market_data.py` has no accelerated-replay alternate data source yet. Build this before judging if your slot is off-hours — it was deprioritized in favor of finishing the graded consensus core first.
- **Cross-session trust persistence.** `historical_reliability_i` resets to the 0.5 prior each session (SQLite file per run); warm-starting from a prior session's `agent_predictions` is unbuilt.
- **In-memory portfolio doesn't survive an API restart.** `Portfolio` lives in `api/main.py`'s process memory; a restart loses position state (decision/trade history in SQLite is unaffected).
- A full 10-stock cycle takes multiple minutes on Groq's free tier and can hit rate limits under repeated testing (validated live — the fallback-to-neutral path handled it cleanly, but budget for it before a live demo, or trim the watchlist).

---

## 8. Stretch goals (only after the above gaps are addressed)

- Build replay mode (see above — this is higher priority than anything below it if your demo is off-hours).
- Split the unified critic back into Risk/Profit/Macro critics (matches the reference diagram exactly).
- Add Sector Intelligence as its own agent instead of folded into Opportunity Critic.
- Add a Planner Agent that dynamically selects which specialist agents to consult per cycle, instead of always running the full roster (matches the reference diagram's Investment Planner Agent box).
- Persist historical reliability/trust across sessions.
- Real-time social sentiment (StockTwits/Twitter) alongside news.
- Calibration plot of confidence vs. actual hit-rate per agent.
- Multi-day backtest mode to show the consensus weights adapting over time.
- Parallelize across symbols in the orchestrator (currently only within-symbol agent calls are concurrent) to cut cycle time.

---

## 9. Immediate next step

Say the word and I'll scaffold the actual repo (`backend/agents/`, `core/consensus_engine.py`, `core/llm_router.py`, `db/schema.sql`, `api/main.py`, `frontend/`) with working skeletons so hour 0–2 is a git-clone instead of a blank folder.
