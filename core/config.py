"""Shared constants — frozen alongside core/schemas.py so neither side hardcodes
its own copy of a number the other side also depends on. Tune the *_THRESHOLD
and rate constants during the hour-18 dry run; don't hand-edit them in two
places.
"""

import os

from dotenv import load_dotenv

load_dotenv()

CAPITAL = 10_000.0
LEVERAGE = 2
BUYING_POWER = CAPITAL * LEVERAGE

# Fixed — the Forecasting agent one-hot-encodes symbols against this exact
# list at training time (backend/agents/forecasting.py). Never shrink this
# to "trim the watchlist"; it silently breaks the trained model's feature
# alignment. Use ACTIVE_WATCHLIST below instead.
WATCHLIST = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "INFY",
    "ICICIBANK",
    "SBIN",
    "AXISBANK",  # was TATAMOTORS — delisted after the 2025 Tata Motors PV/CV demerger (now TMPV / TMCV)
    "ITC",
    "LT",
    "ADANIENT",
]

# The subset actually evaluated each cycle by the orchestrator/API — trim
# via env for rate-limit/demo reasons without touching WATCHLIST above.
# e.g. ACTIVE_WATCHLIST=INFY,TCS,RELIANCE in .env
ACTIVE_WATCHLIST = [s.strip() for s in os.getenv("ACTIVE_WATCHLIST", ",".join(WATCHLIST)).split(",") if s.strip()]

# Consensus decision thresholds — §3
THETA_HOLD = 0.15
THETA_BUY = 0.35
THETA_SELL = 0.35
THETA_VAR = 0.05  # weighted disagreement above this, inside the hold band -> WAIT instead of HOLD
LAMBDA_HERDING = 0.5  # weight of herding_penalty inside trust_i
GAMMA_AGREEMENT = 0.4  # exponent on agreement_live_i in w_i_raw
HERDING_PENALTY_MIN_AGREEMENT_RATE = 0.8  # only penalize agreement rates above this
SWITCH_SAFETY_MARGIN = 0.002  # extra edge required over round-trip cost before SWITCH fires

# Risk Management Layer — core/portfolio.py
MAX_POSITION_FRACTION = 0.4  # no single stock may exceed this fraction of buying power
MAX_RISK_SCORE = 0.75  # expected_risk_return.risk_score above this halves the allocation

# NSE intraday equity retail cost model — §4 (fractions, not percentages)
BROKERAGE_FLAT_CAP = 20.0
BROKERAGE_PCT = 0.0003
STT_PCT = 0.00025
EXCHANGE_TXN_PCT = 0.0000297
SEBI_CHARGE_PER_CRORE = 10.0
STAMP_DUTY_PCT = 0.00003
GST_PCT = 0.18
SLIPPAGE_PCT_RANGE = (0.0002, 0.0005)

# Session timing (IST)
SESSION_START = "09:15"
SESSION_SQUARE_OFF = "15:15"  # forced close, ahead of actual 15:30 market close

# Forecasting agent (backend/agents/forecasting.py)
FORECAST_HORIZON_BARS = 3  # 3 x 5-min bars = next 15 minutes
FORECAST_EPSILON = 0.0005  # |predicted return| below this -> NEUTRAL
FORECAST_CONFIDENCE_SCALE = 0.005  # |predicted return| at/above this -> confidence 1.0

# API / orchestrator runtime (api/main.py)
CYCLE_INTERVAL_SECONDS = 300  # one orchestrator cycle every 5 minutes
DB_PATH = "committee.db"

# Replay mode (backend/data/replay.py) — set REPLAY_MODE=true in .env when
# NSE is closed, so the pipeline runs against cached historical bars on an
# accelerated virtual clock instead of live (empty) market data.
REPLAY_MODE = os.getenv("REPLAY_MODE", "false").lower() == "true"
REPLAY_SPEED = float(os.getenv("REPLAY_SPEED", "60"))  # 1 real second = this many simulated seconds
REPLAY_CYCLE_INTERVAL_SECONDS = 10  # real seconds between cycles while replaying
