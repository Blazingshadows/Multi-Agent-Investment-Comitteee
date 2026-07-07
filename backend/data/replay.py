"""Replay mode — reads cached historical bars (data/history/*.parquet, from
scripts/backfill_history.py) and serves them as if they were live, advancing
a shared virtual clock at REPLAY_SPEED x real time. fetch_ohlcv() has the
same signature/shape as backend.data.market_data.fetch_ohlcv, so the
orchestrator can swap between live and replay without any calling code
changing (see backend/orchestrator.py's _fetch_ohlcv dispatch).
"""

import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.config import REPLAY_SPEED, REPLAY_START_DATE

HISTORY_DIR = Path(__file__).parent.parent.parent / "data" / "history"
SESSION_LENGTH = pd.Timedelta(hours=6)  # one simulated "4-6 hour" trading session
WARMUP_BARS = 100  # skip the first N bars so RSI/MACD/SMA are already warmed up at replay start

_full_history_cache: dict[str, pd.DataFrame] = {}
_start_wall_time: float | None = None
_start_sim_time: pd.Timestamp | None = None


def _load_full_history(symbol: str) -> pd.DataFrame:
    if symbol not in _full_history_cache:
        path = HISTORY_DIR / f"{symbol}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No cached history for {symbol} at {path} — run `python -m scripts.backfill_history` first.")
        _full_history_cache[symbol] = pd.read_parquet(path)
    return _full_history_cache[symbol]


def _default_start(df: pd.DataFrame) -> pd.Timestamp:
    if REPLAY_START_DATE:
        day_bars = df[df.index.date.astype(str) == REPLAY_START_DATE]
        if len(day_bars) > 0:
            return day_bars.index[0]
        # requested date not in this symbol's cached range — fall through
    return df.index[min(WARMUP_BARS, len(df) - 1)]


def _ensure_clock_started(symbol: str) -> None:
    global _start_wall_time, _start_sim_time
    if _start_sim_time is not None:
        return
    df = _load_full_history(symbol)
    _start_sim_time = _default_start(df)
    _start_wall_time = time.time()


def _current_sim_timestamp() -> pd.Timestamp:
    elapsed_wall_seconds = time.time() - _start_wall_time
    return _start_sim_time + pd.Timedelta(seconds=elapsed_wall_seconds * REPLAY_SPEED)


def current_sim_time() -> datetime:
    """Wall-clock-independent "now" for the replay session — api/main.py
    uses this instead of datetime.now() when REPLAY_MODE is on.
    """
    _ensure_clock_started_default()
    return _current_sim_timestamp().floor("us").to_pydatetime()  # datetime has no ns precision


def is_session_over() -> bool:
    """True once the virtual clock has advanced a full simulated session
    past the replay start point — api/main.py triggers force_square_off()
    on this instead of a real-clock SESSION_SQUARE_OFF comparison.
    """
    if _start_sim_time is None:
        return False
    return _current_sim_timestamp() >= _start_sim_time + SESSION_LENGTH


def _ensure_clock_started_default() -> None:
    """current_sim_time()/is_session_over() may be called before any
    fetch_ohlcv() — bootstrap the clock off whichever symbol has cached
    history first.
    """
    if _start_sim_time is not None:
        return
    for path in sorted(HISTORY_DIR.glob("*.parquet")):
        _ensure_clock_started(path.stem)
        return
    raise FileNotFoundError(f"No cached history in {HISTORY_DIR} — run `python -m scripts.backfill_history` first.")


def fetch_ohlcv(symbol: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    _ensure_clock_started(symbol)
    df = _load_full_history(symbol)
    now = _current_sim_timestamp()
    window = df[df.index <= now]
    return window.tail(400)  # bounded trailing window, similar to a real 5d/5m live pull
