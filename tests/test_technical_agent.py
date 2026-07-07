import numpy as np
import pandas as pd

from backend.agents.technical import analyze
from core.schemas import Direction


def _trending_ohlcv(n: int = 60, drift: float = 1.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-07-08 09:15", periods=n, freq="5min")
    close = 1000 + np.cumsum(rng.normal(drift, 1.0, n))
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.3, n),
            "high": close + abs(rng.normal(1, 0.3, n)),
            "low": close - abs(rng.normal(1, 0.3, n)),
            "close": close,
            "volume": rng.integers(10_000, 100_000, n),
        },
        index=timestamps,
    )


def test_uptrend_produces_bullish_call():
    df = _trending_ohlcv(drift=3.0)
    result = analyze("INFY", {"ohlcv": df})
    assert result.direction == Direction.BULLISH
    assert 0.0 < result.confidence <= 1.0


def test_downtrend_produces_bearish_call():
    df = _trending_ohlcv(drift=-3.0)
    result = analyze("INFY", {"ohlcv": df})
    assert result.direction == Direction.BEARISH


def test_handles_too_little_history():
    df = _trending_ohlcv(n=3)
    result = analyze("INFY", {"ohlcv": df})
    assert result.direction == Direction.NEUTRAL
    assert result.confidence == 0.0
