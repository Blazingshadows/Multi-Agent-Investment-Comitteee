"""Hermetic test — synthetic OHLCV, no network call and no dependency on the
gitignored data/history/ cache, so this runs the same on any machine as long
as models/forecasting_lgbm.txt has been trained once.
"""

import numpy as np
import pandas as pd
import pytest

from backend.agents.forecasting import MODEL_PATH, analyze
from core.schemas import Direction

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="run `python -m scripts.train_forecasting_model` first"
)


def _synthetic_ohlcv(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-07-08 09:15", periods=n, freq="5min")
    close = 1000 + np.cumsum(rng.normal(0, 2, n))
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.5, n),
            "high": close + abs(rng.normal(1, 0.5, n)),
            "low": close - abs(rng.normal(1, 0.5, n)),
            "close": close,
            "volume": rng.integers(10_000, 100_000, n),
        },
        index=timestamps,
    )


def test_analyze_returns_valid_agent_output():
    df = _synthetic_ohlcv()
    result = analyze("INFY", {"ohlcv": df})

    assert result.agent == "Forecasting"
    assert result.direction in (Direction.BEARISH, Direction.NEUTRAL, Direction.BULLISH)
    assert 0.0 <= result.confidence <= 1.0
    assert result.reasoning


def test_analyze_handles_too_little_history():
    df = _synthetic_ohlcv(n=3)  # not enough bars for rolling/lagged features
    result = analyze("INFY", {"ohlcv": df})

    assert result.direction == Direction.NEUTRAL
    assert result.confidence == 0.0
