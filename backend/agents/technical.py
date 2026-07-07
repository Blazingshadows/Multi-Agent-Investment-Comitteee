"""Technical agent — fully deterministic, no LLM call needed for the math
(per PROJECT.md §2). Combines three normalized sub-signals (RSI momentum,
MACD histogram scaled by ATR, price distance from a 20-period SMA) into one
direction + confidence. Reasoning text is templated from the actual computed
values, not LLM-generated prose.
"""

import numpy as np
import pandas as pd

from core.indicators import atr, macd, rsi, sma
from core.schemas import AgentOutput, Direction

SMA_PERIOD = 20
NEUTRAL_EPS = 0.05  # |raw_score| below this -> NEUTRAL


def analyze(symbol: str, context: dict) -> AgentOutput:
    """context['ohlcv'] must be a DataFrame with open/high/low/close/volume
    columns, indexed by timestamp ascending, with enough history for RSI-14,
    MACD-26, SMA-20, and ATR-14 to have warmed up (>= ~30 bars).
    """
    df = context["ohlcv"]
    close = df["close"]

    rsi_series = rsi(close)
    macd_df = macd(close)
    sma_series = sma(close, SMA_PERIOD)
    atr_series = atr(df["high"], df["low"], close)

    rsi_val = rsi_series.iloc[-1]
    histogram = macd_df["histogram"].iloc[-1]
    sma_val = sma_series.iloc[-1]
    atr_val = atr_series.iloc[-1]
    price = close.iloc[-1]

    if pd.isna(rsi_val) or pd.isna(histogram) or pd.isna(sma_val) or pd.isna(atr_val):
        return AgentOutput(
            agent="Technical",
            direction=Direction.NEUTRAL,
            confidence=0.0,
            reasoning="Not enough history yet to compute RSI/MACD/SMA/ATR (indicators still warming up).",
            evidence=[],
        )

    rsi_score = float(np.clip((rsi_val - 50) / 50, -1, 1))
    macd_score = float(np.clip(histogram / atr_val, -1, 1)) if atr_val > 0 else 0.0
    pct_vs_sma = price / sma_val - 1
    sma_score = float(np.clip(pct_vs_sma / 0.02, -1, 1))

    raw_score = (rsi_score + macd_score + sma_score) / 3

    if raw_score > NEUTRAL_EPS:
        direction = Direction.BULLISH
    elif raw_score < -NEUTRAL_EPS:
        direction = Direction.BEARISH
    else:
        direction = Direction.NEUTRAL

    confidence = min(1.0, abs(raw_score))

    macd_bias = "bullish" if histogram > 0 else "bearish"
    reasoning = (
        f"RSI={rsi_val:.1f} (momentum {'above' if rsi_val > 50 else 'below'} midline), "
        f"MACD histogram={histogram:.2f} ({macd_bias} momentum, {histogram / atr_val:.2f}x ATR), "
        f"price {pct_vs_sma:+.1%} vs {SMA_PERIOD}-period SMA."
    )

    return AgentOutput(
        agent="Technical",
        direction=direction,
        confidence=round(confidence, 3),
        reasoning=reasoning,
        evidence=[
            f"rsi_14={rsi_val:.1f}",
            f"macd_histogram={histogram:.3f}",
            f"close_vs_sma20={pct_vs_sma:+.3%}",
            f"atr_14={atr_val:.3f}",
        ],
    )
