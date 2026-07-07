"""Forecasting agent — a genuinely trained LightGBM model
(scripts/train_forecasting_model.py), not an LLM call. Predicts the forward
return over the next FORECAST_HORIZON_BARS 5-min bars from lagged OHLCV
features. build_features()/build_target() are imported by the training
script too, so train and inference can never compute features differently.
"""

from pathlib import Path

import lightgbm as lgb
import pandas as pd

from core.config import FORECAST_CONFIDENCE_SCALE, FORECAST_EPSILON, FORECAST_HORIZON_BARS, WATCHLIST
from core.indicators import rsi, sma
from core.schemas import AgentOutput, Direction

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "forecasting_lgbm.txt"

NUMERIC_FEATURES = [
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10",
    "volatility_10", "volume_ratio",
    "rsi_14", "close_vs_sma10", "close_vs_sma30",
    "minutes_since_open",
]
SYMBOL_FEATURES = [f"sym_{s}" for s in WATCHLIST]
FEATURE_COLUMNS = NUMERIC_FEATURES + SYMBOL_FEATURES

_model: lgb.Booster | None = None


def build_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """One-hot encodes `symbol` against the fixed WATCHLIST instead of using
    LightGBM's categorical-feature machinery — avoids any risk of train/serve
    category-code mismatch, at the cost of a few extra always-0/1 columns.
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    out["ret_1"] = close.pct_change(1)
    out["ret_2"] = close.pct_change(2)
    out["ret_3"] = close.pct_change(3)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)
    out["volatility_10"] = close.pct_change().rolling(10).std()
    out["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    out["rsi_14"] = rsi(close, 14)
    out["close_vs_sma10"] = close / sma(close, 10) - 1
    out["close_vs_sma30"] = close / sma(close, 30) - 1
    out["minutes_since_open"] = df.index.hour * 60 + df.index.minute - (9 * 60 + 15)
    for sym in WATCHLIST:
        out[f"sym_{sym}"] = 1.0 if sym == symbol else 0.0
    return out


def build_target(df: pd.DataFrame, horizon: int = FORECAST_HORIZON_BARS) -> pd.Series:
    return df["close"].shift(-horizon) / df["close"] - 1


def _load_model() -> lgb.Booster:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model at {MODEL_PATH} — run `python -m scripts.train_forecasting_model` first."
            )
        _model = lgb.Booster(model_file=str(MODEL_PATH))
    return _model


def predict_return(symbol: str, context: dict) -> float | None:
    """The raw model output, exposed separately so the orchestrator can use
    it directly (e.g. for ConsensusResult.expected_risk_return.expected_return)
    without parsing it back out of analyze()'s reasoning text. None if there
    isn't enough history yet.
    """
    model = _load_model()
    df = context["ohlcv"]
    features = build_features(df, symbol)
    latest = features.iloc[[-1]][FEATURE_COLUMNS]
    if latest.isna().any(axis=1).item():
        return None
    return float(model.predict(latest)[0])


def analyze(symbol: str, context: dict) -> AgentOutput:
    """context['ohlcv'] must be a DataFrame with open/high/low/close/volume
    columns, indexed by timestamp ascending (same shape as
    backend.data.market_data.fetch_ohlcv's output).
    """
    predicted_return = predict_return(symbol, context)

    if predicted_return is None:
        return AgentOutput(
            agent="Forecasting",
            direction=Direction.NEUTRAL,
            confidence=0.0,
            reasoning="Not enough bars in this window to compute all lagged/rolling features yet.",
            evidence=[],
        )

    features = build_features(context["ohlcv"], symbol)
    latest = features.iloc[[-1]][FEATURE_COLUMNS]

    if predicted_return > FORECAST_EPSILON:
        direction = Direction.BULLISH
    elif predicted_return < -FORECAST_EPSILON:
        direction = Direction.BEARISH
    else:
        direction = Direction.NEUTRAL

    confidence = min(1.0, abs(predicted_return) / FORECAST_CONFIDENCE_SCALE)
    horizon_minutes = FORECAST_HORIZON_BARS * 5

    return AgentOutput(
        agent="Forecasting",
        direction=direction,
        confidence=round(confidence, 3),
        reasoning=f"Model predicts {predicted_return:+.3%} over the next {horizon_minutes} minutes.",
        evidence=[
            f"predicted_return={predicted_return:+.4f}",
            f"rsi_14={latest['rsi_14'].item():.1f}",
            f"volatility_10={latest['volatility_10'].item():.4f}",
        ],
    )
