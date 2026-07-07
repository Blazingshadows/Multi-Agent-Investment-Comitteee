"""Trains the Forecasting agent's model. Run once after backfilling history
(scripts/backfill_history.py), and re-run any time you want to refresh it
closer to the demo.

Pools all watchlist stocks into a single model (symbol as a one-hot feature)
since any one stock's ~2 months of 5-min bars is too little data alone.
Splits chronologically *within each symbol* before pooling, so the test set
never leaks future information into training.

Usage: python -m scripts.train_forecasting_model
"""

from pathlib import Path

import lightgbm as lgb
import pandas as pd

from backend.agents.forecasting import FEATURE_COLUMNS, MODEL_PATH, build_features, build_target
from core.config import WATCHLIST

HISTORY_DIR = Path(__file__).parent.parent / "data" / "history"


def load_pooled_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frames, test_frames = [], []
    for symbol in WATCHLIST:
        path = HISTORY_DIR / f"{symbol}.parquet"
        if not path.exists():
            print(f"skip {symbol}: no cached history, run scripts/backfill_history.py")
            continue
        df = pd.read_parquet(path)
        features = build_features(df, symbol)
        features["target"] = build_target(df)
        features = features.dropna()

        split_idx = int(len(features) * 0.8)
        train_frames.append(features.iloc[:split_idx])
        test_frames.append(features.iloc[split_idx:])

    return pd.concat(train_frames), pd.concat(test_frames)


def main() -> None:
    train_df, test_df = load_pooled_dataset()
    print(f"train rows: {len(train_df)}, test rows: {len(test_df)}")

    train_set = lgb.Dataset(train_df[FEATURE_COLUMNS], label=train_df["target"])
    test_set = lgb.Dataset(test_df[FEATURE_COLUMNS], label=test_df["target"], reference=train_set)

    params = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "num_leaves": 15,
        "learning_rate": 0.05,
        "min_data_in_leaf": 50,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=200,
        valid_sets=[test_set],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(20)],
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"saved -> {MODEL_PATH}")

    preds = model.predict(test_df[FEATURE_COLUMNS])
    mae = (preds - test_df["target"]).abs().mean()
    directional_accuracy = ((preds > 0) == (test_df["target"] > 0)).mean()
    print(f"test MAE: {mae:.5f}")
    print(f"directional accuracy: {directional_accuracy:.3f}")

    importances = dict(zip(FEATURE_COLUMNS, model.feature_importance().tolist()))
    top5 = sorted(importances.items(), key=lambda kv: -kv[1])[:5]
    print(f"top 5 features by importance: {top5}")


if __name__ == "__main__":
    main()
