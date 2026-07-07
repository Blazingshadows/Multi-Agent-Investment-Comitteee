"""Run once, early — hour 0-2 per PROJECT.md. Pulls ~60 days of 5-min bars
for every watchlist stock and caches to data/history/<symbol>.parquet, so
the Forecasting agent (Person A) has training data ready without depending
on yfinance's rate limits later in the day.

Usage: python -m scripts.backfill_history
"""

from pathlib import Path

from backend.data.market_data import backfill_history
from core.config import WATCHLIST

OUT_DIR = Path(__file__).parent.parent / "data" / "history"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in WATCHLIST:
        df = backfill_history(symbol)
        out_path = OUT_DIR / f"{symbol}.parquet"
        df.to_parquet(out_path)
        print(f"{symbol}: {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
