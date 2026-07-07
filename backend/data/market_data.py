"""yfinance OHLCV + fundamentals ingestion for the watchlist:
- fetch_ohlcv() / fetch_watchlist_snapshot() — live/delayed pull for the
  current cycle, fed into the orchestrator loop each run.
- backfill_history() — one-off pull of several weeks of 5-min bars per
  stock, for the Forecasting agent's training set. Run this once, early
  (see scripts/backfill_history.py), not on every cycle.
- fetch_fundamentals() — a curated subset of yfinance's Ticker.info, for
  the Fundamental agent.
"""

import pandas as pd
import yfinance as yf

from core.config import WATCHLIST


def nse_ticker(symbol: str) -> str:
    return f"{symbol}.NS"


def fetch_ohlcv(symbol: str, period: str = "1d", interval: str = "5m") -> pd.DataFrame:
    """yfinance limits intraday granularity by lookback window: 1m data only
    goes back 7 days, 5m/15m go back 60 days. Columns are flattened and
    lowercased (open/high/low/close/volume) — yfinance returns a
    (field, ticker) MultiIndex for a single-ticker download, which every
    downstream agent would otherwise have to unwrap itself.
    """
    df = yf.download(nse_ticker(symbol), period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df.index.name = "timestamp"
    return df


def fetch_watchlist_snapshot(interval: str = "5m") -> dict[str, pd.DataFrame]:
    """Live pull across the whole watchlist for one orchestrator cycle."""
    return {symbol: fetch_ohlcv(symbol, period="1d", interval=interval) for symbol in WATCHLIST}


def backfill_history(symbol: str, period: str = "60d", interval: str = "5m") -> pd.DataFrame:
    return fetch_ohlcv(symbol, period=period, interval=interval)


FUNDAMENTAL_FIELDS = [
    "trailingPE", "forwardPE", "marketCap", "earningsGrowth", "revenueGrowth",
    "profitMargins", "returnOnEquity", "debtToEquity", "trailingEps", "forwardEps",
    "priceToBook", "dividendYield", "sector", "longName",
]


def fetch_fundamentals(symbol: str) -> dict:
    """Missing fields are omitted rather than sent as None — keeps the LLM
    prompt from being cluttered with "N/A" noise for whatever this ticker
    doesn't report.
    """
    info = yf.Ticker(nse_ticker(symbol)).info
    return {field: info[field] for field in FUNDAMENTAL_FIELDS if info.get(field) is not None}
