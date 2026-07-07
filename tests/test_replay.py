"""Hermetic — uses synthetic parquet data in a tmp HISTORY_DIR, not the real
gitignored data/history/ cache, and monkeypatches time.time() so clock
advancement is deterministic rather than depending on real elapsed time.
"""

import numpy as np
import pandas as pd
import pytest

from backend.data import replay


@pytest.fixture(autouse=True)
def _reset_replay_state(tmp_path, monkeypatch):
    """Every test gets a fresh clock and a fresh, isolated HISTORY_DIR."""
    monkeypatch.setattr(replay, "HISTORY_DIR", tmp_path)
    replay._full_history_cache.clear()
    replay._start_wall_time = None
    replay._start_sim_time = None
    yield
    replay._full_history_cache.clear()
    replay._start_wall_time = None
    replay._start_sim_time = None


def _write_synthetic_history(path, n=300, seed=0):
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-06-01 09:15", periods=n, freq="5min")
    close = 1000 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": rng.integers(10_000, 100_000, n)},
        index=timestamps,
    )
    df.index.name = "timestamp"
    df.to_parquet(path)
    return df


def test_replay_start_date_picks_that_days_first_bar(tmp_path, monkeypatch):
    _write_synthetic_history(tmp_path / "TEST.parquet")  # spans 2026-06-01 09:15 through 2026-06-02
    monkeypatch.setattr(replay, "REPLAY_START_DATE", "2026-06-02")
    fake_wall_time = [1_000_000.0]
    monkeypatch.setattr(replay.time, "time", lambda: fake_wall_time[0])

    replay.fetch_ohlcv("TEST")
    assert replay._start_sim_time.date().isoformat() == "2026-06-02"


def test_replay_start_date_falls_back_when_not_in_range(tmp_path, monkeypatch):
    df = _write_synthetic_history(tmp_path / "TEST.parquet")
    monkeypatch.setattr(replay, "REPLAY_START_DATE", "2099-01-01")  # not in this file's range
    fake_wall_time = [1_000_000.0]
    monkeypatch.setattr(replay.time, "time", lambda: fake_wall_time[0])

    replay.fetch_ohlcv("TEST")
    assert replay._start_sim_time == df.index[replay.WARMUP_BARS]


def test_fetch_ohlcv_returns_bars_up_to_sim_time(tmp_path, monkeypatch):
    _write_synthetic_history(tmp_path / "TEST.parquet")

    fake_wall_time = [1_000_000.0]
    monkeypatch.setattr(replay.time, "time", lambda: fake_wall_time[0])

    df = replay.fetch_ohlcv("TEST")
    assert len(df) > 0
    assert df.index[-1] <= replay._current_sim_timestamp()


def test_clock_advances_with_wall_time(tmp_path, monkeypatch):
    _write_synthetic_history(tmp_path / "TEST.parquet")

    fake_wall_time = [1_000_000.0]
    monkeypatch.setattr(replay.time, "time", lambda: fake_wall_time[0])

    first = replay.fetch_ohlcv("TEST")
    fake_wall_time[0] += 60  # 60 real seconds later
    second = replay.fetch_ohlcv("TEST")

    assert second.index[-1] > first.index[-1]


def test_is_session_over_false_at_start(tmp_path, monkeypatch):
    _write_synthetic_history(tmp_path / "TEST.parquet")
    fake_wall_time = [1_000_000.0]
    monkeypatch.setattr(replay.time, "time", lambda: fake_wall_time[0])

    replay.fetch_ohlcv("TEST")
    assert replay.is_session_over() is False


def test_is_session_over_true_after_a_full_session_elapses(tmp_path, monkeypatch):
    _write_synthetic_history(tmp_path / "TEST.parquet")
    fake_wall_time = [1_000_000.0]
    monkeypatch.setattr(replay.time, "time", lambda: fake_wall_time[0])

    replay.fetch_ohlcv("TEST")
    # SESSION_LENGTH is 6 simulated hours; REPLAY_SPEED seconds of sim time
    # pass per real second, so this many real seconds covers it with margin.
    seconds_needed = replay.SESSION_LENGTH.total_seconds() / replay.REPLAY_SPEED
    fake_wall_time[0] += seconds_needed + 10
    assert replay.is_session_over() is True


def test_current_sim_time_bootstraps_from_any_cached_symbol(tmp_path, monkeypatch):
    _write_synthetic_history(tmp_path / "SOMESTOCK.parquet")
    fake_wall_time = [1_000_000.0]
    monkeypatch.setattr(replay.time, "time", lambda: fake_wall_time[0])

    now = replay.current_sim_time()
    assert now is not None


def test_missing_history_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="backfill_history"):
        replay.fetch_ohlcv("NONEXISTENT")
