"""FastAPI REST + WebSocket layer — serves live cycle results, trade log,
and portfolio curve to the dashboard. Runs the orchestrator as a background
task every CYCLE_INTERVAL_SECONDS.

Locking model: `_cycle_lock` only guards against two cycles running at once
(the background loop racing a manual /api/cycle/run trigger) — it is never
held across a read. Read endpoints open their own short-lived connection
each request (db/schema.sql is WAL-mode, so reads never block on the writer
mid-cycle); a lock held for a whole multi-minute orchestrator cycle would
otherwise freeze the entire dashboard for that cycle's duration.
"""

import asyncio
import threading
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.orchestrator import run_cycle, run_square_off
from core.config import CAPITAL, CYCLE_INTERVAL_SECONDS, DB_PATH, SESSION_SQUARE_OFF, WATCHLIST
from core.portfolio import Portfolio
from db.persistence import get_decision_log, get_portfolio_curve, get_trade_history, init_db

_cycle_lock = threading.Lock()
_state: dict = {"conn": None, "portfolio": None, "last_cycle_ts": None, "squared_off": False}
_websocket_clients: set[WebSocket] = set()


def _read_conn():
    """Fresh connection per read — cheap for SQLite, avoids any thread/lock
    contention with the writer connection the cycle loop holds.
    """
    return init_db(DB_PATH)


async def _broadcast(message: dict) -> None:
    dead = []
    for ws in _websocket_clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _websocket_clients.discard(ws)


async def _cycle_loop() -> None:
    while True:
        now = datetime.now()
        square_off_time = now.strftime("%H:%M") >= SESSION_SQUARE_OFF

        if square_off_time and not _state["squared_off"]:
            with _cycle_lock:
                await asyncio.to_thread(run_square_off, _state["conn"], _state["portfolio"], now)
                _state["squared_off"] = True
            await _broadcast({"type": "square_off", "cycle_ts": now.isoformat()})
        elif not square_off_time:
            with _cycle_lock:
                rows = await asyncio.to_thread(run_cycle, _state["conn"], _state["portfolio"], now)
                _state["last_cycle_ts"] = now.isoformat()
            await _broadcast({"type": "cycle_complete", "cycle_ts": now.isoformat(), "decisions": len(rows)})

        await asyncio.sleep(CYCLE_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["conn"] = init_db(DB_PATH)
    _state["portfolio"] = Portfolio()
    task = asyncio.create_task(_cycle_loop())
    yield
    task.cancel()


app = FastAPI(title="Autonomous Multi-Agent Investment Committee", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/status")
def status():
    return {
        "last_cycle_ts": _state["last_cycle_ts"],
        "squared_off": _state["squared_off"],
        "watchlist": WATCHLIST,
        "cycle_interval_seconds": CYCLE_INTERVAL_SECONDS,
    }


@app.get("/api/decisions")
def decisions(limit: int = 50):
    return get_decision_log(_read_conn(), limit=limit)


@app.get("/api/trades")
def trades():
    return get_trade_history(_read_conn())


@app.get("/api/portfolio")
def portfolio():
    curve = get_portfolio_curve(_read_conn())
    p = _state["portfolio"]
    return {
        "cash": p.cash,
        "positions": p.positions,
        "latest_value": curve[-1]["portfolio_value"] if curve else p.cash,
        "curve": curve,
    }


@app.get("/api/summary")
def summary():
    """Final Output fields per the PS: portfolio value, net profit, growth%,
    trade history, complete decision log."""
    conn = _read_conn()
    curve = get_portfolio_curve(conn)
    latest_value = curve[-1]["portfolio_value"] if curve else CAPITAL
    net_profit = latest_value - CAPITAL
    return {
        "final_portfolio_value": latest_value,
        "net_profit": net_profit,
        "portfolio_growth_pct": (net_profit / CAPITAL) * 100,
        "trade_history": get_trade_history(conn),
        "decision_log": get_decision_log(conn),
    }


@app.post("/api/cycle/run")
async def trigger_cycle():
    """Manual trigger — run one cycle immediately without waiting for the
    interval. Useful for demo control and testing. Shares _cycle_lock with
    the background loop so the two can never run concurrently."""
    now = datetime.now()
    with _cycle_lock:
        rows = await asyncio.to_thread(run_cycle, _state["conn"], _state["portfolio"], now)
    await _broadcast({"type": "cycle_complete", "cycle_ts": now.isoformat(), "decisions": len(rows)})
    return {"decisions": len(rows)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _websocket_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _websocket_clients.discard(websocket)
