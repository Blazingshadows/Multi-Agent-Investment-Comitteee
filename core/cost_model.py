"""Person B owns this file. Implemented for real, not stubbed — the §4 NSE
intraday cost model is pure deterministic arithmetic with no dependency on
live data, so there's nothing to fake.
"""

from core.config import (
    BROKERAGE_FLAT_CAP,
    BROKERAGE_PCT,
    EXCHANGE_TXN_PCT,
    GST_PCT,
    SEBI_CHARGE_PER_CRORE,
    SLIPPAGE_PCT_RANGE,
    STAMP_DUTY_PCT,
    STT_PCT,
)
from core.schemas import Action, CostBreakdown

CRORE = 10_000_000


def apply_costs(action: Action, qty: float, price: float) -> tuple[float, CostBreakdown]:
    """Only meaningful for BUY/SELL — a SWITCH decomposes into a SELL of the
    old position and a BUY of the new one, called separately.
    """
    turnover = qty * price

    brokerage = min(BROKERAGE_FLAT_CAP, BROKERAGE_PCT * turnover)
    stt = STT_PCT * turnover if action == Action.SELL else 0.0
    exchange_txn_charges = EXCHANGE_TXN_PCT * turnover
    sebi_charges = SEBI_CHARGE_PER_CRORE * (turnover / CRORE)
    stamp_duty = STAMP_DUTY_PCT * turnover if action == Action.BUY else 0.0
    gst = GST_PCT * (brokerage + exchange_txn_charges)
    slippage = (sum(SLIPPAGE_PCT_RANGE) / 2) * turnover  # midpoint; randomize within range later if desired

    breakdown = CostBreakdown(
        brokerage=brokerage,
        stt=stt,
        exchange_txn_charges=exchange_txn_charges,
        sebi_charges=sebi_charges,
        stamp_duty=stamp_duty,
        gst=gst,
        slippage=slippage,
    )

    net_cash_flow = (-turnover if action == Action.BUY else turnover) - breakdown.total_cost
    return net_cash_flow, breakdown
