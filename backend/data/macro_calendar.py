"""Static macro/policy calendar — per PROJECT.md §5 ("static macro table:
RBI repo rate, budget, sector calendar"), deliberately not a live API.
Update the dates periodically; they don't need to be exact to the day to be
useful context. Output flags (rbi_policy_day, earnings_day) map directly
onto core.consensus_engine.CONTEXT_BOOSTS keys.
"""

from datetime import date

RBI_MPC_DATES = [
    date(2026, 2, 6), date(2026, 4, 8), date(2026, 6, 10),
    date(2026, 8, 12), date(2026, 10, 8), date(2026, 12, 10),
]

UNION_BUDGET_DATE = date(2026, 2, 1)

# Most NSE large-caps report within these windows each quarter.
EARNINGS_SEASON_WINDOWS = [
    (date(2026, 1, 10), date(2026, 2, 5)),
    (date(2026, 4, 10), date(2026, 5, 5)),
    (date(2026, 7, 10), date(2026, 8, 5)),
    (date(2026, 10, 10), date(2026, 11, 5)),
]


def get_macro_context(as_of: date | None = None) -> dict:
    as_of = as_of or date.today()

    upcoming_mpc = [d for d in RBI_MPC_DATES if d >= as_of]
    days_to_next_mpc = (min(upcoming_mpc) - as_of).days if upcoming_mpc else None

    return {
        "date": as_of.isoformat(),
        "rbi_policy_day": days_to_next_mpc is not None and days_to_next_mpc <= 2,
        "days_to_next_rbi_mpc": days_to_next_mpc,
        "earnings_day": any(start <= as_of <= end for start, end in EARNINGS_SEASON_WINDOWS),
        "days_to_budget": (UNION_BUDGET_DATE - as_of).days if as_of <= UNION_BUDGET_DATE else None,
    }
