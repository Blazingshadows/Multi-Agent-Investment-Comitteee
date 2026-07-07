from datetime import date

from backend.data.macro_calendar import get_macro_context


def test_rbi_policy_day_flag_near_mpc_date():
    context = get_macro_context(date(2026, 2, 5))
    assert context["rbi_policy_day"] is True
    assert context["days_to_next_rbi_mpc"] == 1


def test_rbi_policy_day_flag_far_from_mpc_date():
    context = get_macro_context(date(2026, 7, 8))
    assert context["rbi_policy_day"] is False


def test_earnings_day_flag_within_window():
    context = get_macro_context(date(2026, 7, 15))
    assert context["earnings_day"] is True


def test_earnings_day_flag_outside_window():
    context = get_macro_context(date(2026, 3, 1))
    assert context["earnings_day"] is False
