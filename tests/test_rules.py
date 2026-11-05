"""Tests for the Tier 2 rules (karaglof.rules).

The key property is "no future data": appending later weeks to a lake's
series must never change the alert level of an earlier week.
"""

import pandas as pd

from karaglof.rules import (
    LEVEL_AMBER,
    LEVEL_GREEN,
    LEVEL_RED,
    assign_alert_levels,
    count_false_alarms,
    lead_time_days,
)


def make_weeks(areas, anomalies, start="2022-04-04", lake_id="L1"):
    weeks = pd.date_range(start, periods=len(areas), freq="7D").strftime("%Y-%m-%d")
    change = [None, None] + [
        (areas[i] / areas[i - 2] - 1.0) * 100.0 for i in range(2, len(areas))
    ]
    return pd.DataFrame({
        "lake_id": lake_id,
        "week_start": weeks,
        "area_km2_s1": areas,
        "area_change_2w_pct": change,
        "t2m_anomaly_c": anomalies,
    })


def alerts_from(levels, start="2022-04-04"):
    weeks = pd.date_range(start, periods=len(levels), freq="7D").strftime("%Y-%m-%d")
    return pd.DataFrame({"week_start": weeks, "level": levels})


def test_future_weeks_do_not_change_past_levels():
    # Week 3 is the record area so far with a +2.5 C anomaly -> amber.
    past = make_weeks([0.20, 0.21, 0.22], [0.5, 1.0, 2.5])
    # Same three weeks plus much larger later weeks: a whole-series
    # percentile would push week 3 below p90 and silently turn it green.
    full = make_weeks([0.20, 0.21, 0.22, 0.40, 0.50, 0.60],
                      [0.5, 1.0, 2.5, 0.5, 0.5, 0.5])
    before = assign_alert_levels(past)["level"].tolist()
    after = assign_alert_levels(full)["level"].tolist()[: len(before)]
    assert before[-1] == LEVEL_AMBER
    assert before == after


def test_levels_follow_the_rules():
    # Week 3: area at its record so far with a +2.5 C anomaly -> amber.
    # Week 4: record again with +3.5 C -> red (p95 and >+3 C).
    levels = assign_alert_levels(
        make_weeks([0.20, 0.20, 0.21, 0.22], [0.0, 0.0, 2.5, 3.5])
    )["level"].tolist()
    assert levels == [LEVEL_GREEN, LEVEL_GREEN, LEVEL_AMBER, LEVEL_RED]


def test_fast_growth_alone_is_amber():
    # +20% in 2 weeks with no heat; area below its earlier peak.
    levels = assign_alert_levels(
        make_weeks([0.50, 0.20, 0.20, 0.24], [0.0, 0.0, 0.0, 0.0])
    )["level"].tolist()
    assert levels[-1] == LEVEL_AMBER


def test_lead_time_uses_only_the_run_leading_into_the_event():
    # An unrelated amber in week 1, then green, then a run from week 4.
    alerts = alerts_from(["amber", "green", "green", "amber", "red"])
    # Weeks start 4 Apr ... 2 May; drainage 7 May; run starts 25 Apr.
    assert lead_time_days(alerts, "2022-05-07") == 12


def test_no_lead_time_when_warning_ended_long_before():
    alerts = alerts_from(["amber", "green", "green", "green", "green"])
    assert lead_time_days(alerts, "2022-05-07") is None


def test_false_alarms_exclude_the_leading_run():
    alerts = alerts_from(["amber", "green", "green", "amber", "red"])
    assert count_false_alarms(alerts, ["2022-05-07"]) == 1
    assert count_false_alarms(alerts, []) == 3
