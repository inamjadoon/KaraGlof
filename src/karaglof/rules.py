"""Tier 2 weekly warning-sign rules (alerts.csv).

Transparent rules, not a trained model — each Tier 2 lake has only a handful
of events. Starting thresholds (SPECS §5) are tuned once on 2015-2021 and
then frozen; they are never adjusted after looking at the Shisper 2022 replay.

Levels:
    green — none of the amber conditions hold;
    amber — area grew >15% in 2 weeks, or area above the lake's 90th
        percentile and 7-day temperature anomaly above +2 deg C;
    red   — amber conditions hold two weeks in a row, or area above the
        95th percentile with a heat anomaly above +3 deg C.

`assign_alert_levels` takes a lake-week table (SPECS §3) with at least the
columns lake_id, week_start, area_km2_s1, area_change_2w_pct and
t2m_anomaly_c, and returns one row per lake-week with the alert level and
human-readable reasons. Everything is computed only from that week's row and
earlier rows — never from the future.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LEVEL_GREEN = "green"
LEVEL_AMBER = "amber"
LEVEL_RED = "red"


@dataclass(frozen=True)
class Thresholds:
    """Frozen Tier 2 thresholds (SPECS §5). Tuned once on 2015-2021, never
    adjusted after looking at the Shisper 2022 replay."""

    growth_2w_pct: float = 15.0
    percentile_amber: float = 90.0
    anomaly_amber_c: float = 2.0
    percentile_red: float = 95.0
    anomaly_red_c: float = 3.0
    red_weeks_in_a_row: int = 2


DEFAULT_THRESHOLDS = Thresholds()


#: A week "leads into" a drainage if the drainage happens 0-13 days after the
#: week starts — the same 14-day window as `event_in_window` (SPECS §3).
EVENT_WINDOW_DAYS = 14


def _past_only_percentile(area: pd.Series) -> pd.Series:
    """Percentile of each week's area among that lake's weeks up to and
    including it. Later weeks never change an earlier week's value, so the
    rule cannot see the future. Missing areas stay missing."""
    vals = area.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for i, v in enumerate(vals):
        if np.isnan(v):
            continue
        history = vals[: i + 1]
        history = history[~np.isnan(history)]
        out[i] = (history <= v).mean() * 100.0
    return pd.Series(out, index=area.index)


def _reasons(row, th: Thresholds) -> str:
    reasons = []
    if row["_grew_fast"]:
        reasons.append(f"area grew >{th.growth_2w_pct:g}% in 2 weeks")
    if row["_hot_and_high"]:
        reasons.append(
            f"area above lake p{th.percentile_amber:g} "
            f"and t2m anomaly >+{th.anomaly_amber_c:g} C"
        )
    if row["_red_direct"]:
        reasons.append(
            f"area above lake p{th.percentile_red:g} "
            f"and t2m anomaly >+{th.anomaly_red_c:g} C"
        )
    if row["_amber_streak"]:
        reasons.append(f"amber conditions {th.red_weeks_in_a_row} weeks in a row")
    return "; ".join(reasons)


def assign_alert_levels(
    lake_week: pd.DataFrame, thresholds: Thresholds = DEFAULT_THRESHOLDS
) -> pd.DataFrame:
    """Assign green/amber/red per lake-week following the frozen rules.

    The area percentile is computed per lake from that week and earlier weeks
    only (see `_past_only_percentile`), so every decision uses only data up
    to that week. Returns columns lake_id, week_start, level, reasons.
    """
    df = lake_week.sort_values(["lake_id", "week_start"]).copy()
    df["area_pct"] = df.groupby("lake_id", group_keys=False)["area_km2_s1"].transform(
        _past_only_percentile
    )

    df["_grew_fast"] = df["area_change_2w_pct"] > thresholds.growth_2w_pct
    df["_hot_and_high"] = (df["area_pct"] >= thresholds.percentile_amber) & (
        df["t2m_anomaly_c"] > thresholds.anomaly_amber_c
    )
    df["_amber_now"] = df["_grew_fast"] | df["_hot_and_high"]
    df["_amber_streak"] = (
        df.groupby("lake_id")["_amber_now"]
        .rolling(thresholds.red_weeks_in_a_row, min_periods=thresholds.red_weeks_in_a_row)
        .sum()
        .reset_index(level=0, drop=True)
        >= thresholds.red_weeks_in_a_row
    )
    df["_red_direct"] = (df["area_pct"] >= thresholds.percentile_red) & (
        df["t2m_anomaly_c"] > thresholds.anomaly_red_c
    )

    df["level"] = LEVEL_GREEN
    df.loc[df["_amber_now"], "level"] = LEVEL_AMBER
    df.loc[df["_red_direct"] | df["_amber_streak"], "level"] = LEVEL_RED
    df["reasons"] = df.apply(_reasons, axis=1, args=(thresholds,))
    df.loc[df["level"] == LEVEL_GREEN, "reasons"] = ""

    return df[["lake_id", "week_start", "level", "reasons"]].reset_index(drop=True)


def _raised(alerts: pd.DataFrame) -> pd.Series:
    return alerts["level"].isin([LEVEL_AMBER, LEVEL_RED])


def lead_time_days(alerts: pd.DataFrame, event_date: str) -> int | None:
    """Days between the start of the warning run that leads into a drainage
    and the drainage date (SPECS §5).

    `alerts` holds one lake's weekly rows. A warning run is a stretch of
    consecutive amber/red weeks; it counts only if its last week starts
    within EVENT_WINDOW_DAYS before the drainage. Older, unrelated alerts
    (e.g. from earlier seasons) never stretch the lead time. Returns None
    when no warning led into the event.
    """
    run = _warning_run(alerts, pd.Timestamp(event_date))
    if not run:
        return None
    return (pd.Timestamp(event_date) - min(run)).days


def _warning_run(alerts: pd.DataFrame, event: pd.Timestamp) -> list[pd.Timestamp]:
    """Week starts of the warning run leading into `event` (empty if none)."""
    a = alerts.assign(week=pd.to_datetime(alerts["week_start"]))
    a = a[a["week"] <= event].sort_values("week")
    weeks = a["week"].tolist()
    raised = _raised(a).tolist()

    i = len(weeks) - 1
    if i < 0 or not raised[i] or (event - weeks[i]).days >= EVENT_WINDOW_DAYS:
        return []
    run = [weeks[i]]
    while i > 0 and raised[i - 1]:
        i -= 1
        run.append(weeks[i])
    return run


def count_false_alarms(alerts: pd.DataFrame, event_dates: list[str]) -> int:
    """Amber/red weeks that are not part of a warning run leading into a
    drainage (SPECS §5). Uses the same run definition as `lead_time_days`,
    so a week counted in a lead time is never also a false alarm.
    `alerts` holds one lake's weekly rows; `event_dates` are that lake's
    drainage dates."""
    explained = set()
    for d in event_dates:
        explained.update(_warning_run(alerts, pd.Timestamp(d)))
    raised_weeks = pd.to_datetime(alerts.loc[_raised(alerts), "week_start"])
    return sum(week not in explained for week in raised_weeks)
