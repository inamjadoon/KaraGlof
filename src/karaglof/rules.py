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


def _amber_mask(df: pd.DataFrame, th: Thresholds) -> pd.Series:
    grew_fast = df["area_change_2w_pct"] > th.growth_2w_pct
    hot_and_high = (df["area_pct"] >= th.percentile_amber) & (
        df["t2m_anomaly_c"] > th.anomaly_amber_c
    )
    return grew_fast | hot_and_high


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

    The area percentile is computed per lake over the provided weekly series
    (the lake's full weekly record); only the current and earlier rows feed
    each decision. Returns columns lake_id, week_start, level, reasons.
    """
    df = lake_week.sort_values(["lake_id", "week_start"]).copy()
    df["area_pct"] = df.groupby("lake_id")["area_km2_s1"].rank(pct=True) * 100.0

    df["_grew_fast"] = df["area_change_2w_pct"] > thresholds.growth_2w_pct
    df["_hot_and_high"] = (df["area_pct"] >= thresholds.percentile_amber) & (
        df["t2m_anomaly_c"] > thresholds.anomaly_amber_c
    )
    df["_amber_now"] = _amber_mask(df, thresholds)
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


def lead_time_days(alerts: pd.DataFrame, event_date: str) -> int | None:
    """Days between the first amber-or-red week and the drainage date."""
    raised = alerts[alerts["level"].isin([LEVEL_AMBER, LEVEL_RED])]
    if raised.empty:
        return None
    first = raised["week_start"].min()
    return (pd.Timestamp(event_date) - pd.Timestamp(first)).days
