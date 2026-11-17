"""Lake area over time (notebook 02): yearly for all lakes, weekly for Tier 2.

Weekly areas come from optical satellites (Sentinel-2 from 2017, Landsat 8/9):
Sentinel-1 radar was tested and does not see the ice-covered Shisper lake.

Earth Engine produces raw numbers per lake: water area and how many clear
satellite looks there were. These functions turn them into the released
tables, with the rules written down and tested here instead of in notebook
cells:

    yearly_table   -> lake_area_yearly.csv  (one row per lake per year)
    weekly_series  -> lake_area_weekly.csv  (one row per Tier 2 lake per week)
    event_response -> did the weekly area drop when a recorded outburst happened?

No gap is ever filled silently (SPECS §4 check): a lake-year with too few clear
looks gets an empty area, and a week with no usable clear look stays empty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def yearly_table(raw: pd.DataFrame, lake_ids, years, min_clear_obs: int) -> pd.DataFrame:
    """One row per lake per year: lake_id, year, area_km2, n_clear_obs, filled.

    `raw` has lake_id, year, water_km2 (largest-season water area) and
    clear_obs (median number of clear looks over the lake). Lake-years
    missing from `raw` or with fewer than `min_clear_obs` clear looks get an
    empty area. Nothing is interpolated, so `filled` is always False here.
    """
    grid = pd.MultiIndex.from_product([list(lake_ids), list(years)], names=["lake_id", "year"])
    out = raw.set_index(["lake_id", "year"])[["water_km2", "clear_obs"]].reindex(grid)
    out["n_clear_obs"] = out["clear_obs"].fillna(0).round().astype(int)
    out["area_km2"] = out["water_km2"].round(4)
    out.loc[out["n_clear_obs"] < min_clear_obs, "area_km2"] = np.nan
    out["filled"] = False
    return out.reset_index()[["lake_id", "year", "area_km2", "n_clear_obs", "filled"]]


def week_start(dates) -> pd.Series:
    """Monday of the week containing each date (the lake_week key, SPECS §3)."""
    d = pd.to_datetime(pd.Series(dates)).dt.normalize()
    return d - pd.to_timedelta(d.dt.weekday, unit="D")


def weekly_series(acq: pd.DataFrame, lake_ids, start: str, end: str,
                  min_valid_frac: float) -> pd.DataFrame:
    """Weekly area per lake: lake_id, week_start, area_km2, n_obs.

    `acq` has one row per satellite look and lake: lake_id, date, water_km2,
    valid_frac (share of the lake zone seen clearly). Looks covering less
    than `min_valid_frac` are dropped; the rest are combined per week
    with the median, which resists a single noisy look. Every week from
    `start` to `end` appears; weeks without a usable look stay empty.
    """
    a = acq[acq["valid_frac"] >= min_valid_frac].copy()
    a["week_start"] = week_start(a["date"]).values
    agg = a.groupby(["lake_id", "week_start"]).agg(
        area_km2=("water_km2", "median"), n_obs=("water_km2", "size"))
    weeks = pd.date_range(week_start([start]).iloc[0], end, freq="W-MON")
    grid = pd.MultiIndex.from_product([list(lake_ids), weeks], names=["lake_id", "week_start"])
    out = agg.reindex(grid)
    out["n_obs"] = out["n_obs"].fillna(0).astype(int)
    out["area_km2"] = out["area_km2"].round(4)
    return out.reset_index()


def event_response(weekly: pd.DataFrame, events: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """Mean weekly area in the `weeks` before and after each outburst.

    The week containing the drainage date is left out of both windows. A
    real outburst should show a clear drop (negative change_pct). This is a
    data check of the weekly record, not a tuning step.
    """
    rows = []
    for e in events.itertuples(index=False):
        w = weekly[weekly["lake_id"] == e.lake_id]
        event_week = week_start([e.date]).iloc[0]
        span = pd.Timedelta(days=7 * int(weeks))
        before = w[(w["week_start"] < event_week) & (w["week_start"] >= event_week - span)]
        after = w[(w["week_start"] > event_week) & (w["week_start"] <= event_week + span)]
        b, a = before["area_km2"].mean(), after["area_km2"].mean()
        change = (a / b - 1) * 100 if pd.notna(a) and pd.notna(b) and b > 0 else np.nan
        rows.append({
            "event_id": e.event_id, "lake_id": e.lake_id, "date": e.date,
            "weeks_with_data_before": int(before["area_km2"].notna().sum()),
            "weeks_with_data_after": int(after["area_km2"].notna().sum()),
            "before_km2": round(b, 4) if pd.notna(b) else np.nan,
            "after_km2": round(a, 4) if pd.notna(a) else np.nan,
            "change_pct": round(change, 1) if pd.notna(change) else np.nan,
        })
    return pd.DataFrame(rows)
