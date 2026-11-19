"""Feature tables (notebook 03): lake_year (Tier 1) and lake_week (Tier 2).

Built by notebook 03 and frozen afterwards (Gate 2): no new features after it.

**The leakage rule (CLAUDE.md, NFR-3).** A Tier 1 feature for year Y may only
use data up to 31 December of year Y-1. Every yearly feature here is computed
on a per-lake yearly source table and then *shifted forward one year*, so the
row for year Y only sees years <= Y-1. `check_no_future_leak` proves it on the
real data: it scrambles every source value from year Y onwards and checks that
the features for year Y do not change.

Weekly (Tier 2) features for week t use only data from week t and earlier;
`event_in_window` is the evaluation label and is never a feature.

Deliberately NOT features: `dam_type` (it was partly derived from the outburst
records, so it would leak the label), `name` and `tier2`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from karaglof.area import week_start

#: Climate anomalies are measured against 1980-1999, entirely before the study
#: period, so no anomaly can contain information from a later year.
CLIMATE_BASELINE = (1980, 1999)

#: Smallest area used as a denominator for growth percentages, so a lake
#: growing from (almost) nothing gives a large but finite number.
AREA_FLOOR_KM2 = 0.01

FEATURE_GROUPS = {
    "lake": ["area_km2_prev", "area_growth_3y_pct", "area_trend_km2_per_yr"],
    "glacier": ["velocity_prev", "velocity_change", "surge_ratio_prev", "surge_flag_prev"],
    "climate": ["pdd_prev_summer", "heat_anomaly_prev", "precip_prev_summer_mm",
                "snowmelt_prev_summer_mm"],
    "terrain": ["elev_m", "slope_around_deg", "dist_to_glacier_m", "parent_glacier_km2"],
}
FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]

#: Plain-language description of every lake_year column: (meaning, unit, source, time rule).
DATA_DICTIONARY = {
    "lake_id": ("Lake identifier from notebook 01", "-", "notebook 01", "key"),
    "year": ("Year the row describes", "year", "-", "key"),
    "glof": ("1 if a recorded outburst happened at the lake in this year", "0/1",
             "GLOF database v4.2 + news reports", "label, never a feature"),
    "area_km2_prev": ("Lake area in the previous melt season", "km²", "Landsat (notebook 02)",
                      "year Y-1"),
    "area_growth_3y_pct": ("Change of lake area from Y-4 to Y-1", "%", "Landsat (notebook 02)",
                           "years Y-4..Y-1"),
    "area_trend_km2_per_yr": ("Linear trend of lake area over the 5 previous years", "km²/year",
                              "Landsat (notebook 02)", "years Y-5..Y-1"),
    "velocity_prev": ("Speed of the fastest 10% of glacier ice within 3 km of the lake", "m/year",
                      "ITS_LIVE v2 annual mosaics", "year Y-1"),
    "velocity_change": ("Change of that glacier speed from Y-2 to Y-1", "m/year",
                        "ITS_LIVE v2 annual mosaics", "years Y-2..Y-1"),
    "surge_ratio_prev": ("Glacier speed in Y-1 divided by its median over the 5 years before",
                         "ratio", "ITS_LIVE v2 annual mosaics", "years Y-6..Y-1"),
    "surge_flag_prev": ("1 if surge_ratio_prev >= 2 (the glacier was surging)", "0/1",
                        "ITS_LIVE v2 annual mosaics", "years Y-6..Y-1"),
    "pdd_prev_summer": ("Positive degree-days, May-September of the previous year", "°C·day",
                        "ERA5-Land daily", "year Y-1"),
    "heat_anomaly_prev": ("Mean May-September temperature of Y-1 minus its 1980-1999 mean", "°C",
                          "ERA5-Land daily", "year Y-1 (baseline before 2000)"),
    "precip_prev_summer_mm": ("Precipitation, May-September of the previous year", "mm",
                              "ERA5-Land daily", "year Y-1"),
    "snowmelt_prev_summer_mm": ("Snowmelt, May-September of the previous year", "mm w.e.",
                                "ERA5-Land daily", "year Y-1"),
    "elev_m": ("Mean elevation of the lake surface", "m", "NASADEM (2000 SRTM)", "static"),
    "slope_around_deg": ("Mean terrain slope in a 300 m ring around the lake", "°",
                         "NASADEM (2000 SRTM)", "static"),
    "dist_to_glacier_m": ("Distance from the lake to its parent glacier outline", "m",
                          "RGI v7 (~2000 outlines)", "static"),
    "parent_glacier_km2": ("Area of the parent glacier", "km²", "RGI v7 (~2000 outlines)",
                           "static"),
}


def _grid(lake_ids, years) -> pd.MultiIndex:
    return pd.MultiIndex.from_product([list(lake_ids), list(years)], names=["lake_id", "year"])


def _shift_to_next_year(per_year: pd.DataFrame, lake_ids, years) -> pd.DataFrame:
    """Values computed for source year y become features of year y + 1."""
    out = per_year.copy()
    out.index = pd.MultiIndex.from_arrays(
        [out.index.get_level_values("lake_id"), out.index.get_level_values("year") + 1],
        names=["lake_id", "year"])
    return out.reindex(_grid(lake_ids, years))


def _trend(values: np.ndarray) -> float:
    """Least-squares slope per step, if at least 3 of the values exist."""
    x = np.arange(len(values), dtype=float)
    ok = ~np.isnan(values)
    if ok.sum() < 3:
        return np.nan
    return float(np.polyfit(x[ok], values[ok], 1)[0])


def lake_features(area: pd.DataFrame, lake_ids, years) -> pd.DataFrame:
    """From lake_area_yearly (lake_id, year, area_km2):

    area_km2_prev          area in year Y-1
    area_growth_3y_pct     % change from Y-4 to Y-1 (denominator floored at AREA_FLOOR_KM2)
    area_trend_km2_per_yr  linear trend over Y-5..Y-1 (needs >= 3 measured years)
    """
    src_years = range(min(years) - 6, max(years))
    a = area.set_index(["lake_id", "year"])["area_km2"].reindex(_grid(lake_ids, src_years))
    w = a.unstack("year")
    per = pd.DataFrame(index=a.index)
    per["area_km2_prev"] = a
    lag3 = w.shift(3, axis=1).stack(future_stack=True).reindex(a.index)
    per["area_growth_3y_pct"] = (a - lag3) / lag3.clip(lower=AREA_FLOOR_KM2) * 100
    trend = w.T.rolling(5, min_periods=1).apply(_trend, raw=True).T
    per["area_trend_km2_per_yr"] = trend.stack(future_stack=True).reindex(a.index)
    return _shift_to_next_year(per, lake_ids, years)


def glacier_features(velocity: pd.DataFrame, lake_ids, years, surge_factor: float = 2.0,
                     baseline_years: int = 5, min_baseline: int = 3) -> pd.DataFrame:
    """From glacier speed near each lake (lake_id, year, v_m_per_yr = speed of
    the fastest 10% of glacier ice within 3 km of the lake, so a surging dam
    glacier is not averaged away by slower neighbouring ice):

    velocity_prev     speed in year Y-1
    velocity_change   speed change from Y-2 to Y-1
    surge_ratio_prev  speed in Y-1 / median speed of the `baseline_years` before it
    surge_flag_prev   1 if surge_ratio_prev >= surge_factor (a surge), 0 if not, empty if unknown
    """
    src_years = range(min(years) - baseline_years - 2, max(years))
    v = velocity.set_index(["lake_id", "year"])["v_m_per_yr"].reindex(_grid(lake_ids, src_years))
    w = v.unstack("year")
    base = (w.shift(1, axis=1).T.rolling(baseline_years, min_periods=min_baseline).median().T
            .stack(future_stack=True).reindex(v.index))
    per = pd.DataFrame(index=v.index)
    per["velocity_prev"] = v
    per["velocity_change"] = v - w.shift(1, axis=1).stack(future_stack=True).reindex(v.index)
    per["surge_ratio_prev"] = v / base.where(base > 0)
    per["surge_flag_prev"] = (per["surge_ratio_prev"] >= surge_factor).astype(float).where(
        per["surge_ratio_prev"].notna())
    return _shift_to_next_year(per, lake_ids, years)


def climate_features(climate: pd.DataFrame, lake_ids, years) -> pd.DataFrame:
    """From melt-season (May-September) ERA5-Land values per lake and year
    (lake_id, year, pdd, t_mean_c, precip_mm, snowmelt_mm):

    pdd_prev_summer           positive degree-days in Y-1
    heat_anomaly_prev         mean temperature in Y-1 minus the 1980-1999 mean
    precip_prev_summer_mm     precipitation in Y-1
    snowmelt_prev_summer_mm   snowmelt in Y-1
    """
    c = climate.set_index(["lake_id", "year"])
    in_base = c.index.get_level_values("year").to_series().between(*CLIMATE_BASELINE).values
    baseline = c.loc[in_base, "t_mean_c"].groupby("lake_id").mean()
    per = pd.DataFrame(index=c.index)
    per["pdd_prev_summer"] = c["pdd"]
    per["heat_anomaly_prev"] = c["t_mean_c"] - c.index.get_level_values("lake_id").map(baseline)
    per["precip_prev_summer_mm"] = c["precip_mm"]
    per["snowmelt_prev_summer_mm"] = c["snowmelt_mm"]
    return _shift_to_next_year(per, lake_ids, years)


def build_lake_year(lakes: pd.DataFrame, events: pd.DataFrame, area: pd.DataFrame,
                    velocity: pd.DataFrame, climate: pd.DataFrame, terrain: pd.DataFrame,
                    years) -> pd.DataFrame:
    """The Tier 1 table (SPECS §3): lake_id, year, glof label, then FEATURES.

    glof = 1 if any recorded outburst at that lake fell in that year.
    """
    ids = list(lakes["lake_id"])
    out = pd.concat([lake_features(area, ids, years), glacier_features(velocity, ids, years),
                     climate_features(climate, ids, years)], axis=1).reset_index()
    out = out.merge(terrain[["lake_id"] + FEATURE_GROUPS["terrain"]], on="lake_id", how="left")
    burst = set(zip(events["lake_id"], pd.to_datetime(events["date"]).dt.year, strict=True))
    keys = zip(out["lake_id"], out["year"], strict=True)
    out["glof"] = [int(k in burst) for k in keys]
    return out[["lake_id", "year", "glof"] + FEATURES]


def check_no_future_leak(build, sources: dict[str, pd.DataFrame], year: int,
                         seed: int = 42) -> None:
    """Raise AssertionError if any feature for `year` depends on data from
    `year` or later.

    `build(**sources)` must return a lake_year table. Every numeric value in
    every source table with a `year` >= `year` is replaced by random numbers;
    the feature rows for `year` must stay exactly the same.
    """
    rng = np.random.default_rng(seed)
    before = build(**sources)
    scrambled = {}
    for name, df in sources.items():
        df = df.copy()
        if "year" in df.columns:
            future = df["year"] >= year
            for col in df.columns:
                if col not in ("lake_id", "year") and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].astype(float)       # int/bool columns can hold the noise
                    df.loc[future, col] = rng.normal(1000, 500, int(future.sum()))
        scrambled[name] = df
    after = build(**scrambled)
    a = before[before["year"] == year].set_index("lake_id")[FEATURES]
    b = after[after["year"] == year].set_index("lake_id")[FEATURES]
    pd.testing.assert_frame_equal(a, b, check_exact=True, obj=f"features for {year}")


# --------------------------------------------------------------------------- Tier 2 (weekly)

def weekly_climate(daily: pd.DataFrame) -> pd.DataFrame:
    """From daily ERA5-Land at each Tier 2 lake (lake_id, date, t2m_c, precip_mm):

    t2m_anomaly_c  weekly mean temperature minus the 1980-1999 mean of the same
                   week of the year
    pdd_7d         positive degree-days over the week
    rain_7d_mm     precipitation over the week
    """
    d = daily.copy()
    d["week_start"] = week_start(d["date"]).values
    d["pdd"] = d["t2m_c"].clip(lower=0)
    wk = d.groupby(["lake_id", "week_start"]).agg(
        t2m_7d=("t2m_c", "mean"), pdd_7d=("pdd", "sum"), rain_7d_mm=("precip_mm", "sum"),
        days=("t2m_c", "size")).reset_index()
    wk = wk[wk["days"] == 7]                                   # complete weeks only
    wk["woy"] = wk["week_start"].dt.isocalendar().week.clip(upper=52).astype(int)
    in_base = wk["week_start"].dt.year.between(*CLIMATE_BASELINE)
    base = wk[in_base].groupby(["lake_id", "woy"])["t2m_7d"].mean().rename("t2m_base")
    wk = wk.merge(base, on=["lake_id", "woy"], how="left")
    wk["t2m_anomaly_c"] = wk["t2m_7d"] - wk["t2m_base"]
    return wk[["lake_id", "week_start", "t2m_anomaly_c", "pdd_7d", "rain_7d_mm"]]


def area_change_2w_pct(weekly_area: pd.DataFrame, max_lookback_weeks: int = 4) -> pd.Series:
    """Percentage change of the weekly area against 2 weeks earlier.

    Clear looks are missing in cloudy weeks, so the reference is the most
    recent measured area 2 to `max_lookback_weeks` weeks before (2 weeks
    preferred). Only earlier weeks are used. The denominator is floored at
    AREA_FLOOR_KM2.
    """
    out = pd.Series(np.nan, index=weekly_area.index)
    for _, g in weekly_area.sort_values("week_start").groupby("lake_id"):
        area = g.set_index("week_start")["area_km2"]
        ref = pd.Series(np.nan, index=area.index)
        for lag in range(max_lookback_weeks, 1, -1):          # 4, 3, then 2 wins
            shifted = area.shift(lag, freq="7D").reindex(area.index)
            ref = shifted.where(shifted.notna(), ref)
        change = (area - ref) / ref.clip(lower=AREA_FLOOR_KM2) * 100
        out.loc[g.index] = change.values
    return out


def event_in_window(weeks: pd.DataFrame, events: pd.DataFrame, window_days: int = 14) -> pd.Series:
    """Evaluation label (never a feature): 1 if a day-dated drainage falls
    0..window_days-1 days after week_start, 0 if not. If an outburst is
    known only to the month, every week whose window overlaps that month
    is left empty (unknown), not 0."""
    label = pd.Series(0.0, index=weeks.index)
    for e in events.itertuples(index=False):
        mask = weeks["lake_id"] == e.lake_id
        date = pd.Timestamp(e.date)
        days = (date - pd.to_datetime(weeks["week_start"])).dt.days
        if e.date_certainty == "day":
            label[mask & days.between(0, window_days - 1)] = 1.0
        elif e.date_certainty == "month":
            month_end = date + pd.offsets.MonthEnd(0)
            starts = pd.to_datetime(weeks["week_start"])
            overlaps = (days <= window_days - 1) & (starts <= month_end)
            label[mask & overlaps & (label != 1.0)] = np.nan
    return label


def build_lake_week(weekly_area: pd.DataFrame, weekly_clim: pd.DataFrame,
                    events: pd.DataFrame) -> pd.DataFrame:
    """The Tier 2 table (SPECS §3): lake_id, week_start, area_km2, n_obs,
    area_change_2w_pct, t2m_anomaly_c, pdd_7d, rain_7d_mm, event_in_window."""
    w = weekly_area.copy()
    w["week_start"] = pd.to_datetime(w["week_start"])
    w = w.merge(weekly_clim, on=["lake_id", "week_start"], how="left")
    w["area_change_2w_pct"] = area_change_2w_pct(w)
    w["event_in_window"] = event_in_window(w, events)
    return w[["lake_id", "week_start", "area_km2", "n_obs", "area_change_2w_pct",
              "t2m_anomaly_c", "pdd_7d", "rain_7d_mm", "event_in_window"]]
