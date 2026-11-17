"""Tests for the feature tables (karaglof.features), above all the leakage rule:
a Tier 1 feature for year Y may only use data up to the end of year Y-1."""

import numpy as np
import pandas as pd
import pytest

from karaglof import features as kf

YEARS = list(range(2000, 2006))
LAKES = pd.DataFrame({"lake_id": ["A", "B"]})


def sources():
    src_years = range(1980, 2006)
    rng = np.random.default_rng(0)
    area = pd.DataFrame([{"lake_id": lid, "year": y, "area_km2": float(rng.uniform(0.02, 0.3))}
                         for lid in ["A", "B"] for y in range(1994, 2006)])
    velocity = pd.DataFrame([{"lake_id": lid, "year": y, "v_m_per_yr": float(rng.uniform(20, 60))}
                             for lid in ["A", "B"] for y in range(1992, 2006)])
    climate = pd.DataFrame([{"lake_id": lid, "year": y, "pdd": float(rng.uniform(300, 600)),
                             "t_mean_c": float(rng.uniform(2, 6)), "precip_mm": 100.0,
                             "snowmelt_mm": 50.0}
                            for lid in ["A", "B"] for y in src_years])
    terrain = pd.DataFrame({"lake_id": ["A", "B"], "elev_m": [4000, 4500],
                            "slope_around_deg": [20, 25], "dist_to_glacier_m": [0, 500],
                            "parent_glacier_km2": [30, 2]})
    events = pd.DataFrame({"lake_id": ["A"], "date": ["2003-05-07"]})
    return {"lakes": LAKES, "events": events, "area": area, "velocity": velocity,
            "climate": climate, "terrain": terrain}


def build(**s):
    return kf.build_lake_year(s["lakes"], s["events"], s["area"], s["velocity"], s["climate"],
                              s["terrain"], YEARS)


def test_lake_year_shape_label_and_columns():
    t = build(**sources())
    assert len(t) == 2 * len(YEARS)
    assert list(t.columns) == ["lake_id", "year", "glof"] + kf.FEATURES
    assert t.set_index(["lake_id", "year"]).loc[("A", 2003), "glof"] == 1
    assert t["glof"].sum() == 1
    for leaky in ["dam_type", "name", "tier2"]:
        assert leaky not in t.columns


def test_every_column_is_in_the_data_dictionary():
    assert list(kf.DATA_DICTIONARY) == ["lake_id", "year", "glof"] + kf.FEATURES


def test_features_use_the_previous_year():
    s = sources()
    t = build(**s).set_index(["lake_id", "year"])
    a = s["area"].set_index(["lake_id", "year"])["area_km2"]
    assert t.loc[("A", 2003), "area_km2_prev"] == a.loc[("A", 2002)]
    a02, a99 = a.loc[("A", 2002)], a.loc[("A", 1999)]
    expected = (a02 - a99) / max(a99, kf.AREA_FLOOR_KM2) * 100
    assert t.loc[("A", 2003), "area_growth_3y_pct"] == pytest.approx(expected)


@pytest.mark.parametrize("year", YEARS)
def test_no_future_leak_on_every_year(year):
    kf.check_no_future_leak(build, sources(), year)


def test_leak_check_catches_a_leaky_feature():
    def leaky_build(**s):
        t = build(**s)
        same_year = s["area"].set_index(["lake_id", "year"])["area_km2"]
        keys = zip(t.lake_id, t.year, strict=True)
        t["area_km2_prev"] = [same_year.get(k, np.nan) for k in keys]
        return t
    with pytest.raises(AssertionError):
        kf.check_no_future_leak(leaky_build, sources(), 2003)


def test_surge_flag_against_previous_years():
    v = pd.DataFrame({"lake_id": "A", "year": range(1993, 2005),
                      "v_m_per_yr": [30, 31, 29, 30, 32, 30, 31, 30, 29, 30, 95, 40]})
    g = kf.glacier_features(v, ["A"], [2003, 2004, 2005]).loc["A"]
    assert g.loc[2003, "surge_flag_prev"] == 0          # 2002 was normal
    assert g.loc[2004, "surge_flag_prev"] == 1          # 2003: 95 vs ~30 before -> surge
    assert g.loc[2004, "surge_ratio_prev"] == pytest.approx(95 / 30)
    assert g.loc[2004, "velocity_change"] == 65


def test_climate_anomaly_uses_the_1980_1999_baseline():
    c = pd.DataFrame({"lake_id": "A", "year": range(1980, 2003), "pdd": 1.0,
                      "t_mean_c": [4.0] * 20 + [5.0, 6.5, 9.0], "precip_mm": 1.0,
                      "snowmelt_mm": 1.0})
    f = kf.climate_features(c, ["A"], [2001, 2002]).loc["A"]
    assert f.loc[2001, "heat_anomaly_prev"] == pytest.approx(1.0)   # 2000: 5.0 - 4.0
    assert f.loc[2002, "heat_anomaly_prev"] == pytest.approx(2.5)   # 2001: 6.5 - 4.0


def test_weekly_climate_anomaly_and_complete_weeks():
    days = pd.date_range("1980-01-07", "2022-05-15", freq="D")
    daily = pd.DataFrame({"lake_id": "S", "date": days, "precip_mm": 1.0,
                          "t2m_c": np.where(days.year >= 2022, 3.0, 1.0)})
    w = kf.weekly_climate(daily).set_index("week_start")
    assert w.loc["2022-05-02", "t2m_anomaly_c"] == pytest.approx(2.0)
    assert w.loc["2022-05-02", "pdd_7d"] == pytest.approx(21.0)
    assert w.loc["2022-05-02", "rain_7d_mm"] == pytest.approx(7.0)
    assert pd.Timestamp("2022-05-16") not in w.index      # incomplete week dropped


def test_area_change_uses_only_earlier_weeks_and_skips_gaps():
    weeks = pd.date_range("2022-03-07", periods=6, freq="W-MON")
    wa = pd.DataFrame({"lake_id": "S", "week_start": weeks,
                       "area_km2": [0.10, np.nan, 0.12, np.nan, 0.20, 0.05]})
    ch = kf.area_change_2w_pct(wa)
    assert np.isnan(ch.iloc[0])                            # nothing earlier
    assert ch.iloc[2] == pytest.approx(20.0)               # 0.12 vs 0.10 (2 weeks back)
    assert ch.iloc[4] == pytest.approx((0.20 / 0.12 - 1) * 100)
    assert ch.iloc[5] == pytest.approx((0.05 / 0.12 - 1) * 100)   # 2 back is empty -> 3 back


def test_event_in_window_day_and_month_certainty():
    starts = pd.date_range("2022-04-04", periods=10, freq="W-MON")       # 4 Apr .. 6 Jun
    weeks = pd.DataFrame({"lake_id": "S", "week_start": starts})
    ev = pd.DataFrame({"lake_id": ["S"], "date": ["2022-05-07"], "date_certainty": ["day"]})
    # drainage 7 May: weeks of 25 Apr (12 days before) and 2 May (5 days) are 1
    assert kf.event_in_window(weeks, ev).tolist() == [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    ev_m = pd.DataFrame({"lake_id": ["S"], "date": ["2022-05-01"], "date_certainty": ["month"]})
    lab = kf.event_in_window(weeks, ev_m)
    # "sometime in May": every week whose 14-day window touches May is unknown
    assert lab.iloc[:2].tolist() == [0, 0]               # 4 and 11 Apr: windows end before May
    assert lab.iloc[2:9].isna().all()                    # 18 Apr .. 30 May
    assert lab.iloc[9] == 0                              # 6 Jun: after May
