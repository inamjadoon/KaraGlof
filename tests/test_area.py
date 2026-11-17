"""Tests for lake-area tables (karaglof.area)."""

import numpy as np
import pandas as pd

from karaglof.area import event_response, week_start, weekly_series, yearly_table


def test_yearly_table_full_grid_and_no_silent_fill():
    raw = pd.DataFrame({"lake_id": ["A", "A", "B"], "year": [2000, 2001, 2000],
                        "water_km2": [0.10, 0.12, 0.05], "clear_obs": [5, 1, 4]})
    t = yearly_table(raw, ["A", "B"], [2000, 2001], min_clear_obs=3)
    assert len(t) == 4                                   # every lake x every year
    a = t.set_index(["lake_id", "year"])
    assert a.loc[("A", 2000), "area_km2"] == 0.10
    assert np.isnan(a.loc[("A", 2001), "area_km2"])      # only 1 clear look -> empty
    assert np.isnan(a.loc[("B", 2001), "area_km2"])      # missing -> empty, not 0
    assert a.loc[("B", 2001), "n_clear_obs"] == 0
    assert not t["filled"].any()


def test_week_start_is_monday():
    assert week_start(["2022-05-07"]).iloc[0] == pd.Timestamp("2022-05-02")   # Saturday -> Monday
    assert week_start(["2022-05-02"]).iloc[0] == pd.Timestamp("2022-05-02")


def test_weekly_series_median_coverage_and_gaps():
    acq = pd.DataFrame({
        "lake_id": ["S", "S", "S", "S"],
        "date": ["2022-04-04", "2022-04-06", "2022-04-08", "2022-04-20"],
        "water_km2": [0.20, 0.30, 0.90, 0.25],
        "valid_frac": [1.0, 1.0, 1.0, 0.5],              # last image covers only half the lake
    })
    w = weekly_series(acq, ["S"], "2022-04-04", "2022-04-25", min_valid_frac=0.9)
    assert w["week_start"].dt.strftime("%Y-%m-%d").tolist() == [
        "2022-04-04", "2022-04-11", "2022-04-18", "2022-04-25"]
    assert w.loc[0, "area_km2"] == 0.30               # median resists the 0.90 outlier
    assert w.loc[0, "n_obs"] == 3
    assert np.isnan(w.loc[2, "area_km2"])             # partial image dropped, week stays empty
    assert w.loc[2, "n_obs"] == 0


def test_event_response_detects_a_drop():
    weeks = pd.date_range("2022-03-28", periods=10, freq="W-MON")
    area = [0.20, 0.22, 0.25, 0.30, 0.32, 0.05, 0.04, 0.04, 0.05, 0.05]
    weekly = pd.DataFrame({"lake_id": "S", "week_start": weeks, "area_km2": area})
    events = pd.DataFrame({"event_id": ["EV"], "lake_id": ["S"], "date": ["2022-05-04"]})
    r = event_response(weekly, events, weeks=3).iloc[0]
    assert r.weeks_with_data_before == 3 and r.weeks_with_data_after == 3
    assert r.change_pct < -50
