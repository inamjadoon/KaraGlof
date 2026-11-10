"""Tests for the lake and event table rules (karaglof.labels)."""

import pandas as pd
import pytest

from karaglof import labels


@pytest.mark.parametrize("value, expected", [
    (pd.Timestamp("2022-05-07 00:00:00"), ("2022-05-07", "day")),
    ("2019-06-22", ("2019-06-22", "day")),
    ("2017-07-28 00:00:00", ("2017-07-28", "day")),
    ("2021-05", ("2021-05-01", "month")),
    ("2018", ("2018-01-01", "year")),
    (2010, ("2010-01-01", "year")),
    (2010.0, ("2010-01-01", "year")),
    ("﻿2008-01-06", ("2008-01-06", "day")),
])
def test_parse_partial_date(value, expected):
    date, certainty = labels.parse_partial_date(value)
    assert (date.strftime("%Y-%m-%d"), certainty) == expected


@pytest.mark.parametrize("value", [None, float("nan"), "Late 18th Century", "", "nan"])
def test_unparseable_dates_give_none(value):
    assert labels.parse_partial_date(value) == (None, None)


def test_range_uses_earliest_possible_date():
    date, certainty, note = labels.event_date({"Date": float("nan"), "Date_Min": "1860"})
    assert (date.year, certainty) == (1860, "year")
    assert "Date_Min" in note


def db_rows(**overrides):
    base = {"ID_old": 1, "Glacier": "Khurdopin", "Lake": "", "Lake_type": "ice",
            "RGI_Glacier_Id": "RGI60-14.04404", "lat": 36.34, "lon": 75.48,
            "Date": "2017-07-28", "Date_Min": "", "Date_Max": "", "Mechanism": ""}
    return {**base, **overrides}


def test_prepare_events_reasons():
    db = pd.DataFrame([
        db_rows(),
        db_rows(Lake_type="water pocket"),
        db_rows(Date="1999"),
        db_rows(Glacier="Attabad"),
        db_rows(Date="", Date_Min=""),
        db_rows(),
        db_rows(Date=1533),  # older than pandas' datetime range
    ])
    inside = pd.Series([True, True, True, True, True, False, True])
    ev = labels.prepare_events(db, inside)
    reasons = ev["exclude_reason"].tolist()
    assert reasons[0] == ""
    assert "water pocket" in reasons[1]
    assert "study period" in reasons[2]
    assert "excluded site" in reasons[3]
    assert reasons[4] == "no usable date"
    assert "outside" in reasons[5]
    assert ev.loc[0, "dam_type"] == "ice"
    # Undated or centuries-old records must not break the date column.
    assert pd.api.types.is_datetime64_any_dtype(ev["date"])
    assert ev.loc[6, "date_text"] == "1533-01-01"
    assert pd.isna(ev.loc[6, "date"])


def test_assign_lake_ids_is_stable_and_per_basin():
    lakes = pd.DataFrame({
        "basin": ["hunza", "shigar", "hunza", "shimshal"],
        "lat": [36.30, 35.70, 36.45, 36.34],
        "lon": [74.50, 75.90, 74.88, 75.48],
    })
    a = labels.assign_lake_ids(lakes)
    b = labels.assign_lake_ids(lakes.sample(frac=1, random_state=0))
    assert a["lake_id"].tolist() == b["lake_id"].tolist()
    ids = dict(zip(a["lat"], a["lake_id"], strict=True))
    assert ids[36.45] == "KG-HUN-001"  # northernmost Hunza lake first
    assert ids[36.30] == "KG-HUN-002"
    assert ids[35.70] == "KG-SHG-001"
    assert ids[36.34] == "KG-SHM-001"


def test_match_events_radius_and_reasons():
    events = pd.DataFrame({"exclude_reason": ["", "", "water pocket: ..."]})
    dist = pd.DataFrame({"KG-HUN-001": [500.0, 9000.0, 10.0],
                         "KG-HUN-002": [2500.0, 7000.0, 20.0]})
    m = labels.match_events(events, dist, radius_m=3000)
    assert m["lake_id"].tolist() == ["KG-HUN-001", "", ""]
    assert "no lake within 3000 m" in m.loc[1, "exclude_reason"]
    assert m.loc[2, "exclude_reason"] == "water pocket: ..."  # never re-matched


def matched_events():
    return pd.DataFrame({
        "lake_id": ["KG-SHM-001", "KG-SHM-001", "KG-HUN-001", ""],
        "date": pd.to_datetime(["2018-05-10", "2018-04-01", "2022-05-07", "2001-01-01"]),
        "date_certainty": ["day", "month", "day", "year"],
        "db_id": ["10", "11", "12", "13"],
        "glacier": ["Khurdopin", "Khurdopin", "Shisphare", "X"],
        "lake_type": ["ice"] * 4, "db_rgi_id": [""] * 4, "match_dist_m": [100.0] * 4,
        "mechanism": [""] * 4, "date_min": [""] * 4, "date_max": [""] * 4,
        "date_note": [""] * 4,
        "exclude_reason": ["", "", "", "no lake within 3000 m"],
    })


def test_build_events_table_ids_and_columns():
    ev = labels.build_events_table(matched_events())
    assert list(ev.columns) == ["event_id", "lake_id", "date", "date_certainty",
                                "source", "notes"]
    assert ev["event_id"].tolist() == ["EV-2018-KG-SHM-001", "EV-2018-KG-SHM-001-2",
                                       "EV-2022-KG-HUN-001"]
    assert ev["date"].tolist() == ["2018-04-01", "2018-05-10", "2022-05-07"]


def lake_table(**overrides):
    base = pd.DataFrame({
        "lake_id": ["KG-HUN-001", "KG-HUN-002", "KG-SHM-001"],
        "name": ["Shisper", "Passu", "Khurdopin"],
        "tier2": [True, True, True],
    })
    for k, v in overrides.items():
        base[k] = v
    return base


def test_check_labels_passes_on_clean_tables():
    ev = labels.build_events_table(matched_events())
    unmatched = matched_events()[lambda d: d["lake_id"] == ""]
    lakes = lake_table(lake_id=["KG-HUN-001", "KG-HUN-002", "KG-SHM-001"])
    assert labels.check_labels(lakes, ev, unmatched, ["Shisper", "Passu", "Khurdopin"]) == []


def test_check_labels_catches_problems():
    ev = labels.build_events_table(matched_events())
    unmatched = pd.DataFrame({"exclude_reason": [""]})
    lakes = lake_table(name=["Shisper", "Attabad", "Khurdopin"], tier2=[True, False, True])
    problems = " | ".join(labels.check_labels(lakes, ev, unmatched,
                                              ["Shisper", "Passu", "Khurdopin"]))
    assert "without a reason" in problems
    assert "'Passu' found 0 times" in problems
    assert "excluded site 'attabad'" in problems
