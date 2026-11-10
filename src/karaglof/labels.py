"""Lake and event tables (notebook 01): lakes.csv and events.csv (SPECS §3).

Event source: the Glacial Lake Outburst Flood Database by Lützow & Veh
(v4.2, Zenodo, CC BY 4.0). It compiles the regional inventories (Bhambri et
al., Veh et al., ICIMOD, ...) into one sheet per RGI region; the Karakoram is
in sheet RGI60_14_SouthAsiaWest.

This module holds every rule that turns raw records into labels, so each rule
is unit-tested and the notebook only calls these functions:

    load_glof_database  -> clean raw rows (drop header rows, fix text)
    parse_partial_date  -> "2018", "2018-04", "2018-04-10", Timestamp -> date + certainty
    prepare_events      -> one candidate row per record, with exclusion reasons
    assign_lake_ids     -> stable IDs like KG-HUN-001
    match_events        -> attach each event to the nearest lake within a radius
    build_events_table  -> final events.csv rows with stable event IDs
    check_labels        -> Gate 1 checks (every event has a lake + date, or a reason)
"""

from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd

GLOF_DB_RECORD = "https://zenodo.org/api/records/23370149/files"
GLOF_DB_FILE = "glofdatabase_v4-2.ods"
GLOF_DB_URL = f"{GLOF_DB_RECORD}/{GLOF_DB_FILE}/content"
GLOF_DB_SOURCE = "Lutzow & Veh GLOF database v4.2 (doi:10.5281/zenodo.23370149)"
GLOF_DB_SHEET = "RGI60_14_SouthAsiaWest"

STUDY_YEARS = (2000, 2025)
BASIN_CODES = {"hunza": "HUN", "shimshal": "SHM", "shigar": "SHG"}
CERTAINTIES = ("day", "month", "year")

#: Database lake types mapped to the lakes.csv dam_type values (SPECS §3).
DAM_TYPES = {"ice": "ice", "moraine": "moraine", "bedrock": "bedrock"}
#: Database lake types that are not a lake at all: an englacial water pocket
#: bursts from inside the glacier, so there is no mapped lake to rank.
NOT_A_LAKE = {"water pocket"}
#: Sites that must never be labelled as GLOF lakes (PRD §4).
EXCLUDED_SITES = {"attabad": "landslide-dammed lake, not glacial",
                  "gayari": "avalanche site, not a lake"}

_DATE_RE = re.compile(r"^(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?(?:[ T].*)?$")


def _clean_text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    # Some cells carry an invisible byte-order mark (U+FEFF) before the text.
    return str(value).replace("﻿", "").strip()


def load_glof_database(path, sheet: str = GLOF_DB_SHEET) -> pd.DataFrame:
    """Read one region sheet and keep only real event rows.

    The first rows of every sheet describe the columns and give an example;
    they have no numeric coordinates, so dropping rows without numeric
    lat/lon removes them (and any record that cannot be located).
    """
    raw = pd.read_excel(path, engine="odf", sheet_name=sheet)
    df = raw.copy()
    df["lat"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    for col in ["Glacier", "RGI_Glacier_Id", "Lake", "Lake_type", "River",
                "Mechanism", "Reference"]:
        if col in df:
            df[col] = df[col].map(_clean_text)
    df["Lake_type"] = df["Lake_type"].str.lower()
    return df


def parse_partial_date(value) -> tuple[pd.Timestamp | None, str | None]:
    """Turn a database date cell into (first day, certainty).

    Cells can be a full timestamp, a year number (2010 or 2010.0) or text
    "YYYY", "YYYY-MM" or "YYYY-MM-DD". Anything else ("Late 18th Century")
    gives (None, None).
    """
    if value is None:
        return None, None
    if isinstance(value, pd.Timestamp | datetime):
        return pd.Timestamp(value).normalize(), "day"
    if isinstance(value, int | float | np.integer | np.floating):
        if np.isnan(value) or not float(value).is_integer():
            return None, None
        return pd.Timestamp(int(value), 1, 1), "year"
    m = _DATE_RE.match(_clean_text(value))
    if m is None:
        return None, None
    year, month, day = m.groups()
    if day is not None:
        return pd.Timestamp(int(year), int(month), int(day)), "day"
    if month is not None:
        return pd.Timestamp(int(year), int(month), 1), "month"
    return pd.Timestamp(int(year), 1, 1), "year"


def event_date(row) -> tuple[pd.Timestamp | None, str | None, str]:
    """Drainage date of one record: `Date`, else the earliest possible date
    (`Date_Min`) as SPECS §3 asks for ranges. Returns (date, certainty, note)."""
    date, certainty = parse_partial_date(row.get("Date"))
    if date is not None:
        return date, certainty, ""
    date, certainty = parse_partial_date(row.get("Date_Min"))
    if date is not None:
        return date, certainty, "date is the earliest possible date (Date_Min)"
    return None, None, ""


def prepare_events(db: pd.DataFrame, in_study_area: pd.Series) -> pd.DataFrame:
    """One row per database record with its parsed date, dam type and, when
    it cannot become a label, the reason (`exclude_reason`; empty = usable).

    `in_study_area` is a boolean Series aligned with `db`: True when the
    record's point lies inside the three basins (computed in the notebook).
    """
    rows = []
    for i, r in db.iterrows():
        date, certainty, date_note = event_date(r)
        lake_type = r["Lake_type"]
        names = f"{r['Glacier']} {r['Lake']}".lower()
        reason = ""
        if not in_study_area.loc[i]:
            reason = "outside the Hunza, Shimshal and Shigar basins"
        elif lake_type in NOT_A_LAKE:
            reason = f"{lake_type}: englacial outburst, no mappable lake"
        elif any(site in names for site in EXCLUDED_SITES):
            site = next(s for s in EXCLUDED_SITES if s in names)
            reason = f"excluded site ({EXCLUDED_SITES[site]})"
        elif date is None:
            reason = "no usable date"
        elif not STUDY_YEARS[0] <= date.year <= STUDY_YEARS[1]:
            reason = f"outside the study period {STUDY_YEARS[0]}-{STUDY_YEARS[1]}"
        rows.append({
            "db_row": i,
            "db_id": _clean_text(r.get("ID_old", r.get("ID"))),
            "glacier": r["Glacier"],
            "lake_name": r["Lake"],
            "lake_type": lake_type,
            "dam_type": DAM_TYPES.get(lake_type, "unknown"),
            "db_rgi_id": r["RGI_Glacier_Id"],
            "lat": r["lat"],
            "lon": r["lon"],
            # Text keeps every date readable (some are centuries old); the
            # datetime column is only filled inside the study period.
            "date_text": date.strftime("%Y-%m-%d") if date is not None else "",
            "date": date if reason == "" else None,
            "date_certainty": certainty,
            "date_note": date_note,
            "date_min": _clean_text(r.get("Date_Min")),
            "date_max": _clean_text(r.get("Date_Max")),
            "mechanism": r.get("Mechanism", ""),
            "exclude_reason": reason,
        })
    out = pd.DataFrame(rows)
    # Without this, unusable rows would leave a mixed object column.
    out["date"] = pd.to_datetime(out["date"])
    return out


def assign_lake_ids(lakes: pd.DataFrame) -> pd.DataFrame:
    """Give every lake a stable ID `KG-<BASIN>-<NNN>`.

    Within a basin, lakes are numbered north to south (then west to east),
    so the same inventory always gets the same IDs.
    """
    out = lakes.copy()
    out["_lat"] = out["lat"].round(5)
    out["_lon"] = out["lon"].round(5)
    out = out.sort_values(["basin", "_lat", "_lon"], ascending=[True, False, True])
    out["lake_id"] = [
        f"KG-{BASIN_CODES[b]}-{n:03d}"
        for b, n in zip(out["basin"], out.groupby("basin").cumcount() + 1, strict=True)
    ]
    return out.drop(columns=["_lat", "_lon"]).reset_index(drop=True)


def match_events(events: pd.DataFrame, lake_dist_m: pd.DataFrame,
                 radius_m: float) -> pd.DataFrame:
    """Attach each usable event to its nearest lake.

    `lake_dist_m` has one row per event (same index as `events`) and one
    column per lake_id, holding the distance in metres from the event point
    to the lake polygon (0 if inside). Database points often sit on the
    glacier or dam rather than in the lake, hence a search radius.
    Events with no lake within `radius_m` get an exclusion reason instead.
    """
    out = events.copy()
    out["lake_id"] = ""
    out["match_dist_m"] = np.nan
    usable = out["exclude_reason"] == ""
    if usable.any() and lake_dist_m.shape[1] > 0:
        d = lake_dist_m.loc[out.index[usable]]
        nearest = d.idxmin(axis=1)
        dist = d.min(axis=1)
        ok = dist <= radius_m
        out.loc[ok[ok].index, "lake_id"] = nearest[ok]
        out.loc[dist.index, "match_dist_m"] = dist.round(0)
        far = ok[~ok].index
        out.loc[far, "exclude_reason"] = [
            f"no lake within {radius_m:.0f} m (nearest {dist[i]:.0f} m)" for i in far
        ]
    elif usable.any():
        out.loc[usable, "exclude_reason"] = "no lakes in inventory"
    return out


def build_events_table(matched: pd.DataFrame, source: str = GLOF_DB_SOURCE) -> pd.DataFrame:
    """Final events.csv rows (SPECS §3) for matched events.

    IDs follow `EV-<year>-<lake_id>`; a second event at the same lake in
    the same year gets `-2`, a third `-3`, in date order.
    """
    ev = matched[matched["lake_id"] != ""].sort_values(["lake_id", "date"]).copy()
    ev["year"] = ev["date"].dt.year
    seq = ev.groupby(["lake_id", "year"]).cumcount() + 1
    ev["event_id"] = [
        f"EV-{y}-{lake}" + (f"-{n}" if n > 1 else "")
        for y, lake, n in zip(ev["year"], ev["lake_id"], seq, strict=True)
    ]
    notes = []
    for r in ev.itertuples(index=False):
        parts = [f"glacier={r.glacier}", f"db_lake_type={r.lake_type}",
                 f"db_rgi_id={r.db_rgi_id}", f"match_dist_m={r.match_dist_m:.0f}"]
        if r.mechanism:
            parts.append(f"mechanism={r.mechanism}")
        if r.date_min or r.date_max:
            parts.append(f"db_range={r.date_min}..{r.date_max}")
        if r.date_note:
            parts.append(r.date_note)
        notes.append("; ".join(parts))
    return pd.DataFrame({
        "event_id": ev["event_id"],
        "lake_id": ev["lake_id"],
        "date": ev["date"].dt.strftime("%Y-%m-%d"),
        "date_certainty": ev["date_certainty"],
        "source": [f"{source}, record {i}" for i in ev["db_id"]],
        "notes": notes,
    }).sort_values(["date", "lake_id"]).reset_index(drop=True)


def check_labels(lakes: pd.DataFrame, events: pd.DataFrame, unmatched: pd.DataFrame,
                 tier2_names: list[str]) -> list[str]:
    """Gate 1 checks (SPECS §4, §9). Returns a list of problems; empty = pass."""
    problems = []
    if lakes["lake_id"].duplicated().any():
        problems.append("duplicate lake_id values")
    if events["event_id"].duplicated().any():
        problems.append("duplicate event_id values")
    unknown = set(events["lake_id"]) - set(lakes["lake_id"])
    if unknown:
        problems.append(f"events point to unknown lakes: {sorted(unknown)}")
    if events["date"].isna().any() or (events["date"] == "").any():
        problems.append("events without a date")
    if not events["date_certainty"].isin(CERTAINTIES).all():
        problems.append("events with a date_certainty outside day/month/year")
    if (unmatched["exclude_reason"] == "").any():
        problems.append("unmatched records without a reason")
    for name in tier2_names:
        hit = lakes[(lakes["name"] == name) & lakes["tier2"]]
        if len(hit) != 1:
            problems.append(f"Tier 2 lake {name!r} found {len(hit)} times (need exactly 1)")
    if int(lakes["tier2"].sum()) != len(tier2_names):
        problems.append(f"{int(lakes['tier2'].sum())} lakes flagged tier2, "
                        f"expected {len(tier2_names)}")
    names = lakes["name"].fillna("").str.lower()
    for site in EXCLUDED_SITES:
        if names.str.contains(site).any():
            problems.append(f"excluded site {site!r} is in the lake table")
    return problems
