"""Generate the sample released files served by the API (this directory).

Run from the repository root after `pip install -e .` (or rely on the
sys.path fallback below):

    python api/data/make_sample_data.py

Every file matches the released-file schemas in docs/SPECS.md §3 so the real
notebook outputs can replace them later without code changes. All values are
synthetic but internally consistent: alert levels are computed by the real
frozen rules in karaglof.rules, lead times in metrics.json match the alerts,
and ranking ranks follow the scores.

The real pipeline overwrites these files (see docs/SPECS.md §4); this script
only exists to regenerate believable samples.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd  # noqa: E402

from karaglof.rules import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    assign_alert_levels,
    count_false_alarms,
    lead_time_days,
)

DATA_DIR = Path(__file__).resolve().parent
WEEK = timedelta(weeks=1)

LAKES = [
    # lake_id, name, basin, lat, lon, area_ref_km2, dam_type, glacier_id, tier2
    ("KG-HUN-001", "Shisper", "hunza", 36.4339, 74.8547, 0.28, "ice", "RGI60-14.00011", True),
    ("KG-HUN-002", "Passu", "hunza", 36.4621, 74.8973, 0.12, "ice", "RGI60-14.00023", True),
    ("KG-HUN-012", "", "hunza", 36.5187, 74.6631, 0.06, "moraine", "RGI60-14.00058", False),
    ("KG-SHM-001", "Khurdopin", "shimshal", 36.3912, 75.6981, 0.20, "ice", "RGI60-14.00137", True),
    ("KG-SHM-003", "", "shimshal", 36.4498, 75.6044, 0.04, "moraine", "RGI60-14.00192", False),
    ("KG-SHG-004", "", "shigar", 35.6012, 75.4233, 0.09, "bedrock", "RGI60-14.00241", False),
]

EVENTS = [
    # event_id, lake_id, date, date_certainty, source, notes
    ("EV-2019-SHI", "KG-HUN-001", "2019-07-21", "day", "Bhambri et al. inventory", ""),
    ("EV-2020-SHI", "KG-HUN-001", "2020-06-28", "day", "Bhambri et al. inventory", ""),
    ("EV-2022-SHI", "KG-HUN-001", "2022-05-07", "day", "news + Bhambri et al.",
     "Hassanabad bridge destroyed"),
    ("EV-2022-KHU", "KG-SHM-001", "2022-07-15", "month", "Bhambri et al. inventory",
     "Date known to the month"),
]


def write_csv(name: str, header: list[str], rows: list[list]) -> None:
    with (DATA_DIR / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def weekly_rows(start: date, areas: list[float], anomalies: list[float], pdds: list[int],
                rains: list[float], event_date: date | None) -> list[list]:
    """Build lake_week rows; the 2-week area change is derived from the curve."""
    rows = []
    for i, area in enumerate(areas):
        week = start + i * WEEK
        # No 2-week change exists for the first two weeks: leave it empty, not 0.
        change_2w = None if i < 2 else round((area / areas[i - 2] - 1.0) * 100.0, 1)
        in_window = int(event_date is not None and 0 <= (event_date - week).days <= 13)
        rows.append([
            week.isoformat(),
            round(area, 3),
            change_2w,
            round(anomalies[i], 1),
            pdds[i],
            round(rains[i], 1),
            in_window,
        ])
    return rows


def build_lake_week() -> list[list]:
    """One melt-season 2022 series per Tier 2 lake (SPECS §3 lake_week schema)."""
    header = ["lake_id", "week_start", "area_km2_s1", "area_change_2w_pct",
              "t2m_anomaly_c", "pdd_7d", "rain_7d_mm", "event_in_window"]
    rows: list[list] = []

    # Shisper: fills slowly through early April, fast from mid-April 2022,
    # drains on 7 May. First amber 18 April (record area + >2 C anomaly), red
    # from 25 April (amber streak), lead time 19 days.
    rows += [["KG-HUN-001"] + r for r in weekly_rows(
        date(2022, 3, 7),
        areas=[0.20, 0.205, 0.21, 0.22, 0.23, 0.24, 0.26, 0.30, 0.35,
               0.07, 0.072, 0.074, 0.077, 0.08, 0.083, 0.086, 0.089, 0.092],
        anomalies=[0.4, 0.6, 0.9, 1.2, 1.5, 1.9, 2.2, 2.8, 3.4,
                   1.1, 0.8, 0.7, 0.9, 1.0, 1.2, 1.4, 1.5, 1.6],
        pdds=[12, 15, 19, 24, 30, 38, 45, 58, 66, 40, 30, 28, 30, 34, 38, 42, 45, 46],
        rains=[2.1, 1.4, 3.2, 0.8, 1.1, 2.6, 1.9, 0.4, 0.2,
               6.8, 3.1, 1.2, 0.9, 0.6, 1.3, 0.7, 0.5, 0.4],
        event_date=date(2022, 5, 7),
    )]

    # Khurdopin: steady fill, first amber 27 June, red 4 July, drains 15 July.
    rows += [["KG-SHM-001"] + r for r in weekly_rows(
        date(2022, 5, 30),
        areas=[0.14, 0.145, 0.15, 0.16, 0.19, 0.23, 0.26, 0.22, 0.15, 0.12, 0.11, 0.10, 0.10],
        anomalies=[0.8, 1.0, 1.3, 1.6, 1.9, 2.6, 3.1, 1.8, 0.9, 0.6, 0.5, 0.5, 0.4],
        pdds=[22, 26, 31, 37, 44, 55, 61, 42, 28, 22, 19, 18, 17],
        rains=[1.8, 1.1, 2.4, 0.9, 1.5, 0.6, 0.3, 4.2, 2.8, 1.6, 0.9, 0.7, 0.5],
        event_date=date(2022, 7, 15),
    )]

    # Passu: quiet season with one hot-week blip in July (a counted false alarm).
    rows += [["KG-HUN-002"] + r for r in weekly_rows(
        date(2022, 6, 6),
        areas=[0.100, 0.101, 0.102, 0.103, 0.104, 0.106, 0.112, 0.105, 0.104,
               0.103, 0.103, 0.102, 0.102, 0.101, 0.101, 0.100],
        anomalies=[0.5, 0.7, 0.9, 1.1, 1.4, 2.1, 2.6, 0.8, 0.6, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2],
        pdds=[25, 29, 34, 39, 46, 57, 63, 35, 27, 23, 20, 19, 18, 17, 15, 15],
        rains=[1.2, 0.8, 1.9, 0.6, 1.0, 0.4, 0.2, 3.6, 2.1, 1.2, 0.8, 0.6, 0.5, 0.4, 0.3, 0.3],
        event_date=None,
    )]
    return header, rows


def build_lake_area_yearly() -> list[list]:
    """Yearly largest summer area per lake (notebook 02 output shape)."""
    rows = []
    shisper = {2000: 0.01, 2005: 0.05, 2010: 0.12, 2015: 0.19, 2018: 0.22,
               2019: 0.28, 2020: 0.30, 2021: 0.31, 2022: 0.37, 2023: 0.05,
               2024: 0.10, 2025: 0.14}
    for lake_id, *_ in LAKES:
        if lake_id == "KG-HUN-001":
            curve = shisper
        elif lake_id == "KG-SHM-001":
            curve = {2015: 0.08, 2018: 0.13, 2020: 0.15, 2021: 0.17, 2022: 0.26,
                     2023: 0.11, 2024: 0.13, 2025: 0.16}
        else:
            base = 0.05 if lake_id != "KG-SHG-004" else 0.08
            curve = {y: round(base + 0.002 * (y - 2015), 3) for y in range(2015, 2026)}
        years = sorted(curve)
        for y in years:
            filled = lake_id == "KG-SHG-004" and y == 2015
            rows.append([lake_id, y, curve[y], filled])
    return ["lake_id", "year", "area_km2", "filled"], rows


def build_lake_year() -> list[list]:
    """Tier 1 model table (SPECS §3), one row per lake per year 2020-2025."""
    header = ["lake_id", "year", "glof",
              "area_km2_prev", "area_growth_3y", "area_growth_rate",
              "velocity_prev", "velocity_change", "surge_flag_prev", "dh_dt",
              "pdd_prev_summer", "heat_anomaly_prev", "rain_prev_summer", "snowmelt_onset",
              "elev_m", "slope_dam_deg", "dist_to_glacier_m"]
    glof_years = {("KG-HUN-001", 2022), ("KG-SHM-001", 2022)}
    rows = []
    for n, (lake_id, *_rest) in enumerate(LAKES):
        for year in range(2020, 2026):
            surge = int(lake_id == "KG-HUN-001" and year >= 2021)
            rows.append([
                lake_id, year, int((lake_id, year) in glof_years),
                round(0.05 + 0.04 * n + 0.01 * (year - 2020), 3),
                round(8.0 + 6.0 * n + (22.0 if surge else 0.0), 1),
                round(2.0 + 0.8 * n + (6.0 if surge else 0.0), 1),
                round(30.0 + 25.0 * surge + 2.0 * n, 1),
                round((55.0 if surge else 1.5 * n) - 1.0, 1),
                surge,
                round(-0.6 + 0.05 * n + (0.4 if surge else 0.0), 2),
                int(900 + 40 * n + 15 * (year - 2020)),
                round(0.4 + 0.1 * n + (0.9 if surge else 0.0), 1),
                round(120.0 + 9.0 * n, 1),
                int(140 + 3 * n + (6 if surge else 0)),
                int(3100 + 120 * n),
                round(18.0 + 2.5 * n, 1),
                int(40 + 15 * n),
            ])
    return header, rows


def build_ranking() -> list[list]:
    """risk score per lake per year; rank follows the score within each year."""
    score_by_lake = {
        "KG-HUN-001": {2020: 0.30, 2021: 0.36, 2022: 0.41, 2023: 0.12, 2024: 0.18, 2025: 0.22},
        "KG-SHM-001": {2020: 0.14, 2021: 0.19, 2022: 0.33, 2023: 0.13, 2024: 0.15, 2025: 0.17},
        "KG-HUN-002": {y: round(0.10 + 0.01 * (y - 2020), 2) for y in range(2020, 2026)},
        "KG-HUN-012": {y: round(0.06 + 0.005 * (y - 2020), 3) for y in range(2020, 2026)},
        "KG-SHM-003": {y: round(0.05 + 0.008 * (y - 2020), 3) for y in range(2020, 2026)},
        "KG-SHG-004": {y: round(0.08 + 0.006 * (y - 2020), 3) for y in range(2020, 2026)},
    }
    rows = []
    for year in range(2020, 2026):
        ordered = sorted(score_by_lake, key=lambda lake: score_by_lake[lake][year], reverse=True)
        for rank, lake_id in enumerate(ordered, start=1):
            rows.append([lake_id, year, score_by_lake[lake_id][year], rank])
    return ["lake_id", "year", "risk_score", "rank"], rows


def build_shap_top(ranking_rows: list[list]) -> list[list]:
    """Top 5 drivers per lake-year (SPECS §3), deterministic from the score."""
    templates = [
        ("area_growth_3y", "{v:.0f}", "Lake area grew {t}% over the 3 years before this year"),
        ("surge_flag_prev", "{v:.0f}", "The parent glacier was surging the year before"),
        ("velocity_change", "{v:.1f}", "Glacier flow changed by {t} m/yr before this year"),
        ("heat_anomaly_prev", "{v:.1f}", "Last summer was {t} C warmer than the long-term mean"),
        ("pdd_prev_summer", "{v:.0f}", "Last summer had {t} positive degree-days (melt energy)"),
    ]
    rows = []
    for lake_id, year, score, _rank in ranking_rows:
        for i, (feature, fmt, text) in enumerate(templates):
            shap = round(score * (0.30 - 0.04 * i), 4)
            value = 10.0 + i * 3.0
            rows.append([lake_id, year, feature, fmt.format(v=value),
                         shap, text.format(t=fmt.format(v=value))])
    return ["lake_id", "year", "feature", "value", "shap", "text"], rows


#: Tier 2 replay cases: (case_id, lake_id, event_date or None for a quiet season).
CASES = [
    ("shisper-2022", "KG-HUN-001", "2022-05-07"),
    ("khurdopin-2022", "KG-SHM-001", "2022-07-15"),
    ("passu-2022", "KG-HUN-002", None),
]


def build_cases(alerts: pd.DataFrame) -> list[dict]:
    """Lead time and false alarms per case, computed by the real functions in
    karaglof.rules so metrics.json always agrees with alerts.csv."""
    cases = []
    for case_id, lake_id, event_date in CASES:
        lake_alerts = alerts[alerts["lake_id"] == lake_id]
        lake_events = [e[2] for e in EVENTS if e[1] == lake_id]
        cases.append({
            "case_id": case_id,
            "lake_id": lake_id,
            "event_date": event_date,
            "lead_time_days": lead_time_days(lake_alerts, event_date) if event_date else None,
            "false_alarms": count_false_alarms(lake_alerts, lake_events),
        })
    return cases


def build_metrics(alerts: pd.DataFrame) -> dict:
    th = DEFAULT_THRESHOLDS
    return {
        "data_version": "data-v1",
        "model_version": "model-0.1.0-sample",
        "tier1": {
            "split": {"train": [2000, 2017], "validation": [2018, 2019], "test": [2020, 2025]},
            "test_scored_once": True,
            "no_skill_pr_auc": 0.04,
            "models": [
                {"name": "logistic_baseline", "pr_auc": 0.11, "pr_auc_ci95": [0.05, 0.20],
                 "recall_at_top10": 0.17, "recall_at_top10_ci95": [0.00, 0.45],
                 "brier": 0.041, "brier_ci95": [0.020, 0.070]},
                {"name": "random_forest", "pr_auc": 0.24, "pr_auc_ci95": [0.10, 0.43],
                 "recall_at_top10": 0.33, "recall_at_top10_ci95": [0.10, 0.65],
                 "brier": 0.038, "brier_ci95": [0.019, 0.064]},
                {"name": "lightgbm", "pr_auc": 0.31, "pr_auc_ci95": [0.15, 0.52],
                 "recall_at_top10": 0.50, "recall_at_top10_ci95": [0.22, 0.78],
                 "brier": 0.036, "brier_ci95": [0.018, 0.060]},
            ],
            "ablation": {
                "feature_group_removed": "glacier_surge",
                "features": ["velocity_prev", "velocity_change", "surge_flag_prev", "dh_dt"],
                "pr_auc_without_group": 0.22, "pr_auc_without_group_ci95": [0.09, 0.41],
                "delta_pr_auc": -0.09, "delta_pr_auc_ci95": [-0.19, 0.02],
                "note": "Sample-data placeholder; the real ablation is produced by notebook 04.",
            },
        },
        "tier2": {
            "thresholds": {
                "growth_2w_pct": th.growth_2w_pct,
                "percentile_amber": th.percentile_amber,
                "anomaly_amber_c": th.anomaly_amber_c,
                "percentile_red": th.percentile_red,
                "anomaly_red_c": th.anomaly_red_c,
            },
            "thresholds_frozen_before_replay": True,
            "cases": build_cases(alerts),
        },
    }


def build_lakes_geojson() -> dict:
    features = []
    for lake_id, name, basin, lat, lon, area_ref, dam_type, glacier_id, tier2 in LAKES:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "lake_id": lake_id,
                "name": name or None,
                "basin": basin,
                "area_ref_km2": area_ref,
                "dam_type": dam_type,
                "glacier_id": glacier_id,
                "tier2": tier2,
            },
        })
    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    write_csv("lakes.csv",
              ["lake_id", "name", "basin", "lat", "lon", "area_ref_km2",
               "dam_type", "glacier_id", "tier2"],
              [list(lake) for lake in LAKES])
    write_csv("events.csv",
              ["event_id", "lake_id", "date", "date_certainty", "source", "notes"],
              [list(e) for e in EVENTS])

    header, rows = build_lake_week()
    write_csv("lake_week.csv", header, rows)
    alerts = assign_alert_levels(pd.DataFrame(rows, columns=header))
    write_csv("alerts.csv", ["lake_id", "week_start", "level", "reasons"],
              alerts.values.tolist())

    header, rows = build_lake_area_yearly()
    write_csv("lake_area_yearly.csv", header, rows)
    header, rows = build_lake_year()
    write_csv("lake_year.csv", header, rows)
    header, rows = build_ranking()
    write_csv("ranking.csv", header, rows)
    header, rows = build_shap_top(rows)
    write_csv("shap_top.csv", header, rows)

    (DATA_DIR / "metrics.json").write_text(
        json.dumps(build_metrics(alerts), indent=2) + "\n", encoding="utf-8")
    (DATA_DIR / "lakes.geojson").write_text(
        json.dumps(build_lakes_geojson(), indent=2) + "\n", encoding="utf-8")

    print(f"sample data written to {DATA_DIR}")


if __name__ == "__main__":
    main()
