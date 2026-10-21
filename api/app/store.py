"""In-memory data store for the KaraGLOF API.

The API never computes model or satellite results: at startup it loads the
released files from ``api/data`` (SPECS §3) into memory and serves them
read-only. The only derivations here are joins a store must do: attaching
names to ranking rows, merging alert levels with weekly observations, and
lead-time lookups between events and alerts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RELEASED_FILES = [
    "lakes.csv",
    "events.csv",
    "lake_area_yearly.csv",
    "lake_year.csv",
    "lake_week.csv",
    "ranking.csv",
    "shap_top.csv",
    "alerts.csv",
    "metrics.json",
    "lakes.geojson",
]

#: Downloads served under /api/v1/downloads/{name} (SPECS §6).
DOWNLOADS = {
    "lakes": "lakes.csv",
    "events": "events.csv",
    "lake_year": "lake_year.csv",
    "ranking": "ranking.csv",
}


@dataclass
class Case:
    case_id: str
    lake_id: str
    event_date: str | None
    lead_time_days: int | None


class DataStore:
    """Everything the API serves, loaded once at startup."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

        self.lakes = pd.read_csv(data_dir / "lakes.csv", keep_default_na=False)
        self.events = pd.read_csv(data_dir / "events.csv", keep_default_na=False)
        self.area_yearly = pd.read_csv(data_dir / "lake_area_yearly.csv")
        self.lake_year = pd.read_csv(data_dir / "lake_year.csv")
        self.ranking = pd.read_csv(data_dir / "ranking.csv")
        self.shap_top = pd.read_csv(data_dir / "shap_top.csv", keep_default_na=False)

        alerts = pd.read_csv(data_dir / "alerts.csv", keep_default_na=False)
        alerts["reasons"] = alerts["reasons"].map(
            lambda s: [r for r in s.split("; ") if r]
        )
        self.alerts = alerts

        week = pd.read_csv(data_dir / "lake_week.csv")
        self.week_by_key = {
            (r.lake_id, r.week_start): r
            for r in week.itertuples(index=False)
        }

        self.metrics = json.loads((data_dir / "metrics.json").read_text("utf-8"))
        self.geojson = json.loads((data_dir / "lakes.geojson").read_text("utf-8"))

        self._lakes_by_id = {r.lake_id: r for r in self.lakes.itertuples(index=False)}
        self.ranking_years = sorted(self.ranking["year"].unique().tolist())
        self.cases = self._build_cases()

    @classmethod
    def load(cls, data_dir: Path) -> "DataStore":
        missing = [f for f in RELEASED_FILES if not (data_dir / f).is_file()]
        if missing:
            msg = f"released files missing from {data_dir}: {', '.join(missing)}"
            raise RuntimeError(msg)
        return cls(data_dir)

    @property
    def data_version(self) -> str:
        return str(self.metrics["data_version"])

    @property
    def model_version(self) -> str:
        return str(self.metrics["model_version"])

    def get_lake(self, lake_id: str):
        return self._lakes_by_id.get(lake_id)

    def list_lakes(self, basin: str | None, tier2: bool | None):
        df = self.lakes
        if basin is not None:
            df = df[df["basin"] == basin]
        if tier2 is not None:
            df = df[df["tier2"] == tier2]
        return [self._lakes_by_id[i] for i in df["lake_id"]]

    def ranking_for_year(self, year: int, basin: str | None, limit: int):
        df = self.ranking[self.ranking["year"] == year].sort_values("rank")
        rows = []
        for r in df.itertuples(index=False):
            lake = self._lakes_by_id.get(r.lake_id)
            if basin is not None and (lake is None or lake.basin != basin):
                continue
            rows.append({"rank": int(r.rank), "lake_id": r.lake_id,
                         "name": self.lake_name(r.lake_id),
                         "risk_score": float(r.risk_score)})
            if len(rows) >= limit:
                break
        return rows

    def lake_name(self, lake_id: str) -> str | None:
        lake = self._lakes_by_id.get(lake_id)
        return lake.name if lake is not None and lake.name else None

    def geojson_for_year(self, year: int) -> dict:
        rank_by_lake = {
            r.lake_id: r
            for r in self.ranking[self.ranking["year"] == year].itertuples(index=False)
        }
        features = []
        for f in self.geojson["features"]:
            props = dict(f["properties"])
            row = rank_by_lake.get(props["lake_id"])
            props["risk_score"] = float(row.risk_score) if row is not None else None
            props["rank"] = int(row.rank) if row is not None else None
            features.append({"type": "Feature", "geometry": f["geometry"],
                             "properties": props})
        return {"type": "FeatureCollection", "features": features}

    def lake_detail(self, lake_id: str) -> dict | None:
        lake = self.get_lake(lake_id)
        if lake is None:
            return None
        area = self.area_yearly[self.area_yearly["lake_id"] == lake_id]
        risk = self.ranking[self.ranking["lake_id"] == lake_id]
        events = self.events[self.events["lake_id"] == lake_id]
        return {
            "area_history": [
                {"year": int(r.year), "area_km2": float(r.area_km2),
                 "filled": bool(r.filled)}
                for r in area.sort_values("year").itertuples(index=False)
            ],
            "risk_history": [
                {"year": int(r.year), "risk_score": float(r.risk_score),
                 "rank": int(r.rank)}
                for r in risk.sort_values("year").itertuples(index=False)
            ],
            "events": [
                {"event_id": r.event_id, "date": r.date,
                 "date_certainty": r.date_certainty, "source": r.source,
                 "notes": r.notes}
                for r in events.sort_values("date").itertuples(index=False)
            ],
        }

    def explain(self, lake_id: str, year: int) -> list[dict]:
        df = self.shap_top[
            (self.shap_top["lake_id"] == lake_id) & (self.shap_top["year"] == year)
        ]
        df = df.sort_values("shap", ascending=False).head(5)
        return [
            {"feature": r.feature, "value": str(r.value), "shap": float(r.shap),
             "text": r.text}
            for r in df.itertuples(index=False)
        ]

    def alerts_for_lake(self, lake_id: str,
                        from_date: str | None, to_date: str | None) -> list[dict]:
        df = self.alerts[self.alerts["lake_id"] == lake_id]
        rows = []
        for r in df.sort_values("week_start").itertuples(index=False):
            if from_date is not None and r.week_start < from_date:
                continue
            if to_date is not None and r.week_start > to_date:
                continue
            week = self.week_by_key.get((lake_id, r.week_start), None)
            rows.append({
                "week_start": r.week_start,
                "area_km2": float(week.area_km2_s1) if week is not None else None,
                "t2m_anomaly_c": float(week.t2m_anomaly_c) if week is not None else None,
                "level": r.level,
                "reasons": list(r.reasons),
            })
        return rows

    def _build_cases(self) -> list[Case]:
        cases = []
        for e in self.events.itertuples(index=False):
            lake = self._lakes_by_id.get(e.lake_id)
            slug = (lake.name if lake is not None and lake.name
                    else e.lake_id.lower())
            year = e.date[:4]
            lead = None
            alerts = self.alerts[
                (self.alerts["lake_id"] == e.lake_id)
                & (self.alerts["level"].isin(["amber", "red"]))
                & (self.alerts["week_start"] <= e.date)
            ]
            if not alerts.empty:
                lead = (pd.Timestamp(e.date)
                        - pd.Timestamp(alerts["week_start"].min())).days
            cases.append(Case(case_id=f"{slug.lower()}-{year}", lake_id=e.lake_id,
                              event_date=e.date, lead_time_days=lead))
        return cases

    def download_path(self, name: str) -> Path | None:
        filename = DOWNLOADS.get(name)
        if filename is None:
            return None
        return self.data_dir / filename
