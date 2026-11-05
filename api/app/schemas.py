"""Pydantic response models for the KaraGLOF API (SPECS §6).

Field names match the data tables in SPECS §3, and every JSON response
carries ``data_version`` (NFR-2) so the frontend can show which data release
it is displaying.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel

Basin = Literal["hunza", "shimshal", "shigar"]


class LakeOut(BaseModel):
    lake_id: str
    name: str | None
    basin: str
    lat: float
    lon: float
    area_ref_km2: float
    dam_type: str
    glacier_id: str
    tier2: bool


class LakesResponse(BaseModel):
    data_version: str
    lakes: list[LakeOut]


class FeatureCollection(BaseModel):
    """GeoJSON FeatureCollection envelope; ``data_version`` is a foreign
    member (allowed by RFC 7946 §6.1). Feature properties are passed through
    so the map gets exactly what lakes.geojson holds plus risk_score/rank."""

    type: Literal["FeatureCollection"]
    data_version: str
    features: list[dict]


class AreaHistoryPoint(BaseModel):
    year: int
    area_km2: float
    filled: bool


class RiskHistoryPoint(BaseModel):
    year: int
    risk_score: float
    rank: int


class LakeEvent(BaseModel):
    event_id: str
    date: date
    date_certainty: str
    source: str
    notes: str


class LakeDetailResponse(LakeOut):
    data_version: str
    area_history: list[AreaHistoryPoint]
    risk_history: list[RiskHistoryPoint]
    events: list[LakeEvent]


class ExplainItem(BaseModel):
    feature: str
    value: str
    shap: float
    text: str


class ExplainResponse(BaseModel):
    data_version: str
    lake_id: str
    year: int
    items: list[ExplainItem]


class RankingItem(BaseModel):
    rank: int
    lake_id: str
    name: str | None
    risk_score: float


class RankingResponse(BaseModel):
    data_version: str
    year: int
    items: list[RankingItem]


class AlertRow(BaseModel):
    week_start: date
    area_km2: float | None
    t2m_anomaly_c: float | None
    level: str
    reasons: list[str]


class AlertsResponse(BaseModel):
    data_version: str
    lake_id: str
    items: list[AlertRow]


class CaseItem(BaseModel):
    case_id: str
    lake_id: str
    event_date: date | None
    lead_time_days: int | None
    false_alarms: int


class CasesResponse(BaseModel):
    data_version: str
    cases: list[CaseItem]


class HealthResponse(BaseModel):
    status: str
    data_version: str
    model_version: str
