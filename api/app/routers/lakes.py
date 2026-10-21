"""Lake endpoints: /lakes, /lakes.geojson, /lakes/{lake_id}, /lakes/{id}/explain."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from .. import schemas
from ..store import DataStore

router = APIRouter()


def get_store(request: Request) -> DataStore:
    return request.app.state.store


Year = Annotated[int, Query(ge=2000, le=2025)]


def _lake_or_404(store: DataStore, lake_id: str):
    lake = store.get_lake(lake_id)
    if lake is None:
        raise HTTPException(status_code=404, detail="lake not found")
    return lake


def _lake_payload(lake) -> dict:
    return {**lake._asdict(), "name": lake.name or None}


@router.get("/lakes", response_model=schemas.LakesResponse)
def list_lakes(
    store: Annotated[DataStore, Depends(get_store)],
    basin: schemas.Basin | None = None,
    tier2: bool | None = None,
):
    lakes = [_lake_payload(l) for l in store.list_lakes(basin=basin, tier2=tier2)]
    return {"data_version": store.data_version, "lakes": lakes}


@router.get("/lakes.geojson", response_model=schemas.FeatureCollection)
def lakes_geojson(
    store: Annotated[DataStore, Depends(get_store)],
    year: int | None = None,
):
    year = year if year is not None else store.ranking_years[-1]
    if year < 2000 or year > 2025:
        raise HTTPException(status_code=422, detail="year out of range 2000-2025")
    return {**store.geojson_for_year(year), "data_version": store.data_version}


@router.get("/lakes/{lake_id}", response_model=schemas.LakeDetailResponse)
def lake_detail(lake_id: str, store: Annotated[DataStore, Depends(get_store)]):
    lake = _lake_or_404(store, lake_id)
    detail = store.lake_detail(lake_id)
    return {**_lake_payload(lake), "data_version": store.data_version, **detail}


@router.get("/lakes/{lake_id}/explain", response_model=schemas.ExplainResponse)
def explain_lake(
    lake_id: str,
    store: Annotated[DataStore, Depends(get_store)],
    year: Year,
):
    _lake_or_404(store, lake_id)
    return {
        "data_version": store.data_version,
        "lake_id": lake_id,
        "year": year,
        "items": store.explain(lake_id, year),
    }
