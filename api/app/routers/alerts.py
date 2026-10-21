"""Alert endpoints: /alerts/{lake_id}, /cases."""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from .. import schemas
from ..store import DataStore
from .lakes import get_store

router = APIRouter()


@router.get("/alerts/{lake_id}", response_model=schemas.AlertsResponse)
def get_alerts(
    lake_id: str,
    store: Annotated[DataStore, Depends(get_store)],
    from_date: date | None = None,
    to_date: date | None = None,
):
    if store.get_lake(lake_id) is None:
        raise HTTPException(status_code=404, detail="lake not found")
    items = store.alerts_for_lake(lake_id, from_date, to_date)
    return {"data_version": store.data_version, "lake_id": lake_id, "items": items}


@router.get("/cases", response_model=schemas.CasesResponse)
def get_cases(store: Annotated[DataStore, Depends(get_store)]):
    cases = [c.__dict__ for c in store.cases]
    return {"data_version": store.data_version, "cases": cases}
