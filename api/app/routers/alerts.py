"""Alert endpoints: /alerts/{lake_id}, /cases."""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from .. import schemas
from ..store import DataStore
from .lakes import get_store

router = APIRouter()


@router.get("/alerts/{lake_id}", response_model=schemas.AlertsResponse)
def get_alerts(
    lake_id: str,
    store: Annotated[DataStore, Depends(get_store)],
    from_date: Annotated[date | None, Query(alias="from")] = None,
    to_date: Annotated[date | None, Query(alias="to")] = None,
):
    # SPECS §6 names the parameters `from` and `to`; `from` is a Python
    # keyword, so the function arguments use aliases.
    if store.get_lake(lake_id) is None:
        raise HTTPException(status_code=404, detail="lake not found")
    items = store.alerts_for_lake(lake_id, from_date, to_date)
    return {"data_version": store.data_version, "lake_id": lake_id, "items": items}


@router.get("/cases", response_model=schemas.CasesResponse)
def get_cases(store: Annotated[DataStore, Depends(get_store)]):
    return {"data_version": store.data_version, "cases": store.cases}
