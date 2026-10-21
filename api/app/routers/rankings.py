"""Ranking endpoint: /rankings."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from .. import schemas
from ..store import DataStore
from .lakes import Year, get_store

router = APIRouter()


@router.get("/rankings", response_model=schemas.RankingResponse)
def get_rankings(
    store: Annotated[DataStore, Depends(get_store)],
    year: Year,
    limit: int = Query(default=10, ge=1, le=100),
    basin: schemas.Basin | None = None,
):
    items = store.ranking_for_year(year, basin=basin, limit=limit)
    return {"data_version": store.data_version, "year": year, "items": items}
