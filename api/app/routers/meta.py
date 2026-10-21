"""Meta endpoints: /health, /metrics, /downloads/{name}."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from .. import schemas
from ..store import DataStore
from .lakes import get_store

router = APIRouter()


@router.get("/health", response_model=schemas.HealthResponse)
def health(store: Annotated[DataStore, Depends(get_store)]):
    return {
        "status": "ok",
        "data_version": store.data_version,
        "model_version": store.model_version,
    }


@router.get("/metrics")
def get_metrics(store: Annotated[DataStore, Depends(get_store)]):
    """Contents of metrics.json: model results, CIs, ablation, versions."""
    return store.metrics


@router.get("/downloads/{name}")
def get_download(name: str, store: Annotated[DataStore, Depends(get_store)]):
    path = store.download_path(name)
    if path is None:
        raise HTTPException(status_code=404, detail="download not found")
    return FileResponse(path, media_type="text/csv")
