"""KaraGLOF read-only API (SPECS §6).

Run locally from the repository root:

    uvicorn app.main:app --app-dir api --reload

At startup the released files in ``api/data`` are loaded into memory once
(``DataStore``); every route then only reads from that store. There is no
database and no write endpoint. Environment variables:

    KARAGLOF_DATA_DIR  folder with the released files (default: api/data)
    FRONTEND_ORIGIN    comma-separated origins allowed by CORS
                       (default: the Vite dev server, http://localhost:5173)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import alerts, lakes, meta, rankings
from .store import DataStore

DATA_DIR = Path(
    os.environ.get("KARAGLOF_DATA_DIR", Path(__file__).resolve().parents[1] / "data")
)
FRONTEND_ORIGINS = [
    o.strip()
    for o in os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173").split(",")
    if o.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fails fast with a clear message if any released file is missing.
    app.state.store = DataStore.load(DATA_DIR)
    yield


app = FastAPI(
    title="KaraGLOF API",
    version="0.5.0",
    description=(
        "Read-only API for KaraGLOF: yearly GLOF susceptibility rankings and weekly "
        "warning signs for the Hunza, Shimshal and Shigar basins. "
        "Research prototype — not an official warning."
    ),
    lifespan=lifespan,
)

# CORS allows the frontend origin only (SPECS §6); the API is GET-only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

for module in (meta, lakes, rankings, alerts):
    app.include_router(module.router, prefix="/api/v1")
