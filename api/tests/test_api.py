"""API checks from SPECS §9, run against the sample data in api/data."""

import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
METRICS = json.loads((DATA_DIR / "metrics.json").read_text("utf-8"))
SHISPER = "KG-HUN-001"


def get_ok(client, url):
    r = client.get(url)
    assert r.status_code == 200, r.text
    return r.json()


def test_health(client):
    body = get_ok(client, "/api/v1/health")
    assert body == {"status": "ok", "data_version": METRICS["data_version"],
                    "model_version": METRICS["model_version"]}


@pytest.mark.parametrize("url", [
    "/api/v1/lakes",
    "/api/v1/lakes.geojson?year=2022",
    f"/api/v1/lakes/{SHISPER}",
    f"/api/v1/lakes/{SHISPER}/explain?year=2022",
    "/api/v1/rankings?year=2022",
    f"/api/v1/alerts/{SHISPER}",
    "/api/v1/cases",
    "/api/v1/metrics",
])
def test_every_json_response_carries_data_version(client, url):
    assert get_ok(client, url)["data_version"] == METRICS["data_version"]


def test_lakes_filters(client):
    tier2 = get_ok(client, "/api/v1/lakes?tier2=true")["lakes"]
    assert {lake["name"] for lake in tier2} == {"Shisper", "Passu", "Khurdopin"}
    shigar = get_ok(client, "/api/v1/lakes?basin=shigar")["lakes"]
    assert all(lake["basin"] == "shigar" for lake in shigar)


def test_geojson_has_one_feature_per_lake(client):
    lakes = get_ok(client, "/api/v1/lakes")["lakes"]
    fc = get_ok(client, "/api/v1/lakes.geojson?year=2022")
    assert fc["type"] == "FeatureCollection"
    ids = [f["properties"]["lake_id"] for f in fc["features"]]
    assert sorted(ids) == sorted(lake["lake_id"] for lake in lakes)
    for f in fc["features"]:
        assert f["type"] == "Feature"
        assert f["geometry"]["type"] in {"Point", "Polygon", "MultiPolygon"}
        assert {"risk_score", "rank"} <= f["properties"].keys()


def test_rankings_are_ordered_and_limited(client):
    items = get_ok(client, "/api/v1/rankings?year=2022&limit=3")["items"]
    assert [i["rank"] for i in items] == [1, 2, 3]
    scores = [i["risk_score"] for i in items]
    assert scores == sorted(scores, reverse=True)


def test_lake_detail(client):
    body = get_ok(client, f"/api/v1/lakes/{SHISPER}")
    assert body["name"] == "Shisper"
    assert body["area_history"] and body["risk_history"] and body["events"]


def test_explain_returns_top_five_by_size(client):
    items = get_ok(client, f"/api/v1/lakes/{SHISPER}/explain?year=2022")["items"]
    assert len(items) <= 5
    sizes = [abs(i["shap"]) for i in items]
    assert sizes == sorted(sizes, reverse=True)


def test_alerts_date_filter(client):
    items = get_ok(client, f"/api/v1/alerts/{SHISPER}?from=2022-04-18&to=2022-05-02")["items"]
    assert [i["week_start"] for i in items] == ["2022-04-18", "2022-04-25", "2022-05-02"]
    assert all(i["level"] in {"green", "amber", "red"} for i in items)


def test_cases_match_metrics(client):
    cases = get_ok(client, "/api/v1/cases")["cases"]
    assert cases == METRICS["tier2"]["cases"]


def test_download_csv(client):
    r = client.get("/api/v1/downloads/lakes")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert r.text.startswith("lake_id,")


@pytest.mark.parametrize("url", [
    "/api/v1/lakes/NOPE",
    "/api/v1/lakes/NOPE/explain?year=2022",
    "/api/v1/alerts/NOPE",
    "/api/v1/downloads/secrets",
])
def test_unknown_returns_404(client, url):
    assert client.get(url).status_code == 404


@pytest.mark.parametrize("url", [
    "/api/v1/rankings?year=1990",
    "/api/v1/rankings",
    "/api/v1/lakes.geojson?year=2031",
    f"/api/v1/lakes/{SHISPER}/explain?year=abc",
    "/api/v1/lakes?basin=indus",
    f"/api/v1/alerts/{SHISPER}?from=not-a-date",
])
def test_bad_parameters_return_422(client, url):
    assert client.get(url).status_code == 422
