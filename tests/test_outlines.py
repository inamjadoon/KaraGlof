"""Tests for Tier 2 outlines from the lake that drained (karaglof.outlines)."""

import geopandas as gpd
from shapely.geometry import box

from karaglof.outlines import drained_lake

UTM = "EPSG:32643"


def series(*geoms):
    return gpd.GeoSeries(list(geoms), crs=UTM)


def test_picks_the_lake_that_drained_not_the_river():
    lake = box(0, 0, 600, 500)              # 0.30 km², drains
    river = box(1000, 0, 5000, 60)          # long river strip, still there afterwards
    pond = box(-800, 0, -700, 100)          # small permanent pond
    before = series(lake, river, pond)
    drained = series(box(50, 50, 550, 450), box(1200, 0, 1230, 60))   # lake core + river noise
    out = drained_lake(before, drained, min_drained_m2=10_000)
    assert out.equals(lake)


def test_union_of_all_before_parts_touching_the_core():
    west, east = box(0, 0, 300, 300), box(300, 0, 600, 300)
    out = drained_lake(series(west, east), series(box(250, 100, 350, 200)), min_drained_m2=5_000)
    assert abs(out.area - 180_000) < 1


def test_nothing_drained_gives_none():
    before = series(box(0, 0, 500, 500))
    assert drained_lake(before, series(box(0, 0, 20, 20)), min_drained_m2=10_000) is None
    assert drained_lake(before, gpd.GeoSeries([], crs=UTM), min_drained_m2=10_000) is None
