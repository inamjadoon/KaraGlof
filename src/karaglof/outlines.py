"""Tier 2 lake outlines from "the lake that drained" (notebook 01, step 3b).

Why: the global Landsat water layer (JRC GSW) misses ice-dammed lakes such as
Shisper. They are often ice-covered and exist only for months before draining.
Sentinel-2 sees them, but a plain water map also contains rivers and
permanent lakes.

Physics gives a clean selector. Around a dated outburst, water that is present
in the weeks **before** the drainage and gone in the weeks **after** it is the
lake that drained; rivers and permanent lakes are present in both and cancel
out. The lake outline is then the full pre-outburst water body that contains
the drained water.

Inputs are polygons in a metric CRS (UTM), so areas are in m². Pure shapely
and geopandas, so the rule is unit-tested without Earth Engine.
"""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry.base import BaseGeometry


def drained_lake(before: gpd.GeoSeries, drained: gpd.GeoSeries,
                 min_drained_m2: float) -> BaseGeometry | None:
    """Pre-outburst outline of the lake that drained.

    `before`: water polygons in the window before the drainage.
    `drained`: polygons of water present before and absent after.
    Takes the largest drained patch (at least `min_drained_m2`) and returns
    the union of the `before` polygons that touch it, or None if nothing
    drained.
    """
    big = drained[drained.area >= min_drained_m2]
    if big.empty:
        return None
    core = big.loc[big.area.idxmax()]
    parts = before[before.intersects(core)]
    if parts.empty:
        return None
    return parts.union_all()
