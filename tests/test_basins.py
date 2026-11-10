"""Tests for basin delineation (karaglof.basins)."""

import pytest

from karaglof.basins import children_map, delineate, upstream

# Toy river network (arrows point downstream, 0 = sea):
#
#   1 -> 3 -> 5 -> 6 -> 0     main river
#   2 -> 3                    tributary joining at 3
#   4 -> 5                    "neighbour valley" joining at 5
NEXT_DOWN = {1: 3, 2: 3, 3: 5, 4: 5, 5: 6, 6: 0}


def test_upstream_collects_everything_draining_in():
    children = children_map(NEXT_DOWN)
    assert upstream(children, 3) == {1, 2, 3}
    assert upstream(children, 6) == {1, 2, 3, 4, 5, 6}
    assert upstream(children, 4) == {4}


def test_delineate_stops_before_the_excluded_valley():
    outlet, area = delineate(NEXT_DOWN, start=1, exclude={4})
    assert outlet == 3
    assert area == {1, 2, 3}


def test_delineate_without_exclusion_reaches_the_sea():
    outlet, area = delineate(NEXT_DOWN, start=2, exclude=set())
    assert outlet == 6
    assert len(area) == 6


def test_delineate_rejects_inconsistent_anchors():
    with pytest.raises(ValueError):
        delineate(NEXT_DOWN, start=5, exclude={4})
