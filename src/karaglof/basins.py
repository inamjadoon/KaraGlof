"""Basin delineation from HydroBASINS topology (notebook 01).

HydroBASINS splits the land into small sub-basins. Each one has an ID
(HYBAS_ID) and the ID of the sub-basin it drains into (NEXT_DOWN; 0 means it
drains to the sea or a sink). That is enough to rebuild any river basin
without drawing it by hand:

1. Start at a sub-basin that is surely inside the target basin (for example
   the one containing a known glacier).
2. Walk downstream along NEXT_DOWN. At every step the upstream area grows.
3. Stop just before the upstream area would swallow an "exclude" sub-basin
   that is surely *outside* the target (for example a glacier in the
   neighbouring valley). The last upstream area before that is the basin.

Every rule here is plain Python, so it is unit-tested without Earth Engine.
"""

from __future__ import annotations

from collections import defaultdict


def children_map(next_down: dict[int, int]) -> dict[int, list[int]]:
    """Invert NEXT_DOWN: for every sub-basin, the sub-basins draining into it."""
    children: dict[int, list[int]] = defaultdict(list)
    for basin, down in next_down.items():
        if down:
            children[down].append(basin)
    return children


def upstream(children: dict[int, list[int]], outlet: int) -> set[int]:
    """The outlet sub-basin plus everything that drains into it."""
    seen = {outlet}
    stack = [outlet]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def delineate(next_down: dict[int, int], start: int, exclude: set[int]) -> tuple[int, set[int]]:
    """Walk downstream from `start` and return (outlet, upstream set) of the
    last sub-basin whose upstream area contains none of `exclude`.

    Raises ValueError if `start` itself is upstream of an excluded sub-basin
    (the anchors are inconsistent).
    """
    children = children_map(next_down)
    best: tuple[int, set[int]] | None = None
    current = start
    while current:
        area = upstream(children, current)
        if area & exclude:
            break
        best = (current, area)
        current = next_down.get(current, 0)
    if best is None:
        raise ValueError(f"start basin {start} already drains an excluded basin")
    return best
