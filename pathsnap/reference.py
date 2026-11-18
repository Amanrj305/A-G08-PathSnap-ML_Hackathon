"""Training/scoring ONLY: monotone alignment of ordered reference arcs to GPS.

The supplied .route has no GPS timestamps. These are derived proxy labels, not
independently annotated per-observation ground truth. Never import in inference.
"""
from pathlib import Path
import numpy as np
from .data import geometry, segment_projection


LABEL_VERSION = "monotone-distance-v1"


def load_reference(folder, n_arcs):
    folder = Path(folder)
    route = np.loadtxt(folder / f"{folder.name}.route", dtype=np.int64, ndmin=1)
    if not len(route) or route.min() < 0 or route.max() >= n_arcs:
        raise ValueError("Invalid reference arc indices")
    return route


def align_reference(trip, route):
    points, starts, vectors, _ = geometry(trip)
    n, m = len(points), len(route)
    back = np.empty((n, m), dtype=np.int32)
    previous = np.zeros(m, dtype=np.float64)
    columns = np.arange(m)
    for t in range(n):
        distance, _, _ = segment_projection(points[t], starts[route], vectors[route])
        # Robust distance cost. Monotonicity preserves route order and loop visits.
        emission = np.minimum(distance / 15, 1000) ** 2
        prefix = np.minimum.accumulate(previous)
        minima = np.r_[True, previous[1:] < prefix[:-1]]
        argmin = np.maximum.accumulate(np.where(minima, columns, 0))
        back[t] = argmin
        previous = prefix + emission
        previous -= previous.min()
    chosen = np.empty(n, np.int32)
    j = int(previous.argmin())
    for t in range(n - 1, -1, -1):
        chosen[t] = j
        j = int(back[t, j])
    label = route[chosen]
    distance, _, _ = segment_projection(points, starts[label], vectors[label])
    return label.astype(np.int32), distance.astype(np.float32)
