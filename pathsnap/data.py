"""Strict input parsing. Prediction inputs deliberately exclude .route files."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray


def load_trip(folder: Path) -> Trip:
    folder = Path(folder)
    name = folder.name
    track = np.loadtxt(folder / f"{name}.track", ndmin=2)
    nodes = np.loadtxt(folder / f"{name}.nodes", ndmin=2)
    arcs = np.loadtxt(folder / f"{name}.arcs", dtype=np.int64, ndmin=2)
    if track.shape[1] != 3 or nodes.shape[1] != 2 or arcs.shape[1] != 2:
        raise ValueError(f"{name}: invalid column counts")
    if not all(np.isfinite(x).all() for x in (track, nodes)):
        raise ValueError(f"{name}: non-finite coordinates or timestamps")
    for xy in (track[:, :2], nodes):
        if (np.abs(xy[:, 0]) > 180).any() or (np.abs(xy[:, 1]) > 90).any():
            raise ValueError(f"{name}: coordinates outside lon/lat ranges")
    if (np.diff(track[:, 2]) < 0).any():
        raise ValueError(f"{name}: timestamps must be nondecreasing")
    if len(arcs) == 0 or arcs.min() < 0 or arcs.max() >= len(nodes):
        raise ValueError(f"{name}: invalid node indices")
    return Trip(name, track, nodes, arcs)


def project(lonlat, origin):
    """Local equirectangular metres; suitable for local trip road geometries."""
    scale = np.array([np.cos(np.deg2rad(origin[1])), 1.0]) * (6371008.8 * np.pi / 180)
    return (np.asarray(lonlat) - origin) * scale


def geometry(trip):
    origin = np.mean(trip.track[:, :2], axis=0)
    nodes = project(trip.nodes, origin)
    points = project(trip.track[:, :2], origin)
    starts, ends = nodes[trip.arcs[:, 0]], nodes[trip.arcs[:, 1]]
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    return points, starts, vectors, lengths


def segment_projection(points, starts, vectors):
    den = np.maximum(np.sum(vectors * vectors, axis=-1), 1e-12)
    fractions = np.clip(np.sum((points - starts) * vectors, axis=-1) / den, 0, 1)
    snapped = starts + fractions[..., None] * vectors
    distances = np.linalg.norm(points - snapped, axis=-1)
    return distances, fractions, snapped
