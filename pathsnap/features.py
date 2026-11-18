"""Label-free exact nearest-segment candidates and reusable local features."""
import numpy as np
from scipy.spatial import cKDTree
from .data import geometry, segment_projection

FEATURE_NAMES = [
    "log_distance_m", "distance_m_clipped", "distance_rank", "distance_gap_m",
    "heading_cos_5", "heading_cos_15", "motion_5_m", "speed_5_mps",
    "log_segment_length_m", "projection_fraction", "log_distance_start_m",
    "log_distance_end_m", "forward_speed_mps", "log_prev_segment_distance",
    "log_next_segment_distance", "start_outdegree", "end_outdegree", "log_dt",
]


def candidate_features(trip, k=8, spacing=40.0):
    points, starts, vectors, lengths = geometry(trip)
    if len(lengths) < k:
        raise ValueError(f"Need at least {k} road arcs")
    # Every segment is sampled at <= spacing metres, including both endpoints.
    intervals = np.maximum(1, np.ceil(lengths / spacing).astype(int))
    counts = intervals + 1
    ids = np.repeat(np.arange(len(lengths)), counts)
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    fractions = (np.arange(len(ids)) - offsets) / intervals[ids]
    samples = starts[ids] + fractions[:, None] * vectors[ids]
    tree = cKDTree(samples)
    _, initial = tree.query(points, k=min(max(48, k * 6), len(samples)), workers=4)
    candidates = np.empty((len(points), k), np.int32)
    for t, point in enumerate(points):
        pool = np.unique(ids[initial[t]])
        if len(pool) < k:
            pool = np.arange(len(lengths))
        dist, _, _ = segment_projection(point, starts[pool], vectors[pool])
        kth = np.partition(dist, k - 1)[k - 1]
        # Triangle inequality: any better segment has a sample <= kth+spacing/2.
        nearby = tree.query_ball_point(point, float(kth + spacing / 2 + 1e-6))
        pool = np.unique(ids[nearby])
        dist, _, _ = segment_projection(point, starts[pool], vectors[pool])
        order = np.lexsort((pool, dist))[:k]
        candidates[t] = pool[order]
    s, v = starts[candidates], vectors[candidates]
    dist, frac, snapped = segment_projection(points[:, None, :], s, v)
    length = lengths[candidates]
    unit = v / np.maximum(length[..., None], 1e-6)
    n = len(points)
    index = np.arange(n)
    def motion(window):
        left, right = np.maximum(index - window, 0), np.minimum(index + window, n - 1)
        d = points[right] - points[left]
        mag = np.linalg.norm(d, axis=1)
        dt = np.maximum(trip.track[right, 2] - trip.track[left, 2], 1)
        cos = np.sum(unit * (d / np.maximum(mag[:, None], 1e-6))[:, None, :], axis=-1)
        return cos, mag, mag / dt, d / dt[:, None]
    cos5, mag5, speed5, velocity = motion(5)
    cos15, _, _, _ = motion(15)
    prev = points[np.maximum(index - 3, 0), None, :]
    nxt = points[np.minimum(index + 3, n - 1), None, :]
    pd, _, _ = segment_projection(prev, s, v)
    nd, _, _ = segment_projection(nxt, s, v)
    degrees = np.bincount(trip.arcs[:, 0], minlength=len(trip.nodes))
    dt = np.r_[1., np.maximum(np.diff(trip.track[:, 2]), 0)]
    broadcast = lambda a: np.broadcast_to(a[:, None], dist.shape)
    features = np.stack([
        np.log1p(dist), np.minimum(dist, 500),
        np.broadcast_to(np.arange(k) / max(k - 1, 1), dist.shape),
        np.minimum(dist - dist[:, :1], 300), cos5, cos15,
        broadcast(np.minimum(mag5, 1000)), broadcast(np.minimum(speed5, 100)),
        np.log1p(length), frac, np.log1p(np.linalg.norm(points[:, None] - s, axis=-1)),
        np.log1p(np.linalg.norm(points[:, None] - (s + v), axis=-1)),
        np.clip(np.sum(unit * velocity[:, None], axis=-1), -100, 100),
        np.log1p(pd), np.log1p(nd),
        degrees[trip.arcs[candidates, 0]], degrees[trip.arcs[candidates, 1]],
        broadcast(np.log1p(dt)),
    ], axis=-1).astype(np.float32)
    return {"X": features, "candidates": candidates, "fractions": frac.astype(np.float32),
            "distances": dist.astype(np.float32)}
