"""Audit all supplied trips and cache deterministic label-free features + proxy labels."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pathsnap.data import load_trip, geometry
from pathsnap.features import candidate_features, FEATURE_NAMES
from pathsnap.reference import load_reference, align_reference, LABEL_VERSION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/Student folder')
    ap.add_argument('--cache', default='cache')
    ap.add_argument('--k', type=int, default=8)
    args = ap.parse_args()
    out = Path(args.cache); out.mkdir(parents=True, exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    inventory, hashes = [], {}
    started = time.perf_counter()
    for partition in ['train', 'student_test']:
        folders = sorted((Path(args.data)/partition).iterdir())
        for folder in folders:
            if not folder.is_dir(): continue
            trip = load_trip(folder)
            route = load_reference(folder, len(trip.arcs))
            dest = out / f'{partition}_{trip.trip_id}_k{args.k}.npz'
            if not dest.exists():
                f = candidate_features(trip, args.k)
                reference, error = align_reference(trip, route)
                np.savez_compressed(dest, **f, reference=reference, reference_distance=error,
                                    label_version=LABEL_VERSION)
            with np.load(dest) as z:
                miss = float(np.mean(~(z['candidates'] == z['reference'][:,None]).any(axis=1)))
                q = np.percentile(z['reference_distance'], [50,95,99]).tolist()
            g = coo_matrix((np.ones(len(trip.arcs)), trip.arcs.T), shape=(len(trip.nodes),)*2).tocsr()
            components = connected_components(g, directed=True, connection='weak', return_labels=False)
            _, _, _, lengths = geometry(trip)
            dt = np.diff(trip.track[:,2])
            row = dict(partition=partition, trip_id=trip.trip_id, observations=len(trip.track),
                       nodes=len(trip.nodes), arcs=len(trip.arcs), route_arcs=len(route),
                       weak_components=int(components), isolated_nodes=int(np.sum(np.bincount(trip.arcs.ravel(),minlength=len(trip.nodes))==0)),
                       missing_values=0, duplicate_timestamps=int(np.sum(dt==0)),
                       max_gap_s=float(dt.max()) if len(dt) else 0,
                       gaps_over_10s=int(np.sum(dt>10)), nominal_1s_fraction=float(np.mean(dt==1)),
                       lon_min=float(trip.track[:,0].min()), lon_max=float(trip.track[:,0].max()),
                       lat_min=float(trip.track[:,1].min()), lat_max=float(trip.track[:,1].max()),
                       zero_length_arcs=int(np.sum(lengths<1e-6)),
                       reference_route_continuity=float(np.mean(trip.arcs[route[:-1],1]==trip.arcs[route[1:],0])),
                       reference_distance_p50=q[0], reference_distance_p95=q[1],reference_distance_p99=q[2],
                       candidate_miss_rate=miss)
            inventory.append(row)
            for p in sorted(folder.iterdir()):
                hashes[str(p.relative_to(args.data)).replace('\\','/')] = hashlib.sha256(p.read_bytes()).hexdigest()
            print(f'{partition}/{trip.trip_id}: {len(trip.track)} observations, miss={miss:.3%}', flush=True)
    pd.DataFrame(inventory).to_csv('results/dataset_inventory.csv', index=False)
    meta = dict(k=args.k, feature_names=FEATURE_NAMES, label_version=LABEL_VERSION,
                preparation_seconds=time.perf_counter()-started, source_sha256=hashes)
    Path('results/data_manifest.json').write_text(json.dumps(meta,indent=2))
    print('Prepared',len(inventory),'trips in',round(meta['preparation_seconds'],1),'seconds',flush=True)


if __name__ == '__main__': main()
