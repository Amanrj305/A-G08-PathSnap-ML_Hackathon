"""Grid-search HMM decoder parameters on STUDENT-TEST trips only (allowed by the
Models & Training guidelines section 9). Evaluator trips are never read here.

Run from a run folder containing artifacts/model.joblib. Writes
results/decoder_tuning.csv and, if better than the default, artifacts/decoder_config.json.

Searched settings:
  beta_m             transition scale: how strongly a route whose road distance differs
                     from the GPS distance is penalised (smaller = stricter connectivity)
  emission_weight    how much the classifier probability counts versus the transitions
  backward_jitter_m  how far GPS may drift backwards on the same arc (stationary noise)
  realign            re-assign points to the nearest arc along the DECODED path
"""
import argparse, itertools, json, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
from pathsnap.data import load_trip
from pathsnap.decoder import viterbi, DEFAULT_CONFIG, Router
from scripts.train import log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/Student folder')
    ap.add_argument('--cache', default='cache')
    ap.add_argument('--beta', nargs='+', type=float, default=[2, 3, 5, 15])
    ap.add_argument('--emission-weight', nargs='+', type=float, default=[0.25, 0.5, 1])
    ap.add_argument('--jitter', nargs='+', type=float, default=[15, 25, 40])
    ap.add_argument('--realign', nargs='+', type=int, default=[0, 1], help='0 = off, 1 = on')
    ap.add_argument('--gap-reset', nargs='+', type=float, default=[60])
    args = ap.parse_args()
    meta = json.loads(Path('artifacts/model_metadata.json').read_text())
    model = joblib.load('artifacts/model.joblib')
    Path('results').mkdir(exist_ok=True)
    log(f'DECODER TUNING START model={meta["model_name"]} (student-test trips only)')
    trips = []
    for path in sorted(Path(args.cache).glob(f'student_test_*_k{meta["k"]}.npz')):
        trip_id = path.stem.split('_')[2]
        trip = load_trip(Path(args.data) / 'student_test' / trip_id)
        with np.load(path) as z:
            f = {k: z[k] for k in ['X', 'candidates', 'fractions']}; reference = z['reference']
        with threadpool_limits(limits=6):
            p = model.predict_proba(f['X'].reshape(-1, f['X'].shape[-1]))[:, 1].reshape(f['X'].shape[:2])
        trips.append((trip_id, trip, f, p, reference, Router(trip, DEFAULT_CONFIG['routing_limit_m'])))
    log(f'Scored {len(trips)} student-test trips with the classifier')

    def evaluate(config):
        correct = total = 0; per_trip = {}
        for trip_id, trip, f, p, reference, router in trips:
            pred = viterbi(trip, f, p, config, router=router)['predicted']
            per_trip[trip_id] = float(np.mean(pred == reference))
            correct += int(np.sum(pred == reference)); total += len(reference)
        return correct / total, per_trip

    rows = []
    grid = [dict(DEFAULT_CONFIG)] + [
        dict(DEFAULT_CONFIG, beta_m=b, emission_weight=e, backward_jitter_m=j, realign=bool(r), gap_reset_s=g)
        for b, e, j, r, g in itertools.product(args.beta, args.emission_weight, args.jitter, args.realign, args.gap_reset)]
    for i, config in enumerate(grid, 1):
        t = time.perf_counter()
        acc, per_trip = evaluate(config)
        row = dict(label='DEFAULT' if i == 1 else f'config_{i-1}', beta_m=config['beta_m'],
                   emission_weight=config['emission_weight'], backward_jitter_m=config['backward_jitter_m'],
                   realign=config['realign'], gap_reset_s=config['gap_reset_s'], point_accuracy=acc,
                   trip_mean_accuracy=float(np.mean(list(per_trip.values()))),
                   worst_trip=min(per_trip, key=per_trip.get), worst_trip_accuracy=min(per_trip.values()),
                   seconds=time.perf_counter() - t, **{f'trip_{k}': v for k, v in per_trip.items()})
        rows.append(row)
        log(f'CONFIG {i}/{len(grid)} {row["label"]} beta={row["beta_m"]} emission_weight={row["emission_weight"]} '
            f'jitter={row["backward_jitter_m"]} realign={row["realign"]}: point_acc={acc:.4f} '
            f'trip_mean={row["trip_mean_accuracy"]:.4f}')
        pd.DataFrame(rows).to_csv('results/decoder_tuning.csv', index=False)
    table = pd.DataFrame(rows)
    default_acc = float(table.point_accuracy.iloc[0])
    best = table.sort_values('point_accuracy', ascending=False, kind='stable').iloc[0]
    log(f'DEFAULT point_acc={default_acc:.4f} | BEST point_acc={best.point_accuracy:.4f} with beta={best.beta_m} '
        f'emission_weight={best.emission_weight} jitter={best.backward_jitter_m} realign={best.realign}')
    if best.point_accuracy > default_acc:
        config = dict(DEFAULT_CONFIG, beta_m=float(best.beta_m), emission_weight=float(best.emission_weight),
                      backward_jitter_m=float(best.backward_jitter_m), realign=bool(best.realign),
                      gap_reset_s=float(best.gap_reset_s))
        Path('artifacts').mkdir(exist_ok=True)
        Path('artifacts/decoder_config.json').write_text(json.dumps(config, indent=2))
        log('Saved best decoder settings to artifacts/decoder_config.json')
    else:
        log('Default decoder settings kept (no improvement)')


if __name__ == '__main__':
    main()
