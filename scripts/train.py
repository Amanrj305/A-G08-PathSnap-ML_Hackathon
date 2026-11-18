import argparse, json, time, platform, importlib.metadata
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from pathsnap.data import load_trip
from pathsnap.features import FEATURE_NAMES
from pathsnap.metrics import classification_metrics, top1_metrics, sequence_metrics
from pathsnap.modeling import make_model, SEED, SYLLABUS_MODELS, ALL_MODELS
from pathsnap.reference import load_reference


MODEL_NAMES = SYLLABUS_MODELS


def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log(message):
    print(f'[{now()}] {message}', flush=True)


def training_arrays(cache, k, max_obs, exclude=()):
    xs, ys, groups = [], [], []
    rng = np.random.default_rng(SEED)
    for path in sorted(Path(cache).glob(f'train_*_k{k}.npz')):
        if path.stem.split('_')[1] in set(exclude):
            continue
        with np.load(path) as z:
            n = len(z['X'])
            idx = np.sort(rng.choice(n, min(max_obs, n), replace=False)) if max_obs else np.arange(n)
            xs.append(z['X'][idx])
            ys.append((z['candidates'][idx] == z['reference'][idx,None]).astype(np.uint8))
            groups.extend([path.stem.split('_')[1]] * len(idx))
    return np.concatenate(xs), np.concatenate(ys), np.array(groups)


def cv_score(model, X, y, groups, folds=3):
    rows = []
    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for fold, (train, valid) in enumerate(splitter.split(X, groups=groups)):
        t = time.perf_counter()
        from sklearn.base import clone
        fitted = clone(model).fit(X[train].reshape(-1,X.shape[-1]), y[train].ravel())
        p = fitted.predict_proba(X[valid].reshape(-1,X.shape[-1]))[:,1].reshape(len(valid),-1)
        row = classification_metrics(y[valid], p)
        row.update(fold=fold, top1_accuracy=float(y[valid,np.argmax(p,axis=1)].mean()),
                   seconds=time.perf_counter()-t, train_trips=sorted(set(groups[train])),
                   validation_trips=sorted(set(groups[valid])))
        rows.append(row)
        log(f'  fold {fold+1}/{folds}: top1={row["top1_accuracy"]:.4f} macro_f1={row["macro_f1"]:.4f} ({row["seconds"]:.1f}s)')
    return rows


def evaluate_student(model, cache, data, k, name, output):
    rows, all_y, all_p = [], [], []
    for path in sorted(Path(cache).glob(f'student_test_*_k{k}.npz')):
        trip_id = path.stem.split('_')[2]
        folder = Path(data)/'student_test'/trip_id
        trip = load_trip(folder)
        route = load_reference(folder,len(trip.arcs))
        with np.load(path) as z:
            X, candidates, reference = z['X'], z['candidates'], z['reference']
        t = time.perf_counter()
        p = model.predict_proba(X.reshape(-1,X.shape[-1]))[:,1].reshape(X.shape[:2])
        seconds = time.perf_counter()-t
        y = candidates == reference[:,None]
        pred = candidates[np.arange(len(candidates)),p.argmax(axis=1)]
        row = dict(model=name,trip_id=trip_id,observations=len(X),inference_seconds=seconds,
                   **classification_metrics(y,p), **top1_metrics(candidates,reference,p),
                   **sequence_metrics(trip,pred,reference,route))
        rows.append(row); all_y.append(y.ravel()); all_p.append(p.ravel())
        np.savez_compressed(output/f'{name}_{trip_id}.npz',p=p,predicted=pred)
    metrics = classification_metrics(np.concatenate(all_y),np.concatenate(all_p))
    weights = np.array([r['observations'] for r in rows])
    for key in ['top1_accuracy','candidate_miss_rate','point_accuracy','strict_adjacent_continuity']:
        metrics[key] = float(np.average([r[key] for r in rows],weights=weights))
    for key in ['sampled_sequence_edit_similarity','full_route_edit_similarity','route_arc_f1']:
        metrics[key] = float(np.mean([r[key] for r in rows]))
    metrics['trip_macro_top1'] = float(np.mean([r['top1_accuracy'] for r in rows]))
    metrics['inference_seconds'] = sum(r['inference_seconds'] for r in rows)
    return metrics, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',default='data/Student folder')
    ap.add_argument('--cache',default='cache')
    ap.add_argument('--k',type=int,default=8)
    ap.add_argument('--max-obs-per-trip',type=int,default=600,help='0 uses every observation')
    ap.add_argument('--folds',type=int,default=3)
    ap.add_argument('--models',nargs='+',default=MODEL_NAMES,choices=ALL_MODELS)
    ap.add_argument('--exclude-trips',nargs='*',default=[],
                    help='training trip IDs left out of fitting/CV (e.g. noisy references 00000033 00000054)')
    ap.add_argument('--output',default='results/baseline')
    args = ap.parse_args()
    run_started=time.perf_counter()
    log(f'RUN START  models={args.models}')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    preds=out/'predictions';preds.mkdir(exist_ok=True)
    models=out/'models';models.mkdir(exist_ok=True)
    X,y,groups=training_arrays(args.cache,args.k,args.max_obs_per_trip,args.exclude_trips)
    log(f'Loaded training data: {len(X)} observations from {len(set(groups))} trips'
        + (f' (excluded {args.exclude_trips})' if args.exclude_trips else ''))
    versions={m:importlib.metadata.version(m) for m in ['numpy','scipy','scikit-learn','joblib','pandas']}
    config=dict(vars(args),seed=SEED,observations=len(X),candidate_rows=y.size,
                train_trip_count=len(set(groups)),feature_names=FEATURE_NAMES,versions=versions,
                platform=platform.platform(),python=platform.python_version(),device='CPU',
                primary_selection_metric='mean group-CV observation top-1 accuracy',
                label_warning='Point labels are derived by monotone alignment, not supplied point annotations.')
    (out/'run_config.json').write_text(json.dumps(config,indent=2))
    log(f'Training {len(args.models)} models: {len(X)} observations / {y.size} candidates / {len(set(groups))} trips')
    comparisons, all_trip_rows, cv_records = [],[],{}
    with threadpool_limits(limits=6):
        for i,name in enumerate(args.models,1):
            started_at=now(); model_started=time.perf_counter()
            log(f'MODEL {i}/{len(args.models)} {name}: START (cross-validation)')
            model=make_model(name)
            config_path=out/f'{name}_config.json'
            config_path.write_text(json.dumps(model.get_params(),indent=2,default=str))
            cv=cv_score(model,X,y,groups,args.folds)
            cv_records[name]=cv
            log(f'MODEL {name}: fitting on all training trips')
            t=time.perf_counter()
            model.fit(X.reshape(-1,X.shape[-1]),y.ravel())
            train_seconds=time.perf_counter()-t
            log(f'MODEL {name}: fit done in {train_seconds:.1f}s, scoring student-test trips')
            joblib.dump(model,models/f'{name}.joblib',compress=3)
            metrics,trip_rows=evaluate_student(model,args.cache,args.data,args.k,name,preds)
            train_p=model.predict_proba(X.reshape(-1,X.shape[-1]))[:,1].reshape(y.shape)
            comparison=dict(model=name,cv_macro_f1=float(np.mean([r['macro_f1'] for r in cv])),
                            cv_top1_accuracy=float(np.mean([r['top1_accuracy'] for r in cv])),
                            cv_top1_std=float(np.std([r['top1_accuracy'] for r in cv])),
                            train_top1_accuracy=float(y[np.arange(len(y)),train_p.argmax(axis=1)].mean()),
                            train_seconds=train_seconds,cv_seconds=sum(r['seconds'] for r in cv),**metrics)
            comparison.update(started_at=started_at,finished_at=now(),
                              total_model_seconds=time.perf_counter()-model_started)
            comparisons.append(comparison);all_trip_rows.extend(trip_rows)
            pd.DataFrame(comparisons).to_csv(out/'model_comparison.csv',index=False)
            pd.DataFrame(all_trip_rows).to_csv(out/'per_trip_metrics.csv',index=False)
            (out/'cross_validation.json').write_text(json.dumps(cv_records,indent=2))
            log(f'MODEL {name}: DONE in {comparison["total_model_seconds"]:.1f}s | STUDENT TEST top1={metrics["top1_accuracy"]:.4f} '
                f'binary_accuracy={metrics["accuracy"]:.4f} precision={metrics["precision"]:.4f} recall={metrics["recall"]:.4f}')
    best=max(comparisons,key=lambda r:r['cv_top1_accuracy'])
    artifacts=Path('artifacts');artifacts.mkdir(exist_ok=True)
    selected=joblib.load(models/f'{best["model"]}.joblib')
    joblib.dump(selected,artifacts/'model.joblib',compress=3)
    metadata=dict(model_name=best['model'],k=args.k,feature_names=FEATURE_NAMES,seed=SEED,
                  metrics=best,versions=versions,stage='baseline',label_version='monotone-distance-v1')
    (artifacts/'model_metadata.json').write_text(json.dumps(metadata,indent=2))
    (out/'selection.json').write_text(json.dumps(metadata,indent=2))
    log(f'SELECTED {best["model"]} (cv_top1={best["cv_top1_accuracy"]:.4f}, test_top1={best["top1_accuracy"]:.4f})')
    log(f'RUN END  total {time.perf_counter()-run_started:.1f}s')


if __name__=='__main__':main()
