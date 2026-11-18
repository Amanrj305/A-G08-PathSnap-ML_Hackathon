"""Retrain the same selected syllabus model into a NEW experiment directory."""
import argparse,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from pathsnap.modeling import make_model
from scripts.train import training_arrays,cv_score,evaluate_student,log


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-artifacts',default='artifacts')
    ap.add_argument('--output',required=True)
    ap.add_argument('--max-obs-per-trip',type=int,default=0)
    ap.add_argument('--folds',type=int,default=5)
    ap.add_argument('--params',help='JSON file of hyperparameter overrides')
    ap.add_argument('--data',default='data/Student folder')
    ap.add_argument('--cache',default='cache')
    args=ap.parse_args();out=Path(args.output)
    if out.exists() and any(out.iterdir()):raise SystemExit('Choose a new/empty output directory to preserve experiments.')
    out.mkdir(parents=True,exist_ok=True);(out/'predictions').mkdir(exist_ok=True)
    meta=json.loads((Path(args.source_artifacts)/'model_metadata.json').read_text())
    name=meta['model_name'].removesuffix('_tuned')
    params=meta.get('best_params',{}).copy()
    if args.params:params.update(json.loads(Path(args.params).read_text()))
    model=make_model(name,params);X,y,groups=training_arrays(args.cache,meta['k'],args.max_obs_per_trip)
    start=time.perf_counter()
    log(f'RETRAIN START {name} params={params} observations={len(X)}')
    with threadpool_limits(limits=6):
        folds=cv_score(model,X,y,groups,args.folds)
        log(f'Fitting {name} on all training trips')
        model.fit(X.reshape(-1,X.shape[-1]),y.ravel())
        log('Fit done, scoring student-test trips')
        metrics,rows=evaluate_student(model,args.cache,args.data,meta['k'],name,out/'predictions')
    log(f'RETRAIN END {name}: test_top1={metrics["top1_accuracy"]:.4f} total={time.perf_counter()-start:.1f}s')
    joblib.dump(model,out/'model.joblib',compress=3)
    summary=dict(model=name,params=params,cv_top1_accuracy=float(np.mean([r['top1_accuracy'] for r in folds])),
                 cv_macro_f1=float(np.mean([r['macro_f1'] for r in folds])),observations=len(X),
                 candidate_rows=y.size,seconds=time.perf_counter()-start,**metrics)
    (out/'metrics.json').write_text(json.dumps(summary,indent=2))
    (out/'folds.json').write_text(json.dumps(folds,indent=2));pd.DataFrame(rows).to_csv(out/'per_trip_metrics.csv',index=False)
    meta.update(model_name=name,best_params=params,metrics=summary,stage='local_retrain')
    (out/'model_metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
