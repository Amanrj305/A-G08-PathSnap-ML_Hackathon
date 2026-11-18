"""TPE Bayesian optimization on group-aware TRAINING CV, never evaluator data."""
import argparse,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd,optuna
from threadpoolctl import threadpool_limits
from pathsnap.modeling import make_model,SEED
from scripts.train import training_arrays,cv_score,evaluate_student,log
from pathsnap.modeling import ALL_MODELS


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=ALL_MODELS,default=None,
                    help='model to tune (default: best CV Top-1 in results/baseline/model_comparison.csv)')
    ap.add_argument('--exclude-trips',nargs='*',default=[])
    ap.add_argument('--trials',type=int,default=12)
    ap.add_argument('--folds',type=int,default=3)
    ap.add_argument('--max-obs-per-trip',type=int,default=600)
    ap.add_argument('--data',default='data/Student folder')
    ap.add_argument('--cache',default='cache')
    ap.add_argument('--output',default='results/tuning')
    args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);(out/'predictions').mkdir(exist_ok=True)
    meta=json.loads(Path('artifacts/model_metadata.json').read_text())
    baseline=pd.read_csv('results/baseline/model_comparison.csv')
    best_baseline=baseline.loc[baseline.cv_top1_accuracy.idxmax()]
    name=args.model or best_baseline.model
    log(f'TUNING START model={name} trials={args.trials}')
    X,y,groups=training_arrays(args.cache,meta['k'],args.max_obs_per_trip,args.exclude_trips)
    log(f'Loaded {len(X)} observations from {len(set(groups))} training trips')
    spaces={
        'extra_trees':dict(n_estimators=[100,300],min_samples_leaf=[2,30],max_features=[.3,1.]),
        'hist_gradient_boosting':dict(max_iter=[150,600],learning_rate=[.02,.2,'log'],max_leaf_nodes=[15,127],
                                      min_samples_leaf=[10,100],l2_regularization=[1e-3,10.,'log']),
        'mlp':dict(alpha=[1e-6,1e-2,'log'],learning_rate_init=[1e-4,1e-2,'log']),
        'xgboost':dict(n_estimators=[200,900],learning_rate=[.02,.2,'log'],max_depth=[4,12],
                       min_child_weight=[1,20],subsample=[.6,1.],colsample_bytree=[.5,1.]),
        'lightgbm':dict(n_estimators=[200,900],learning_rate=[.02,.2,'log'],num_leaves=[15,255],
                        min_child_samples=[5,100],subsample=[.6,1.],colsample_bytree=[.5,1.]),
        'catboost':dict(iterations=[300,1200],learning_rate=[.02,.2,'log'],depth=[4,10],l2_leaf_reg=[1.,10.,'log']),
        'random_forest':dict(n_estimators=[50,140],max_depth=[10,24],min_samples_leaf=[4,40],max_features=[.3,1.]),
        'gradient_boosting':dict(n_estimators=[60,160],learning_rate=[.03,.18,'log'],max_depth=[1,3],min_samples_leaf=[15,80],subsample=[.6,1.]),
        'knn':dict(n_neighbors=[5,31],weights=['uniform','distance']),
        'id3':dict(n_bins=[4,10],max_depth=[4,10],min_samples_split=[40,200]),
        'logistic':dict(C=[.01,100.,'log']),
        'gaussian_nb':dict(var_smoothing=[1e-10,1e-4,'log'])}
    space=spaces[name[:-4] if name.endswith('_rel') else name]
    run=dict(vars(args),model=name,seed=SEED,search_space=space,objective='mean group-CV observation Top-1',
             startup_trials=3,reason='Tune the model with the highest mean trip-group CV Top-1 (or the --model given).',
             candidate_rows=y.size,observations=len(X))
    (out/'search_config.json').write_text(json.dumps(run,indent=2))
    journal_path=str(out/'study.journal')
    storage=optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(
        journal_path,lock_obj=optuna.storages.journal.JournalFileOpenLock(journal_path)))
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED,n_startup_trials=3),
                             storage=storage,study_name='pathsnap_'+name,load_if_exists=True)
    if not study.trials:
        initial=make_model(name).get_params()
        # A baseline trial is useful evidence; map pipeline leaf parameter names.
        baseline_params={key:initial.get(key,next((v for k,v in initial.items() if k.endswith('__'+key)),None)) for key in space}
        # The RF default 'sqrt' is approximately sqrt(18)/18, below this search's
        # max_features range. Keep it outside the search and compare baseline later.
        if name!='random_forest':study.enqueue_trial(baseline_params)
    def objective(trial):
        params={}
        integer_params={'n_estimators','max_depth','min_samples_leaf','n_neighbors','n_bins','min_samples_split',
                        'max_iter','max_leaf_nodes','min_child_weight','num_leaves','min_child_samples',
                        'iterations','depth'}
        for key,bounds in space.items():
            if key=='weights':params[key]=trial.suggest_categorical(key,bounds)
            elif key in integer_params:params[key]=trial.suggest_int(key,bounds[0],bounds[1])
            else:params[key]=trial.suggest_float(key,bounds[0],bounds[1],log=len(bounds)>2 and bounds[2]=='log')
        log(f'TRIAL {trial.number+1}/{args.trials} START params={params}')
        folds=cv_score(make_model(name,params),X,y,groups,args.folds)
        trial.set_user_attr('cv_macro_f1',float(np.mean([r['macro_f1'] for r in folds])))
        trial.set_user_attr('cv_log_loss',float(np.mean([r['log_loss'] for r in folds])))
        trial.set_user_attr('folds',folds)
        return float(np.mean([r['top1_accuracy'] for r in folds]))
    def report(s,t):
        s.trials_dataframe().to_csv(out/'trials.csv',index=False)
        log(f'TRIAL {t.number+1}/{args.trials} DONE cv_top1={t.value:.4f} | best so far={s.best_value:.4f} (trial {s.best_trial.number+1})')
    start=time.perf_counter()
    with threadpool_limits(limits=6):
        study.optimize(objective,n_trials=args.trials,callbacks=[report])
        log(f'Refitting best params on all training trips: {study.best_params}')
        model=make_model(name,study.best_params)
        fitstart=time.perf_counter();model.fit(X.reshape(-1,X.shape[-1]),y.ravel());fitsec=time.perf_counter()-fitstart
        log(f'Refit done in {fitsec:.1f}s, scoring student-test trips')
        joblib.dump(model,out/'tuned_model.joblib',compress=3)
        metrics,rows=evaluate_student(model,args.cache,args.data,meta['k'],name+'_tuned',out/'predictions')
    result=dict(model=name+'_tuned',best_params=study.best_params,
                cv_top1_accuracy=study.best_value,cv_macro_f1=study.best_trial.user_attrs['cv_macro_f1'],
                total_seconds=time.perf_counter()-start,train_seconds=fitsec,trials=len(study.trials),**metrics)
    (out/'tuning_summary.json').write_text(json.dumps(result,indent=2));pd.DataFrame(rows).to_csv(out/'per_trip_metrics.csv',index=False)
    # Retain the original first-evaluation artifacts in results/baseline/models.
    if result['cv_top1_accuracy'] >= best_baseline.cv_top1_accuracy:
        joblib.dump(model,'artifacts/model.joblib',compress=3)
        meta.update(model_name=result['model'],metrics=result,stage='tuned',best_params=study.best_params)
        Path('artifacts/model_metadata.json').write_text(json.dumps(meta,indent=2))
        result['selected_for_deployment']=True
    else:result['selected_for_deployment']=False
    (out/'tuning_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    log(f'TUNING END {name}: cv_top1={result["cv_top1_accuracy"]:.4f} test_top1={result["top1_accuracy"]:.4f} '
        f'deployed={result["selected_for_deployment"]} total={result["total_seconds"]:.1f}s')


if __name__=='__main__':main()
