import argparse,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from pathsnap.data import load_trip
from pathsnap.decoder import viterbi,DEFAULT_CONFIG
from pathsnap.metrics import sequence_metrics
from pathsnap.reference import load_reference


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/Student folder');ap.add_argument('--cache',default='cache');args=ap.parse_args()
    out=Path('results/decoder');out.mkdir(parents=True,exist_ok=True)
    meta=json.loads(Path('artifacts/model_metadata.json').read_text());model=joblib.load('artifacts/model.joblib')
    rows=[]
    # Fixed baseline decoder parameters; no evaluator labels or hidden tuning.
    Path('artifacts/decoder_config.json').write_text(json.dumps(DEFAULT_CONFIG,indent=2))
    for path in sorted(Path(args.cache).glob(f'student_test_*_k{meta["k"]}.npz')):
        trip_id=path.stem.split('_')[2];folder=Path(args.data)/'student_test'/trip_id;trip=load_trip(folder);route=load_reference(folder,len(trip.arcs))
        with np.load(path) as z:f={k:z[k] for k in ['X','candidates','fractions']};reference=z['reference']
        with threadpool_limits(limits=6):p=model.predict_proba(f['X'].reshape(-1,f['X'].shape[-1]))[:,1].reshape(f['X'].shape[:2])
        raw=f['candidates'][np.arange(len(p)),p.argmax(axis=1)]
        start=time.perf_counter();decoded=viterbi(trip,f,p);elapsed=time.perf_counter()-start
        for mode,pred in [('classifier',raw),('hmm',decoded['predicted'])]:
            row=dict(trip_id=trip_id,mode=mode,observations=len(p),**sequence_metrics(trip,pred,reference,route))
            if mode=='hmm':row.update(decoder_seconds=elapsed,resets=len(decoded['resets']),bridge_failures=decoded['bridge_failures'])
            rows.append(row)
        payload={k:decoded[k] for k in ['route_chunks','resets','bridge_failures','config']};payload['predicted']=decoded['predicted'].tolist()
        (out/f'{trip_id}.json').write_text(json.dumps(payload))
        print(trip_id,'raw',rows[-2]['point_accuracy'],'hmm',rows[-1]['point_accuracy'],'seconds',round(elapsed,2),flush=True)
        pd.DataFrame(rows).to_csv(out/'per_trip_metrics.csv',index=False)
    table=pd.DataFrame(rows);summary={}
    for mode,frame in table.groupby('mode'):
        summary[mode]={key:float(np.average(frame[key],weights=frame.observations)) for key in ['point_accuracy','strict_adjacent_continuity']}
        summary[mode].update({key:float(frame[key].mean()) for key in ['sampled_sequence_edit_similarity','full_route_edit_similarity','route_arc_f1']})
        summary[mode]['decoder_seconds']=float(frame.decoder_seconds.fillna(0).sum())
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
