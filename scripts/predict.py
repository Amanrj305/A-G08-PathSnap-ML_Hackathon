"""Run real label-free inference on one trip folder, optionally with the HMM."""
import argparse,json,time
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from pathsnap.data import load_trip
from pathsnap.features import candidate_features
from pathsnap.decoder import viterbi,DEFAULT_CONFIG


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trip',default='data/Student folder/student_test/00000007')
    ap.add_argument('--artifacts',default='artifacts')
    ap.add_argument('--output',default='output/prediction.json')
    ap.add_argument('--classifier-only',action='store_true')
    args=ap.parse_args()
    t=time.perf_counter();root=Path(args.artifacts)
    meta=json.loads((root/'model_metadata.json').read_text())
    model=joblib.load(root/'model.joblib');trip=load_trip(Path(args.trip))
    features=candidate_features(trip,k=meta['k'])
    with threadpool_limits(limits=6):
        p=model.predict_proba(features['X'].reshape(-1,features['X'].shape[-1]))[:,1].reshape(features['X'].shape[:2])
    if args.classifier_only:
        prediction=features['candidates'][np.arange(len(p)),p.argmax(axis=1)]
        extra={}
    else:
        cfg=json.loads((root/'decoder_config.json').read_text()) if (root/'decoder_config.json').exists() else DEFAULT_CONFIG
        decoded=viterbi(trip,features,p,cfg);prediction=decoded['predicted']
        extra={k:decoded[k] for k in ['route_chunks','resets','bridge_failures']}
    result=dict(trip_id=trip.trip_id,model=meta['model_name'],observations=len(p),
                predicted_arc_ids=prediction.tolist(),seconds=time.perf_counter()-t,**extra)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result))
    print(json.dumps({k:v for k,v in result.items() if k not in ['predicted_arc_ids','route_chunks']},indent=2))
    print('Predictions saved to',path)


if __name__=='__main__':main()
