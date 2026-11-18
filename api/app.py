"""FastAPI inference with a separate, post-prediction server-side scorer."""
import hashlib,json,logging,os,re,threading,time
from contextlib import asynccontextmanager
from pathlib import Path
import joblib,numpy as np
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,ConfigDict,Field
from threadpoolctl import threadpool_limits
from pathsnap.data import load_trip
from pathsnap.features import candidate_features
from pathsnap.decoder import viterbi,DEFAULT_CONFIG

ROOT=Path(__file__).resolve().parents[1]
DATA_ROOT=Path(os.environ.get('PATHSNAP_DATA_ROOT',str(ROOT/'data/Student folder/student_test'))).resolve()
ARTIFACT_ROOT=Path(os.environ.get('PATHSNAP_ARTIFACT_ROOT',str(ROOT/'artifacts'))).resolve()
MODE=os.environ.get('PATHSNAP_MODE','student_development')
LOGGER=logging.getLogger('uvicorn.error')
LOCK=threading.Lock()


@asynccontextmanager
async def lifespan(app):
    app.state.model=joblib.load(ARTIFACT_ROOT/'model.joblib')
    app.state.metadata=json.loads((ARTIFACT_ROOT/'model_metadata.json').read_text())
    app.state.model_sha256=hashlib.sha256((ARTIFACT_ROOT/'model.joblib').read_bytes()).hexdigest()
    cfg=ARTIFACT_ROOT/'decoder_config.json'
    app.state.decoder_config=json.loads(cfg.read_text()) if cfg.exists() else DEFAULT_CONFIG
    LOGGER.info('PathSnap ready model=%s mode=%s data=%s',app.state.metadata['model_name'],MODE,DATA_ROOT)
    yield


app=FastAPI(title='PathSnap',version='0.1.0',lifespan=lifespan)
origins=os.environ.get('PATHSNAP_UI_ORIGINS','*').split(',')
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_methods=['GET','POST'],allow_headers=['Content-Type'])


class PredictionRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    trip_id:str=Field(pattern=r'^\d{8}$')


def trip_folders():
    if not DATA_ROOT.is_dir():return {}
    return {p.name:p for p in sorted(DATA_ROOT.iterdir()) if p.is_dir() and re.fullmatch(r'\d{8}',p.name)
            and all((p/f'{p.name}.{ext}').is_file() for ext in ['track','nodes','arcs'])}


@app.get('/health')
def health():
    return dict(status='ready',version='0.1.0',model=app.state.metadata['model_name'],
                model_sha256=app.state.model_sha256,mode=MODE,trip_count=len(trip_folders()))


@app.get('/trips')
def trips():
    return dict(mode=MODE,trips=[dict(trip_id=k) for k in trip_folders()])


def score_after_prediction(folder,trip,raw,predicted):
    # Labels are opened ONLY after inference has finished. Never returned as geometry/IDs.
    path=folder/f'{folder.name}.route'
    if not path.exists():
        return dict(available=False,reason='Server reference route was not supplied.')
    from pathsnap.reference import load_reference,align_reference
    from pathsnap.metrics import sequence_metrics
    route=load_reference(folder,len(trip.arcs));reference,_=align_reference(trip,route)
    return dict(available=True,label_basis='Derived monotone alignment to ordered reference route',
                classifier=sequence_metrics(trip,raw,reference,route),
                hmm=sequence_metrics(trip,predicted,reference,route))


@app.post('/predict')
def predict(request:PredictionRequest):
    folder=trip_folders().get(request.trip_id)
    if folder is None:raise HTTPException(404,'Trip is not present in the configured server partition.')
    if not LOCK.acquire(blocking=False):raise HTTPException(503,'A trip is currently processing. Retry shortly.')
    try:
        start=time.perf_counter();trip=load_trip(folder)
        f=candidate_features(trip,k=app.state.metadata['k'])
        feature_seconds=time.perf_counter()-start
        t=time.perf_counter()
        with threadpool_limits(limits=6):
            p=app.state.model.predict_proba(f['X'].reshape(-1,f['X'].shape[-1]))[:,1].reshape(f['X'].shape[:2])
        classifier_seconds=time.perf_counter()-t
        raw=f['candidates'][np.arange(len(p)),p.argmax(axis=1)]
        t=time.perf_counter();decoded=viterbi(trip,f,p,app.state.decoder_config);decoder_seconds=time.perf_counter()-t
        inference_seconds=time.perf_counter()-start
        scoring=score_after_prediction(folder,trip,raw,decoded['predicted'])
        arc_ids=np.unique(f['candidates'])
        network=trip.nodes[trip.arcs[arc_ids]].tolist()
        routes=[]
        for chunk in decoded['route_chunks']:
            lines=trip.nodes[trip.arcs[np.asarray(chunk,dtype=int)]].tolist()
            routes.extend(lines)
        result=dict(trip_id=trip.trip_id,mode=MODE,model=app.state.metadata['model_name'],
                    observations=len(trip.track),gps=trip.track[:,:2].tolist(),
                    predicted_arc_ids=decoded['predicted'].tolist(),
                    road_network=dict(type='MultiLineString',coordinates=network),
                    predicted_route=dict(type='MultiLineString',coordinates=routes),
                    route_chunk_count=len(decoded['route_chunks']),reset_count=len(decoded['resets']),
                    reset_details=decoded['resets'],bridge_failures=decoded['bridge_failures'],
                    timing=dict(features_seconds=feature_seconds,classifier_seconds=classifier_seconds,
                                decoder_seconds=decoder_seconds,inference_seconds=inference_seconds,
                                total_seconds=time.perf_counter()-start),evaluation=scoring)
        LOGGER.info('PREDICT trip=%s model=%s points=%d seconds=%.3f status=ok',trip.trip_id,
                    result['model'],len(trip.track),result['timing']['total_seconds'])
        return result
    except ValueError as exc:
        LOGGER.exception('Invalid trip input: %s',request.trip_id)
        raise HTTPException(422,str(exc)) from exc
    finally:LOCK.release()
