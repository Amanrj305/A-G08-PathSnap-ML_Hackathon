import json
import joblib,numpy as np
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from api import app as server


def test_api_contract_and_reference_isolation(tmp_path,monkeypatch):
    data=tmp_path/'data';folder=data/'00000001';folder.mkdir(parents=True)
    nodes=np.array([[0.,0.],[.001,0.],[.002,0.],[.001,.001],[.002,.001]])
    arcs=np.array([[0,1],[1,0],[1,2],[2,1],[1,3],[3,1],[3,4],[4,3]])
    track=np.array([[.0002,.00001,1],[.0008,.00001,2],[.0015,.00001,3]])
    for ext,array,fmt in [('nodes',nodes,'%.8f'),('arcs',arcs,'%d'),('track',track,'%.8f')]:
        np.savetxt(folder/f'00000001.{ext}',array,fmt=fmt)
    np.savetxt(folder/'00000001.route',[0,2],fmt='%d')
    artifacts=tmp_path/'artifacts';artifacts.mkdir()
    model=DummyClassifier(strategy='prior').fit(np.zeros((4,18)),[0,0,0,1])
    joblib.dump(model,artifacts/'model.joblib')
    (artifacts/'model_metadata.json').write_text(json.dumps(dict(model_name='test_fixture',k=4)))
    monkeypatch.setattr(server,'DATA_ROOT',data)
    monkeypatch.setattr(server,'ARTIFACT_ROOT',artifacts)
    with TestClient(server.app) as client:
        assert client.get('/health').json()['trip_count']==1
        assert client.get('/trips').json()['trips']==[{'trip_id':'00000001'}]
        assert client.post('/predict',json={'trip_id':'../../secret'}).status_code==422
        assert client.post('/predict',json={'trip_id':'00000001','labels':[0,2]}).status_code==422
        assert client.post('/predict',json={'trip_id':'00000002'}).status_code==404
        first=client.post('/predict',json={'trip_id':'00000001'})
        assert first.status_code==200
        first=first.json()
        assert len(first['predicted_arc_ids'])==len(track)
        assert first['evaluation']['available']
        assert not {'reference','reference_route','reference_arc_ids','labels'} & set(first)
        # Changing scoring references cannot change inference.
        np.savetxt(folder/'00000001.route',[1,3],fmt='%d')
        second=client.post('/predict',json={'trip_id':'00000001'}).json()
        assert first['predicted_arc_ids']==second['predicted_arc_ids']
        # No .route file: same prediction and explicit unavailable score.
        (folder/'00000001.route').rename(folder/'held_out_reference.txt')
        third=client.post('/predict',json={'trip_id':'00000001'}).json()
        assert third['predicted_arc_ids']==first['predicted_arc_ids']
        assert not third['evaluation']['available']
