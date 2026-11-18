from pathlib import Path
import numpy as np
from pathsnap.data import Trip, geometry, segment_projection, load_trip
from pathsnap.features import candidate_features
from pathsnap.reference import align_reference
from pathsnap.metrics import classification_metrics, edit_similarity


def synthetic():
    nodes=np.array([[0,0],[.001,0],[.002,0],[.001,.001],[.002,.001]])
    arcs=np.array([[0,1],[1,0],[1,2],[2,1],[1,3],[3,1],[3,4],[4,3]])
    track=np.array([[.0002,.00001,1],[.0008,-.00001,2],[.0015,.00001,3]])
    return Trip('toy',track,nodes,arcs)


def test_candidates_equal_exhaustive_segment_search():
    trip=synthetic(); f=candidate_features(trip,k=4)
    points,starts,vectors,_=geometry(trip)
    for t,p in enumerate(points):
        dist,_,_=segment_projection(p,starts,vectors)
        expected=np.lexsort((np.arange(len(dist)),dist))[:4]
        np.testing.assert_array_equal(f['candidates'][t],expected)


def test_reference_alignment_is_ordered():
    labels,error=align_reference(synthetic(),np.array([0,2]))
    np.testing.assert_array_equal(labels,[0,0,2])
    assert np.all(error<2)


def test_binary_metrics_have_correct_specificity():
    metrics=classification_metrics([0,0,0,1,1],[.1,.2,.8,.9,.1])
    assert metrics['tn']==2 and metrics['fp']==1
    assert metrics['specificity']==2/3 and metrics['recall']==.5


def test_sequence_edit_known_cases():
    assert edit_similarity([1,2,3],[1,2,3])==1
    assert edit_similarity([1,2],[3,4])==0


def test_features_cannot_access_reference(tmp_path):
    trip=synthetic();folder=tmp_path/'toy';folder.mkdir()
    np.savetxt(folder/'toy.track',trip.track)
    np.savetxt(folder/'toy.nodes',trip.nodes)
    np.savetxt(folder/'toy.arcs',trip.arcs,fmt='%d')
    # There is deliberately no .route file anywhere in this inference input.
    parsed=load_trip(folder)
    features=candidate_features(parsed,k=4)
    assert features['X'].shape==(3,4,18)
    assert np.isfinite(features['X']).all()


def test_group_folds_do_not_share_trips():
    import json
    path=Path('results/baseline/cross_validation.json')
    if path.exists():
        for folds in json.loads(path.read_text()).values():
            for fold in folds:
                assert not set(fold['train_trips']) & set(fold['validation_trips'])
