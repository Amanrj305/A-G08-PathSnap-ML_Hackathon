import numpy as np
from pathsnap.data import Trip
from pathsnap.decoder import Router,viterbi


def test_hmm_resolves_reverse_arcs_using_directed_motion():
    nodes=np.array([[0.,0.],[.001,0.],[.002,0.]])
    arcs=np.array([[0,1],[1,0],[1,2],[2,1]])
    trip=Trip('toy',np.array([[.0002,0.,1],[.0008,0.,2],[.0015,0.,3]]),nodes,arcs)
    f=dict(candidates=np.array([[0,1],[0,1],[2,3]]),fractions=np.array([[.2,.8],[.8,.2],[.5,.5]]))
    p=np.array([[.55,.45],[.45,.55],[.6,.4]])
    result=viterbi(trip,f,p,dict(max_speed_mps=120))
    np.testing.assert_array_equal(result['predicted'],[0,0,2])
    assert result['route_chunks']==[[0,2]]


def test_disconnected_graph_resets_instead_of_fabricating_bridge():
    nodes=np.array([[0.,0.],[.001,0.],[1.,1.],[1.001,1.]])
    trip=Trip('toy',np.array([[.0005,0.,1],[1.0005,1.,2]]),nodes,np.array([[0,1],[2,3]]))
    result=viterbi(trip,dict(candidates=np.array([[0],[1]]),fractions=np.array([[.5],[.5]])),np.ones((2,1)))
    assert result['resets']==[dict(observation=1,reason='no_reachable_candidate')]
    assert result['route_chunks']==[[0],[1]]


def test_router_reconstructs_directed_bridge():
    nodes=np.array([[0.,0.],[.001,0.],[.002,0.],[.003,0.]])
    trip=Trip('toy',np.array([[0,0,1],[.003,0,2]]),nodes,np.array([[0,1],[1,2],[2,3]]))
    router=Router(trip)
    assert router.bridge(0,2)==[1]
    assert router.bridge(2,0) is None
