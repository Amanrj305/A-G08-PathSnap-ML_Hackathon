"""Directed-network HMM/Viterbi decoding; no reference data is accepted."""
import heapq
from collections import OrderedDict
import numpy as np
from .data import geometry

DEFAULT_CONFIG = dict(beta_m=15.0, emission_weight=1.0, gap_reset_s=60.,
                      max_speed_mps=55., distance_slack_m=60., routing_limit_m=2000.,
                      backward_jitter_m=15., realign=False)


def realign_to_path(points, starts, vectors, path, scale_m=15.):
    """Re-assign GPS points to arcs of the DECODED path (no reference data).

    Monotone dynamic programme: points keep their order along the predicted path and
    each point prefers the nearest path arc. Fixes early/late switching at junctions
    and gives short bridged arcs their points.
    """
    from .data import segment_projection
    path = np.asarray(path, dtype=int)
    n, m = len(points), len(path)
    back = np.empty((n, m), dtype=np.int32)
    previous = np.zeros(m)
    columns = np.arange(m)
    for t in range(n):
        distance, _, _ = segment_projection(points[t], starts[path], vectors[path])
        emission = np.minimum(distance / scale_m, 1000) ** 2
        prefix = np.minimum.accumulate(previous)
        minima = np.r_[True, previous[1:] < prefix[:-1]]
        back[t] = np.maximum.accumulate(np.where(minima, columns, 0))
        previous = prefix + emission
        previous -= previous.min()
    chosen = np.empty(n, np.int32); j = int(previous.argmin())
    for t in range(n - 1, -1, -1):
        chosen[t] = j; j = int(back[t, j])
    return path[chosen]


class Router:
    def __init__(self,trip,limit=2000.,cache_size=1024):
        self.arcs=trip.arcs
        _,_,_,self.lengths=geometry(trip)
        self.adj=[[] for _ in trip.nodes]
        for arc,(a,b) in enumerate(trip.arcs):
            self.adj[a].append((int(b),float(self.lengths[arc]),arc))
        self.limit,self.cache_size=limit,cache_size
        self.cache=OrderedDict()

    def search(self,source):
        source=int(source)
        if source in self.cache:
            self.cache.move_to_end(source);return self.cache[source]
        distances={source:0.};parent={};queue=[(0.,source)]
        while queue:
            cost,u=heapq.heappop(queue)
            if cost != distances[u]:continue
            for v,length,arc in self.adj[u]:
                new=cost+length
                if new <= self.limit and new < distances.get(v,float('inf')):
                    distances[v]=new;parent[v]=(u,arc);heapq.heappush(queue,(new,v))
        self.cache[source]=(distances,parent)
        if len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        return distances,parent

    def bridge(self,arc_a,arc_b):
        if arc_a==arc_b:return []
        source,target=map(int,(self.arcs[arc_a,1],self.arcs[arc_b,0]))
        dist,parent=self.search(source)
        if target not in dist:return None
        path=[];node=target
        while node != source:
            node,arc=parent[node];path.append(arc)
        return path[::-1]


def transition_distances(router,old,new,frac_old,frac_new,backward_jitter_m=15):
    old,new=np.asarray(old),np.asarray(new)
    distances=np.full((len(old),len(new)),np.inf)
    for i,a in enumerate(old):
        lookup,_=router.search(router.arcs[a,1])
        remainder=(1-frac_old[i])*router.lengths[a]
        for j,b in enumerate(new):
            if a==b:
                advance=(frac_new[j]-frac_old[i])*router.lengths[a]
                if advance>=-backward_jitter_m:distances[i,j]=abs(advance)
            candidate=remainder+lookup.get(int(router.arcs[b,0]),np.inf)+frac_new[j]*router.lengths[b]
            distances[i,j]=min(distances[i,j],candidate)
    return distances


def viterbi(trip,features,probabilities,config=None,router=None):
    config={**DEFAULT_CONFIG,**(config or {})}
    router=router or Router(trip,config['routing_limit_m'])
    candidates,frac=features['candidates'],features['fractions']
    probabilities=np.clip(probabilities,1e-7,1-1e-7)
    probabilities=probabilities/probabilities.sum(axis=1,keepdims=True)
    emissions=np.log(probabilities)*config['emission_weight']
    n,k=candidates.shape
    scores=np.empty((n,k));back=np.full((n,k),-1,np.int32)
    scores[0]=emissions[0]
    points,_,_,_=geometry(trip)
    resets=[]
    for t in range(1,n):
        dt=trip.track[t,2]-trip.track[t-1,2]
        if dt>config['gap_reset_s']:
            scores[t]=emissions[t];resets.append(dict(observation=t,reason='time_gap'));continue
        network=transition_distances(router,candidates[t-1],candidates[t],frac[t-1],frac[t],config['backward_jitter_m'])
        gps_distance=np.linalg.norm(points[t]-points[t-1])
        possible=network <= min(config['routing_limit_m'],config['max_speed_mps']*max(dt,1)+config['distance_slack_m'])
        transition=np.where(possible,-np.abs(network-gps_distance)/config['beta_m'],-np.inf)
        options=scores[t-1,:,None]+transition
        parent=np.argmax(options,axis=0)
        row=options[parent,np.arange(k)]+emissions[t]
        if not np.isfinite(row).any():
            scores[t]=emissions[t];resets.append(dict(observation=t,reason='no_reachable_candidate'));continue
        back[t]=parent;scores[t]=row-np.max(row)
    state=int(scores[-1].argmax());states=np.empty(n,np.int32)
    for t in range(n-1,-1,-1):
        states[t]=state
        parent=int(back[t,state])
        state=parent if parent>=0 else int(scores[t-1].argmax()) if t else 0
    predicted=candidates[np.arange(n),states]
    reset_indices={r['observation'] for r in resets}
    paths=[[int(predicted[0])]];bridge_failures=0;chunk_start=[0]
    for t in range(1,n):
        a,b=map(int,predicted[t-1:t+1])
        if t in reset_indices:
            paths.append([b]);chunk_start.append(t);continue
        if a==b:continue
        bridge=router.bridge(a,b)
        if bridge is None:
            bridge_failures+=1;paths.append([b]);chunk_start.append(t)
        else:paths[-1].extend(bridge+[b])
    viterbi_predicted=predicted.copy()
    if config.get('realign'):
        _,starts,vectors,_=geometry(trip)
        bounds=chunk_start+[n]
        for c,path in enumerate(paths):
            lo,hi=bounds[c],bounds[c+1]
            if hi>lo and len(path)>1:
                predicted[lo:hi]=realign_to_path(points[lo:hi],starts,vectors,path)
    return dict(predicted=predicted,viterbi_predicted=viterbi_predicted,states=states,route_chunks=paths,
                resets=resets,bridge_failures=bridge_failures,config=config)
