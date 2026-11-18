"""ID3: multiway categorical splits selected by Shannon information gain.

Continuous features are quantile-discretized using fitting data only. This is
an ID3 implementation, not scikit-learn's binary CART tree renamed as ID3.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def entropy(counts):
    counts=np.asarray(counts,dtype=float)
    total=counts.sum(axis=-1,keepdims=True)
    probabilities=np.divide(counts,total,out=np.zeros_like(counts),where=total>0)
    return -np.sum(probabilities*np.log2(np.maximum(probabilities,1e-15)),axis=-1)


class ID3Classifier(ClassifierMixin,BaseEstimator):
    def __init__(self,n_bins=6,max_depth=8,min_samples_split=100,min_gain=1e-5,alpha=1.):
        self.n_bins=n_bins;self.max_depth=max_depth;self.min_samples_split=min_samples_split
        self.min_gain=min_gain;self.alpha=alpha

    def fit(self,X,y):
        X,y=check_X_y(X,y);self.n_features_in_=X.shape[1]
        self.classes_,encoded=np.unique(y,return_inverse=True)
        self.bin_edges_=[]
        for column in X.T:
            unique=np.unique(column)
            if len(unique)<=self.n_bins:
                edges=(unique[:-1]+unique[1:])/2
            else:
                edges=np.unique(np.quantile(column,np.arange(1,self.n_bins)/self.n_bins,method='midpoint'))
                edges=edges[(edges>unique[0]) & (edges<unique[-1])]
            self.bin_edges_.append(edges)
        binned=self._bin(X)
        self.node_count_=0
        def build(index,features,depth):
            self.node_count_+=1
            counts=np.bincount(encoded[index],minlength=len(self.classes_))
            probs=(counts+self.alpha)/(len(index)+self.alpha*len(self.classes_))
            node=dict(probabilities=probs,feature=None,children={})
            if depth>=self.max_depth or len(index)<self.min_samples_split or np.count_nonzero(counts)<2 or not features:return node
            base=float(entropy(counts));best_gain=-np.inf;best_feature=None
            for feature in features:
                histogram=np.bincount(binned[index,feature]*len(self.classes_)+encoded[index],
                                      minlength=self.n_bins*len(self.classes_)).reshape(self.n_bins,-1)
                gain=base-float(np.dot(histogram.sum(axis=1)/len(index),entropy(histogram)))
                if gain>best_gain:best_gain,best_feature=gain,feature
            if best_gain<self.min_gain:return node
            node['feature']=best_feature
            remaining=[f for f in features if f!=best_feature]
            for value in np.unique(binned[index,best_feature]):
                child=index[binned[index,best_feature]==value]
                node['children'][int(value)]=build(child,remaining,depth+1)
            return node
        self.tree_=build(np.arange(len(y)),list(range(X.shape[1])),0)
        return self

    def predict_proba(self,X):
        check_is_fitted(self,'tree_');X=check_array(X)
        binned=self._bin(X)
        result=np.empty((len(X),len(self.classes_)))
        def visit(node,index):
            result[index]=node['probabilities']
            feature=node['feature']
            if feature is None:return
            for value,child in node['children'].items():
                subset=index[binned[index,feature]==value]
                if len(subset):visit(child,subset)
        visit(self.tree_,np.arange(len(X)))
        return result

    def predict(self,X):
        return self.classes_[np.argmax(self.predict_proba(X),axis=1)]

    def _bin(self,X):
        return np.column_stack([np.searchsorted(edges,X[:,j],side='right') for j,edges in enumerate(self.bin_edges_)]).astype(np.int16)
