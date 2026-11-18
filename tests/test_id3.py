import numpy as np
from pathsnap.id3 import ID3Classifier,entropy


def test_id3_learns_multiway_categorical_rule():
    X=np.tile(np.array([[0,0],[0,1],[1,0],[1,1],[2,0],[2,1]],float),(50,1))
    y=(X[:,0]==1).astype(int)
    model=ID3Classifier(n_bins=3,min_samples_split=2,max_depth=3).fit(X,y)
    np.testing.assert_array_equal(model.predict(X),y)
    assert model.tree_['feature']==0
    assert len(model.tree_['children'])==3
    assert np.allclose(model.predict_proba(X).sum(axis=1),1)


def test_entropy_pure_and_balanced():
    assert entropy([10,0])==0
    assert entropy([5,5])==1
