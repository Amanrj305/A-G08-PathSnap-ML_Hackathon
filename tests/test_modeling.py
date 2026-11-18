import numpy as np
import pytest
from pathsnap.modeling import RelativeFeatures, make_model, ALL_MODELS


def test_relative_features_compare_within_each_observation():
    X = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)   # 2 points, k=4, 3 features
    out = RelativeFeatures(k=4).fit(X.reshape(-1, 3)).transform(X.reshape(-1, 3)).reshape(2, 4, 9)
    np.testing.assert_allclose(out[..., :3], X)
    np.testing.assert_allclose(out[..., 3:6], X - X.mean(axis=1, keepdims=True))
    np.testing.assert_allclose(out[..., 6:], X - X.max(axis=1, keepdims=True))
    # Second observation's relative values do not depend on the first observation.
    np.testing.assert_allclose(out[1, :, 3:], out[0, :, 3:])


def test_relative_features_reject_partial_blocks():
    with pytest.raises(ValueError):
        RelativeFeatures(k=8).fit(np.zeros((10, 3)))


@pytest.mark.parametrize('name', ALL_MODELS)
def test_every_model_fits_and_scores_blocks(name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40 * 8, 18))
    y = np.zeros(40 * 8, dtype=np.uint8); y[::8] = 1
    params = {'n_estimators': 5} if name.startswith(('random_forest', 'extra_trees', 'gradient_boosting')) else {}
    if name.startswith('hist_gradient_boosting'): params = {'max_iter': 5}
    if name.startswith('mlp'): params = {'max_iter': 5, 'early_stopping': False}
    if name == 'knn': params = {'n_neighbors': 3}
    if name.startswith(('xgboost', 'lightgbm')): params = {'n_estimators': 5}
    if name.startswith('catboost'): params = {'iterations': 5}
    model = make_model(name, params).fit(X, y)
    p = model.predict_proba(X)[:, 1].reshape(40, 8)
    assert p.shape == (40, 8) and np.isfinite(p).all()
