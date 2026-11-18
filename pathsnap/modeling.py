"""Portable scikit-learn model definitions shared by comparison and tuning."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, HistGradientBoostingClassifier)
from .id3 import ID3Classifier

SEED = 35218

# First-evaluation (syllabus) models.
SYLLABUS_MODELS = ['logistic', 'gaussian_nb', 'id3', 'random_forest', 'gradient_boosting', 'knn']
# Post-review models (outside the syllabus, allowed after first-evaluation feedback).
# "_rel" variants add candidate-relative features (see RelativeFeatures).
EXTRA_MODELS = ['extra_trees', 'hist_gradient_boosting', 'mlp',
                'random_forest_rel', 'hist_gradient_boosting_rel', 'mlp_rel',
                'xgboost', 'lightgbm', 'catboost', 'xgboost_rel', 'lightgbm_rel', 'catboost_rel']
ALL_MODELS = SYLLABUS_MODELS + EXTRA_MODELS


class RelativeFeatures(BaseEstimator, TransformerMixin):
    """Compare each candidate with the other candidates of the SAME GPS point.

    Rows arrive as consecutive blocks of k candidates per observation (every caller
    reshapes X[n, k, F] -> [n*k, F]). For each feature we append its difference from
    the observation's candidate mean and from the observation's max value, so the
    model scores a road relative to its competitors, which is what Top-1 needs.
    Output: F raw + F (x - mean_k) + F (x - max_k) = 3F columns. No labels are used.
    """
    def __init__(self, k=8):
        self.k = k

    def fit(self, X, y=None):
        if len(X) % self.k:
            raise ValueError(f'rows ({len(X)}) must be a multiple of k={self.k}')
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if len(X) % self.k:
            raise ValueError(f'rows ({len(X)}) must be a multiple of k={self.k}')
        block = X.reshape(-1, self.k, X.shape[1])
        mean = block - block.mean(axis=1, keepdims=True)
        best = block - block.max(axis=1, keepdims=True)
        return np.concatenate([block, mean, best], axis=2).reshape(len(X), -1)


def _base(name, params):
    if name == 'logistic':
        config=dict(C=1,max_iter=400);config.update(params)
        return make_pipeline(StandardScaler(), LogisticRegression(**config))
    if name == 'gaussian_nb':
        config=dict(var_smoothing=1e-7);config.update(params)
        return make_pipeline(StandardScaler(), GaussianNB(**config))
    if name == 'id3':
        return ID3Classifier(**params)
    if name == 'knn':
        config=dict(n_neighbors=15,weights='distance',algorithm='kd_tree',leaf_size=40,n_jobs=6)
        config.update(params)
        return make_pipeline(StandardScaler(),KNeighborsClassifier(**config))
    if name == 'random_forest':
        config=dict(n_estimators=80,max_depth=18,min_samples_leaf=12,n_jobs=6,random_state=SEED)
        config.update(params)
        return RandomForestClassifier(**config)
    if name == 'gradient_boosting':
        config = dict(n_estimators=100, learning_rate=.08, max_depth=2,min_samples_leaf=30,
                      subsample=.7,max_features='sqrt',random_state=SEED,loss='log_loss')
        config.update(params)
        return GradientBoostingClassifier(**config)
    if name == 'extra_trees':
        config=dict(n_estimators=200,min_samples_leaf=5,max_features=.5,n_jobs=6,random_state=SEED)
        config.update(params)
        return ExtraTreesClassifier(**config)
    if name == 'hist_gradient_boosting':
        config=dict(max_iter=400,learning_rate=.08,max_leaf_nodes=63,min_samples_leaf=40,
                    l2_regularization=1.0,early_stopping=False,random_state=SEED)
        config.update(params)
        return HistGradientBoostingClassifier(**config)
    if name == 'mlp':
        config=dict(hidden_layer_sizes=(128,64),activation='relu',alpha=1e-4,batch_size=1024,
                    learning_rate_init=1e-3,max_iter=80,early_stopping=True,
                    n_iter_no_change=8,random_state=SEED)
        config.update(params)
        return make_pipeline(StandardScaler(), MLPClassifier(**config))
    # Third-party boosting libraries: imported lazily so the API runs without them
    # unless the deployed model actually uses one.
    if name == 'xgboost':
        from xgboost import XGBClassifier
        config=dict(n_estimators=500,learning_rate=.06,max_depth=8,min_child_weight=5,subsample=.8,
                    colsample_bytree=.8,reg_lambda=1.,tree_method='hist',eval_metric='logloss',
                    n_jobs=6,random_state=SEED)
        config.update(params)
        return XGBClassifier(**config)
    if name == 'lightgbm':
        from lightgbm import LGBMClassifier
        config=dict(n_estimators=500,learning_rate=.06,num_leaves=63,min_child_samples=20,subsample=.8,
                    subsample_freq=1,colsample_bytree=.8,reg_lambda=1.,n_jobs=6,random_state=SEED,verbose=-1)
        config.update(params)
        return LGBMClassifier(**config)
    if name == 'catboost':
        from catboost import CatBoostClassifier
        config=dict(iterations=600,learning_rate=.08,depth=8,l2_leaf_reg=3.,random_seed=SEED,
                    thread_count=6,verbose=0,allow_writing_files=False)
        config.update(params)
        return CatBoostClassifier(**config)
    raise ValueError(f'Unknown model: {name}')


def make_model(name, params=None):
    params = dict(params or {})
    if name.endswith('_rel'):
        k = params.pop('k', 8)
        return make_pipeline(RelativeFeatures(k=k), _base(name[:-4], params))
    return _base(name, params)
