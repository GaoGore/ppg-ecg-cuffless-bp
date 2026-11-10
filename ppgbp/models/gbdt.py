"""CatBoost regressors for [SBP, DBP] (one model per target)."""
from __future__ import annotations

import numpy as np
from catboost import CatBoostRegressor

DEFAULT = dict(iterations=1500, depth=5, learning_rate=0.03, l2_leaf_reg=5, rsm=0.2,
               bootstrap_type="Bernoulli", subsample=0.8, verbose=0, thread_count=12)


class CatModel:
    def __init__(self, loss="RMSE", seed=0, **params):
        self.params = {**DEFAULT, **params, "loss_function": loss, "random_seed": seed}
        self.models = []

    def fit(self, X, Y, w=None):
        self.models = []
        for j in range(Y.shape[1]):
            m = CatBoostRegressor(**self.params)
            m.fit(X, Y[:, j], sample_weight=w)
            self.models.append(m)
        return self

    def predict(self, X):
        return np.column_stack([m.predict(X) for m in self.models])


class CatQuantile:
    """Joint quantile model per target (CatBoost MultiQuantile)."""

    def __init__(self, alphas=None, seed=0, **params):
        self.alphas = np.asarray(alphas if alphas is not None else
                                 [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98])
        loss = "MultiQuantile:alpha=" + ",".join(f"{a:g}" for a in self.alphas)
        self.params = {**DEFAULT, **params, "loss_function": loss, "random_seed": seed}
        self.models = []

    def fit(self, X, Y, w=None):
        self.models = []
        for j in range(Y.shape[1]):
            m = CatBoostRegressor(**self.params)
            m.fit(X, Y[:, j], sample_weight=w)
            self.models.append(m)
        return self

    def predict_quantiles(self, X):
        q = np.stack([m.predict(X) for m in self.models], axis=1)   # (n, 2, Q)
        return np.sort(q, axis=2)

    def predict(self, X):
        q = self.predict_quantiles(X)
        return q[:, :, int(np.argmin(np.abs(self.alphas - 0.5)))]
