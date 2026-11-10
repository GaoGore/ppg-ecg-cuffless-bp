"""Regularised linear models: they extrapolate to BP extremes that trees cannot reach."""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler


class LinModel:
    def __init__(self, alpha=30.0, huber=False, gaussianize=True, seed=0):
        self.alpha, self.huber, self.gaussianize = alpha, huber, gaussianize
        self.models = []

    def _pipe(self):
        steps = [SimpleImputer(strategy="median")]
        if self.gaussianize:      # robust to heavy-tailed / outlying feature values
            steps.append(QuantileTransformer(n_quantiles=200, output_distribution="normal", subsample=10 ** 6))
        steps.append(StandardScaler())
        reg = HuberRegressor(alpha=self.alpha, epsilon=1.5, max_iter=2000) if self.huber else Ridge(alpha=self.alpha)
        steps.append(reg)
        return make_pipeline(*steps)

    def fit(self, X, Y, w=None):
        X = np.asarray(X, float)
        keep = np.isfinite(X).mean(axis=0) > 0.5
        self.keep = keep
        self.models = []
        for j in range(Y.shape[1]):
            p = self._pipe()
            kw = {p.steps[-1][0] + "__sample_weight": w} if w is not None else {}
            p.fit(X[:, keep], Y[:, j], **kw)
            self.models.append(p)
        return self

    def predict(self, X):
        X = np.asarray(X, float)[:, self.keep]
        return np.column_stack([m.predict(X) for m in self.models])
