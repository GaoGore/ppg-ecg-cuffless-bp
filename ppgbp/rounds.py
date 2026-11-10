"""Round-level deviations: how much each round's BP differs from the subject's
own mean, predicted from how that round's signals differ from the subject's
mean signals (pulse arrival, heart rate, pulse shape, perfusion) plus the
round number. Within a person, pulse arrival tracks BP changes far better than
it ranks different people, which is why this is modelled separately from the
subject-level BP level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import cache

DELTA_FEATURES = [
    "all__t_d2", "all__t_tan", "all__t_ms", "all__t_pk", "IR__t_d2", "IR__t_ms", "G__t_d2", "G__t_ms",
    "all__rise", "all__w50", "all__sys_area", "all__ri", "all__apg_b", "all__apg_agi", "all__s10", "all__s30",
    "all__pi_log", "all__amp_log", "IR__pi_log", "G__pi_log", "rhy__hr", "hr_ppg", "rhy__rmssd", "rhy__sdnn",
    "rhy__pat_sd", "rhy__riav", "ecg__qt", "ecg__qrs_dur", "acc__motion", "acc__gz",
]


def delta_table() -> tuple[pd.DataFrame, np.ndarray]:
    """Per segment: signal deviations from the subject's mean (train+test), round
    one-hot, number of rounds; plus label deviations for training segments."""
    meta, feats = cache.load()
    X = feats[DELTA_FEATURES].astype(float)
    X = X.where(np.isfinite(X))
    D = X - X.groupby(meta.sid.values).transform("mean")
    D = D.fillna(0.0)
    for r in range(1, 6):
        D[f"round_{r}"] = (meta["round"].values == r).astype(float)
    D["n_rounds"] = meta.n_rounds.values.astype(float)
    y = meta[["sbp", "dbp"]].values
    dy = y - pd.DataFrame(y).groupby(meta.sid.values).transform("mean").values
    D.index = meta.seg_id.values
    return D, dy


class RoundModel:
    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha

    def fit(self, D: pd.DataFrame, dy: np.ndarray):
        self.models = [make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=self.alpha)).fit(D.values, dy[:, j])
                       for j in range(2)]
        return self

    def predict(self, D: pd.DataFrame) -> np.ndarray:
        p = np.column_stack([m.predict(D.values) for m in self.models])
        # deviations of one subject must average to zero over its rounds
        return p
