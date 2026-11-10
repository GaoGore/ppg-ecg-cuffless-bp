"""D5: "vascular age" from signals and its gap to chronological age.

A model predicts age from pulse/ECG features plus sex only (no BP labels are
used, and only training subjects' ages). Training subjects get out-of-fold
predictions; test subjects get predictions from models trained on all
training subjects. The gap (signal age - chronological age) is a marker of
arterial ageing that demographics alone cannot see.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import cache, tables
from .models.gbdt import CatModel

OUT = cache.CACHE_DIR / "agegap.parquet"


def compute(feature_set="core", seeds=3, n_splits=5) -> pd.DataFrame:
    t = tables.load_tables(feature_set)
    S_tr, S_te = t["S_tr"], t["S_te"]
    cols = [c for c in t["cols"] if c not in tables.DEMO] + ["male"]
    X_tr, X_te = S_tr[cols], S_te[cols]
    y = S_tr[["age"]].values
    oof = np.zeros(len(S_tr))
    te = np.zeros(len(S_te))
    for s in range(seeds):
        kf = KFold(n_splits, shuffle=True, random_state=100 + s)
        for tr, va in kf.split(X_tr):
            m = CatModel(seed=s, iterations=1200).fit(X_tr.iloc[tr], y[tr])
            oof[va] += m.predict(X_tr.iloc[va])[:, 0] / seeds
        m = CatModel(seed=s, iterations=1200).fit(X_tr, y)
        te += m.predict(X_te)[:, 0] / seeds
    df = pd.DataFrame({"sid": list(S_tr.index) + list(S_te.index),
                       "age_sig": np.r_[oof, te],
                       "age": np.r_[S_tr.age.values, S_te.age.values]})
    df["age_gap"] = df.age_sig - df.age
    df.drop(columns="age").to_parquet(OUT, index=False)
    r = np.corrcoef(oof, S_tr.age)[0, 1]
    print(f"signal-age model: OOF r={r:.3f}, MAE={np.abs(oof - S_tr.age).mean():.2f} y")
    return df


def load() -> pd.DataFrame:
    return pd.read_parquet(OUT).set_index("sid")
