"""Repeated subject-level cross-validation with scenario-weighted reporting."""
from __future__ import annotations

import os
import time
from typing import Callable

import numpy as np
import pandas as pd

from . import tables, validate


class CVContext:
    """Tables, folds and scenario weights shared by all experiments."""

    def __init__(self, feature_set="core", n_splits=5, repeats=3, seed=0, extra=()):
        t = tables.load_tables(feature_set, extra)
        self.__dict__.update(t)
        self.feature_set = feature_set
        self.n_splits, self.repeats = n_splits, repeats
        fixed = os.environ.get("PPGBP_FOLDS")
        if fixed and os.path.exists(fixed):                 # folds exported from a reference run
            z = np.load(fixed, allow_pickle=False)
            table = pd.DataFrame(z["folds"], index=[str(s) for s in z["sids"]])
            self.subj_folds = [table.loc[self.S_tr.index, r].values for r in range(repeats)]
        else:
            seg_folds = validate.folds(self.m_tr, n_splits, repeats, seed)
            self.subj_folds = [pd.Series(f, index=self.m_tr.sid.values).groupby(level=0).first()
                               .reindex(self.S_tr.index).values for f in seg_folds]
        self.W = validate.scenario_weights(self.m_tr, self.m_te)
        self.Y = self.S_tr[["sbp", "dbp"]].values
        self.offsets = tables.round_offsets(self.m_tr)

    def to_segments(self, subj_pred: np.ndarray, split="train", seg_delta: np.ndarray | None = None) -> np.ndarray:
        """Subject predictions -> rows; per-round deviations from a round-level model if
        given, otherwise the average per-round offsets of the training data."""
        m = self.m_tr if split == "train" else self.m_te
        if seg_delta is not None:
            return subj_pred[m.row.values] + seg_delta
        return subj_pred[m.row.values] + self.offsets[m["round"].values]


def run(ctx: CVContext, name: str, model_fn: Callable, cols=None, weight_fn: Callable | None = None,
        log=True, notes="", verbose=True, return_models=False):
    """model_fn(seed) -> object with fit(X, Y, w) / predict(X) -> (n, 2).
    weight_fn(Y_train_subjects) -> per-subject sample weights (optional)."""
    cols = cols or ctx.cols
    X_tr, X_te = ctx.S_tr[cols], ctx.S_te[cols]
    oof = np.zeros((ctx.repeats, len(ctx.S_tr), 2))
    te = []
    t0 = time.time()
    models = []
    seg_oof = np.zeros((len(ctx.m_tr), 2))
    seg_te = []
    tr_pos = pd.Series(np.arange(len(ctx.m_tr)), index=ctx.m_tr.seg_id.values)
    te_pos = pd.Series(np.arange(len(ctx.m_te)), index=ctx.m_te.seg_id.values)
    for r in range(ctx.repeats):
        f = ctx.subj_folds[r]
        for k in range(ctx.n_splits):
            tr, va = f != k, f == k
            w = weight_fn(ctx.Y[tr]) if weight_fn else None
            m = model_fn(seed=r * 100 + k).fit(X_tr[tr], ctx.Y[tr], w)
            oof[r, va] = m.predict(X_tr[va])
            te.append(m.predict(X_te))
            if hasattr(m, "predict_seg"):
                d = m.predict_seg(X_tr[va])
                seg_oof[tr_pos.loc[d.index].values] += d.values / ctx.repeats
                dt = m.predict_seg(X_te)
                seg_te.append(dt.reindex(ctx.m_te.seg_id.values).values)
            if return_models:
                models.append((r, k, m))
    oof_subj = oof.mean(axis=0)
    oof_seg = ctx.to_segments(oof_subj)
    te_seg = ctx.to_segments(np.mean(te, axis=0), "test")
    met = validate.evaluate(oof_seg, ctx.m_tr, ctx.W)
    # per-repeat spread of the primary metric (single-model noise level)
    per_rep = [validate.evaluate(ctx.to_segments(oof[r]), ctx.m_tr, {"mid": ctx.W["mid"]})["wmae_mid"]
               for r in range(ctx.repeats)]
    met["mid_rep_sd"] = float(np.std(per_rep))
    met["seconds"] = round(time.time() - t0, 1)
    if log:
        validate.log_result(name, met, oof_seg, te_seg, notes=f"{ctx.feature_set} | {notes}")
        np.save(validate.RESULTS / "oof" / f"{name}__subj.npy", oof)                    # (repeats, n_subj, 2)
        np.save(validate.RESULTS / "oof" / f"{name}__subj_test.npy", np.mean(te, axis=0))
        if seg_te:
            np.save(validate.RESULTS / "oof" / f"{name}__segdelta.npy", seg_oof)
            np.save(validate.RESULTS / "oof" / f"{name}__segdelta_test.npy", np.mean(seg_te, axis=0))
    if verbose:
        print(fmt(name, met))
    out = {"oof": oof_seg, "test": te_seg, "oof_subj": oof_subj, "metrics": met, "oof_rep": oof}
    if return_models:
        out["models"] = models
    return out


def fmt(name: str, m: dict) -> str:
    b = " ".join(f"{k[5:]}:{m[k]:+.1f}" for k in m if k.startswith("bias_"))
    return (f"{name:38s} nat {m['wmae_natural']:.3f} | min {m['wmae_min']:.3f} mid {m['wmae_mid']:.3f} "
            f"heavy {m['wmae_heavy']:.3f} | S {m['mae_sbp']:.2f} D {m['mae_dbp']:.2f} | {b} | {m.get('seconds', 0)}s")
