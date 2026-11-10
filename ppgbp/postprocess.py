"""Final transform: shift-aware posterior medians around point predictions.

Fit on out-of-fold subject-level (prediction, truth) pairs: a residual KDE
standardised by a prediction-level-dependent MAD. Apply to any subject-level
point predictions: posterior median of p(y | prediction) * w(y)^tau, where w
holds the AAMI-scenario selection factors per BP stratum.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import shift, validate


@dataclass
class ShiftConfig:
    scenario: str = "mid"
    factors: str = "bponly"        # "bponly" (BP strata only) or "rake" (with test demographic margins)
    tau: float = 1.0
    smooth_sd: float = 4.0
    n_bins: int = 5
    scale_model: str = "loglin"     # "level" (binned MAD by prediction), "loglin" (log|resid| ~ prediction + age),
                                    # "split" (separate log-linear spreads above / below the prediction)
    split_covs: tuple = ()          # extra covariates for the split spreads, from: "male", "med_his", "bmi"


class ShiftCorrector:
    def __init__(self, cfg: ShiftConfig = ShiftConfig()):
        self.cfg = cfg

    def _loglin_X(self, mu, age, j):
        return np.column_stack([np.ones(len(mu)), (mu - self.mu_c[j]) / self.mu_s[j], (age - 40.0) / 20.0])

    def _split_X(self, mu, age, j, covs):
        cols = [np.ones(len(mu)), (mu - self.mu_c[j]) / self.mu_s[j], (age - 40.0) / 20.0]
        for c in self.cfg.split_covs:
            v = np.asarray(covs[c], float)
            cols.append((v - self.cov_c[c]) / self.cov_s[c])
        return np.column_stack(cols)

    def fit(self, mu_oof: np.ndarray, y: np.ndarray, m_tr, m_te, age: np.ndarray | None = None, covs: dict | None = None):
        cfg = self.cfg
        self.resid, self.centers, self.mad = [], [], []
        if cfg.scale_model == "split":
            self.mu_c, self.mu_s = mu_oof.mean(axis=0), mu_oof.std(axis=0)
            self.cov_c = {c: np.nanmean(covs[c]) for c in cfg.split_covs}
            self.cov_s = {c: np.nanstd(covs[c]) + 1e-9 for c in cfg.split_covs}
            self.b_pos, self.b_neg, self.z = [], [], []
            for j in range(2):
                r = y[:, j] - mu_oof[:, j]
                X = np.nan_to_num(self._split_X(mu_oof[:, j], age, j, covs))
                pos = r >= 0
                bp = np.linalg.lstsq(X[pos], np.log(r[pos] + 0.5), rcond=None)[0]
                bn = np.linalg.lstsq(X[~pos], np.log(-r[~pos] + 0.5), rcond=None)[0]
                sp, sn = np.exp(X @ bp), np.exp(X @ bn)
                self.b_pos.append(bp)
                self.b_neg.append(bn)
                self.z.append(np.where(pos, r / sp, r / sn))
            self._set_factors(m_tr, m_te)
            return self
        if cfg.scale_model == "loglin":
            self.mu_c, self.mu_s = mu_oof.mean(axis=0), mu_oof.std(axis=0)
            self.beta = []
            for j in range(2):
                r = y[:, j] - mu_oof[:, j]
                X = self._loglin_X(mu_oof[:, j], age, j)
                b = np.linalg.lstsq(X, np.log(np.abs(r) + 0.5), rcond=None)[0]
                s = np.exp(X @ b)
                self.beta.append(b)
                self.resid.append(r / s * np.median(s))
                self.mad.append(np.median(s))
            self._set_factors(m_tr, m_te)
            return self
        for j in range(2):
            m, r = mu_oof[:, j], y[:, j] - mu_oof[:, j]
            edges = np.quantile(m, np.linspace(0, 1, cfg.n_bins + 1))
            b = np.clip(np.searchsorted(edges, m) - 1, 0, cfg.n_bins - 1)
            mad = np.array([np.median(np.abs(r[b == i])) for i in range(cfg.n_bins)])
            centers = np.array([np.median(m[b == i]) for i in range(cfg.n_bins)])
            s = np.interp(m, centers, mad)
            self.resid.append(r / s * np.median(mad))
            self.centers.append(centers)
            self.mad.append(mad)
        self._set_factors(m_tr, m_te)
        return self

    def _set_factors(self, m_tr, m_te):
        cfg = self.cfg
        if cfg.factors == "rake":
            self.fs, self.fd = validate.bp_factors(m_tr, m_te, cfg.scenario)
        else:
            rs, rd = shift.strata_ratios(m_tr, cfg.scenario)
            self.fs, self.fd = rs / rs[1], rd / rd[1]

    def transform(self, mu: np.ndarray, age: np.ndarray | None = None, covs: dict | None = None) -> np.ndarray:
        pdfs = []
        for j in range(2):
            if self.cfg.scale_model == "split":
                X = np.nan_to_num(self._split_X(mu[:, j], age, j, covs))
                pdfs.append(shift.residual_pdf_split(mu[:, j], self.z[j], np.exp(X @ self.b_pos[j]), np.exp(X @ self.b_neg[j])))
                continue
            if self.cfg.scale_model == "loglin":
                s = np.exp(self._loglin_X(mu[:, j], age, j) @ self.beta[j]) / self.mad[j]
            else:
                s = np.interp(mu[:, j], self.centers[j], self.mad[j]) / np.median(self.mad[j])
            pdfs.append(shift.residual_pdf(mu[:, j], self.resid[j], scale=s))
        return shift.corrected_median(None, None, self.fs, self.fd, tau=self.cfg.tau,
                                      smooth_sd=self.cfg.smooth_sd, pdfs=pdfs)


def crossfit_eval(ctx, mu_oof: np.ndarray, cfg: ShiftConfig = ShiftConfig(), reps: int = 40, seed: int = 0,
                  seg_delta: np.ndarray | None = None) -> dict:
    """Honest estimate of post-correction scenario wMAE: the corrector is fitted
    on a random half of the subjects and evaluated on the other half."""
    rng = np.random.default_rng(seed)
    rows = ctx.m_tr.row.values
    y_seg = ctx.m_tr[["sbp", "dbp"]].values
    out = {k: [] for k in ctx.W}
    raw = {k: [] for k in ctx.W}
    e_raw = validate.wmae_terms(ctx.to_segments(mu_oof, seg_delta=seg_delta), y_seg)
    for _ in range(reps):
        half = rng.random(len(ctx.Y)) < 0.5
        age = ctx.S_tr.age.values
        covs = {c: ctx.S_tr[c].values.astype(float) for c in ("male", "med_his", "bmi")}
        corr = ShiftCorrector(cfg).fit(mu_oof[half], ctx.Y[half], ctx.m_tr, ctx.m_te, age=age[half],
                                       covs={c: v[half] for c, v in covs.items()})
        adj = mu_oof.copy()
        adj[~half] = corr.transform(mu_oof[~half], age=age[~half], covs={c: v[~half] for c, v in covs.items()})
        e = validate.wmae_terms(ctx.to_segments(adj, seg_delta=seg_delta), y_seg)
        ev = ~half[rows]
        for k, w in ctx.W.items():
            out[k].append(np.average(e[ev], weights=w[ev]))
            raw[k].append(np.average(e_raw[ev], weights=w[ev]))
    res = {f"cf_{k}": float(np.mean(v)) for k, v in out.items()}
    res.update({f"raw_{k}": float(np.mean(v)) for k, v in raw.items()})
    return res


def fmt_cf(name: str, r: dict) -> str:
    return (f"{name:34s} raw: nat {r['raw_natural']:.3f} min {r['raw_min']:.3f} mid {r['raw_mid']:.3f} heavy {r['raw_heavy']:.3f}"
            f"  ||  corrected: nat {r['cf_natural']:.3f} min {r['cf_min']:.3f} mid {r['cf_mid']:.3f} heavy {r['cf_heavy']:.3f}")
