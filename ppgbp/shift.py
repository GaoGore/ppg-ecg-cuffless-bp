"""Prediction under a target population's BP-distribution shift.

If p_train(y|x) is the model's predictive distribution, and the target population
was formed by selecting people according to their BP (quota-based, as in AAMI/ISO
81060-2 validation studies), then p_target(y|x) is proportional to p_train(y|x) * w(y)
with w(y) = p_target(y) / p_train(y). Under absolute error the optimal point
prediction is the median of that reweighted distribution.

w(y) is piecewise constant over the AAMI strata (share in target / share in
training), raised to a temperature tau in [0, 1] to hedge against an
over-confident prior.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from . import validate

SBP_CUTS = (100.0, 140.0, 160.0)
DBP_CUTS = (60.0, 85.0, 100.0)
GRID = np.arange(30.0, 230.0, 0.25)


def strata_ratios(m_tr, scenario: str = "mid") -> tuple[np.ndarray, np.ndarray]:
    ts, td = validate.SCENARIOS[scenario]
    sb = np.bincount(validate.sbp_stratum(m_tr.sbp), minlength=4) / len(m_tr)
    db = np.bincount(validate.dbp_stratum(m_tr.dbp), minlength=4) / len(m_tr)
    return np.asarray(ts) / sb, np.asarray(td) / db


def _weight_curve(ratios: np.ndarray, cuts, tau: float, smooth_sd: float) -> np.ndarray:
    # boundaries: SBP "<=100" is stratum 0, ">=140" starts stratum 2 ...; on a
    # 0.25 grid the inclusive/exclusive edge distinction is immaterial
    idx = np.searchsorted(np.asarray(cuts), GRID, side="left")
    lw = tau * np.log(ratios)[idx]
    if smooth_sd > 0:
        k = np.arange(-int(4 * smooth_sd / 0.25), int(4 * smooth_sd / 0.25) + 1) * 0.25
        g = np.exp(-0.5 * (k / smooth_sd) ** 2)
        lw = np.convolve(np.pad(lw, len(k) // 2, mode="edge"), g / g.sum(), mode="valid")
    return np.exp(lw)


def quantiles_to_pdf(q: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """(n, Q) sorted quantiles -> (n, len(GRID)) probability mass on GRID.
    Piecewise-linear CDF through the quantiles; linear tails reaching 0 and 1."""
    n, Q = q.shape
    a = np.asarray(alphas, float)
    pdf = np.empty((n, len(GRID)))
    for i in range(n):
        x = q[i].astype(float).copy()
        for j in range(1, Q):                                   # strictly increasing knots
            if x[j] <= x[j - 1]:
                x[j] = x[j - 1] + 1e-3
        lo_slope = (x[1] - x[0]) / (a[1] - a[0])
        hi_slope = (x[-1] - x[-2]) / (a[-1] - a[-2])
        xs = np.r_[x[0] - a[0] * lo_slope, x, x[-1] + (1 - a[-1]) * hi_slope]
        cs = np.r_[0.0, a, 1.0]
        cdf = np.interp(GRID, xs, cs, left=0.0, right=1.0)
        pdf[i] = np.diff(np.r_[0.0, cdf])
    return pdf


def weighted_median(pdf: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    p = pdf if w is None else pdf * w[None, :]
    c = np.cumsum(p, axis=1)
    c /= c[:, -1:]
    j = (c < 0.5).sum(axis=1)
    j = np.clip(j, 1, len(GRID) - 1)
    c0, c1 = c[np.arange(len(c)), j - 1], c[np.arange(len(c)), j]
    frac = np.where(c1 > c0, (0.5 - c0) / np.where(c1 > c0, c1 - c0, 1), 0.5)
    return GRID[j - 1] + frac * (GRID[j] - GRID[j - 1])


def corrected_median(quant: np.ndarray, alphas, ratios_s, ratios_d, tau=1.0, smooth_sd=0.0,
                     pdfs=None) -> np.ndarray:
    """quant: (n, 2, Q) -> (n, 2) posterior medians under the reweighted prior."""
    n = pdfs[0].shape[0] if pdfs is not None else quant.shape[0]
    out = np.empty((n, 2))
    for j, (ratios, cuts) in enumerate(((ratios_s, SBP_CUTS), (ratios_d, DBP_CUTS))):
        pdf = pdfs[j] if pdfs is not None else quantiles_to_pdf(quant[:, j, :], alphas)
        out[:, j] = weighted_median(pdf, _weight_curve(ratios, cuts, tau, smooth_sd))
    return out


def residual_pdf(mu: np.ndarray, resid: np.ndarray, scale: np.ndarray | None = None, bw: float = 1.5) -> np.ndarray:
    """Predictive mass on GRID for point predictions mu: mu_i + s_i * r, where r
    follows a Gaussian-KDE of (standardised) out-of-fold residuals."""
    d = np.arange(-90.0, 90.25, 0.25)
    r = np.asarray(resid, float)
    hist = np.histogram(r, bins=np.r_[d - 0.125, d[-1] + 0.125])[0].astype(float)
    k = np.arange(-int(4 * bw / 0.25), int(4 * bw / 0.25) + 1) * 0.25
    g = np.exp(-0.5 * (k / bw) ** 2)
    dens = np.convolve(hist, g / g.sum(), mode="same")
    dens /= dens.sum()
    s = np.ones(len(mu)) if scale is None else np.asarray(scale, float)
    pdf = np.interp((GRID[None, :] - mu[:, None]) / s[:, None], d, dens, left=0.0, right=0.0) / s[:, None]
    return pdf / pdf.sum(axis=1, keepdims=True)


def knn_conditional_pdf(mu_fit: np.ndarray, y_fit: np.ndarray, mu_eval: np.ndarray, k: int = 80,
                        bw: float = 2.0, extra_fit: np.ndarray | None = None,
                        extra_eval: np.ndarray | None = None, extra_scale: float = 1.0,
                        exclude: np.ndarray | None = None) -> np.ndarray:
    """p(y | prediction) estimated from out-of-fold (prediction, truth) pairs:
    for each evaluation point, a tricube-weighted Gaussian KDE over the k
    nearest fitted predictions. Captures regression to the mean and
    level-dependent spread. Optional extra conditioning variable (e.g. age)
    enters the distance after scaling."""
    d = np.abs(mu_eval[:, None] - mu_fit[None, :])
    if extra_fit is not None:
        d = np.sqrt(d ** 2 + (extra_scale * (extra_eval[:, None] - extra_fit[None, :])) ** 2)
    if exclude is not None:                                         # e.g. leave-one-subject-out
        d = np.where(exclude, np.inf, d)
    idx = np.argsort(d, axis=1)[:, :k]
    dk = np.take_along_axis(d, idx, axis=1)
    h = dk[:, -1:] + 1e-6
    wt = (1 - (dk / h) ** 3) ** 3
    ys = y_fit[idx]                                                 # (n_eval, k)
    pdf = np.zeros((len(mu_eval), len(GRID)))
    for j in range(k):
        pdf += wt[:, j:j + 1] * np.exp(-0.5 * ((GRID[None, :] - ys[:, j:j + 1]) / bw) ** 2)
    return pdf / pdf.sum(axis=1, keepdims=True)


def joint_knn_posterior(mu_fit: np.ndarray, y_fit: np.ndarray, mu_eval: np.ndarray, fs: np.ndarray,
                        fd: np.ndarray, tau: float = 1.0, k: int = 120, bw: float = 2.0,
                        scale: np.ndarray | None = None, exclude: np.ndarray | None = None) -> np.ndarray:
    """Posterior medians of SBP and DBP for each evaluation subject.

    Neighbours are the k fitted subjects whose (SBP, DBP) predictions are
    closest (distance standardised by `scale`); each neighbour's true (SBP, DBP)
    is a draw from p_train(y | prediction) with tricube weight, re-weighted by
    the selection factors fs[SBP stratum] * fd[DBP stratum] raised to tau.
    Medians are taken from the Gaussian-smoothed (bw) weighted samples.
    """
    scale = np.ones(2) if scale is None else np.asarray(scale, float)
    diff = (mu_eval[:, None, :] - mu_fit[None, :, :]) / scale
    d = np.sqrt((diff ** 2).sum(axis=2))
    if exclude is not None:
        d = np.where(exclude, np.inf, d)
    idx = np.argsort(d, axis=1)[:, :k]
    dk = np.take_along_axis(d, idx, axis=1)
    h = dk[:, -1:] + 1e-9
    kern = (1 - np.clip(dk / h, 0, 1) ** 3) ** 3
    ys = y_fit[idx]                                                    # (n, k, 2)
    sel = (fs[validate.sbp_stratum(ys[:, :, 0])] * fd[validate.dbp_stratum(ys[:, :, 1])]) ** tau
    wt = kern * sel
    wt = wt / wt.sum(axis=1, keepdims=True)
    out = np.empty((len(mu_eval), 2))
    for j in range(2):
        cdf = np.zeros((len(mu_eval), len(GRID)))
        for m in range(k):
            cdf += wt[:, m:m + 1] * ndtr((GRID[None, :] - ys[:, m, j:j + 1]) / bw)
        jj = np.clip((cdf < 0.5).sum(axis=1), 1, len(GRID) - 1)
        r = np.arange(len(mu_eval))
        c0, c1 = cdf[r, jj - 1], cdf[r, jj]
        out[:, j] = GRID[jj - 1] + np.where(c1 > c0, (0.5 - c0) / np.where(c1 > c0, c1 - c0, 1), 0.5) * 0.25
    return out


def residual_pdf_split(mu: np.ndarray, z: np.ndarray, s_pos: np.ndarray, s_neg: np.ndarray, bw: float = 0.15) -> np.ndarray:
    """Predictive mass on GRID with different spreads above and below the point
    prediction: standardised residuals z (r / s_side) share one KDE shape; a
    subject's density is f((y - mu) / s_side) / s_side with the side's scale."""
    zg = np.arange(-10.0, 10.0001, 0.02)
    hist = np.histogram(z, bins=np.r_[zg - 0.01, zg[-1] + 0.01])[0].astype(float)
    k = np.arange(-int(4 * bw / 0.02), int(4 * bw / 0.02) + 1) * 0.02
    g = np.exp(-0.5 * (k / bw) ** 2)
    dens = np.convolve(hist, g / g.sum(), mode="same")
    dens /= dens.sum() * 0.02
    d = GRID[None, :] - mu[:, None]
    s = np.where(d >= 0, s_pos[:, None], s_neg[:, None])
    pdf = np.interp(d / s, zg, dens, left=0.0, right=0.0) / s
    return pdf / pdf.sum(axis=1, keepdims=True)


def stretch(pred: np.ndarray, center: np.ndarray, k: np.ndarray) -> np.ndarray:
    return center + np.asarray(k) * (pred - center)
