"""D2 experiments: shift correction on top of saved subject-level OOF predictions.

Compares, under every scenario, (a) raw predictions, (b) global stretch,
(c) Bayes prior correction using either the model's own quantiles or an
empirical residual distribution around point predictions. Correction
hyper-parameters are chosen by 2-fold cross-fitting over subjects (repeated),
so the reported gains are not tuned on the evaluated subjects.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppgbp import cv, shift, validate  # noqa: E402

R = validate.RESULTS / "oof"


def seg_metrics(ctx, subj_pred, mask=None):
    seg = ctx.to_segments(subj_pred)
    W = ctx.W if mask is None else {k: w * mask[ctx.m_tr.row.values] for k, w in ctx.W.items()}
    return validate.evaluate(seg, ctx.m_tr, W)


def residual_pdfs(mu, resid, hetero, fit_mask):
    """Predictive pdfs per target from residuals of the fit subset."""
    pdfs = []
    for j in range(2):
        r = resid[fit_mask, j]
        if hetero:
            # residual spread as a function of the predicted level (binned MAD)
            bins = np.quantile(mu[fit_mask, j], np.linspace(0, 1, 6))
            b_fit = np.clip(np.searchsorted(bins, mu[fit_mask, j]) - 1, 0, 4)
            mad = np.array([np.median(np.abs(r[b_fit == b])) for b in range(5)])
            s_all = np.interp(mu[:, j], (bins[:-1] + bins[1:]) / 2, mad)
            s_fit = s_all[fit_mask]
            pdfs.append(shift.residual_pdf(mu[:, j], r / s_fit * np.median(mad), scale=s_all / np.median(mad)))
        else:
            pdfs.append(shift.residual_pdf(mu[:, j], r))
    return pdfs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", default="cat_core_rmse")
    ap.add_argument("--quant", default="catq_core")
    args = ap.parse_args()
    ctx = cv.CVContext("core")
    Y = ctx.Y
    mu = np.load(R / f"{args.point}__subj.npy").mean(axis=0)
    resid = Y - mu
    print(cv.fmt(f"{args.point} raw", seg_metrics(ctx, mu)))
    # (b) global stretch around the training mean, cross-fitted
    center = Y.mean(axis=0)
    rng = np.random.default_rng(0)
    grid_k = [(a, b) for a in np.arange(1.0, 1.61, 0.05) for b in np.arange(1.0, 2.01, 0.05)]
    # candidate correction configs
    configs = []
    for prior in ["min", "mid", "heavy"]:
        for tau in [0.25, 0.5, 0.75, 1.0]:
            for sm in [0.0, 4.0]:
                configs.append((prior, tau, sm))
    # precompute predictive pdfs: residual-based (homo / hetero) and quantile-based
    pdf_sets = {"resid": residual_pdfs(mu, resid, False, np.ones(len(Y), bool)),
                "resid_het": residual_pdfs(mu, resid, True, np.ones(len(Y), bool))}
    qpath = R / f"{args.quant}__oofq.npy"
    if qpath.exists():
        Q = np.sort(np.load(qpath).mean(axis=0), axis=2)
        alphas = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98])
        pdf_sets["quantile"] = [shift.quantiles_to_pdf(Q[:, j, :], alphas) for j in range(2)]
        print(cv.fmt(f"{args.quant} median raw", seg_metrics(ctx, Q[:, :, 6])))
    preds = {}
    for src, pdfs in pdf_sets.items():
        for prior, tau, sm in configs:
            rs, rd = shift.strata_ratios(ctx.m_tr, prior)
            preds[(src, prior, tau, sm)] = shift.corrected_median(None, None, rs, rd, tau=tau, smooth_sd=sm, pdfs=pdfs)
    for (a, b) in grid_k:
        preds[("stretch", a, b, 0)] = shift.stretch(mu, center, np.array([a, b]))
    # full-data table for a few readable settings
    for key in [("resid", "mid", 1.0, 0.0), ("resid", "mid", 0.5, 0.0), ("resid_het", "mid", 1.0, 0.0),
                ("resid_het", "mid", 0.5, 0.0), ("resid", "min", 1.0, 0.0), ("resid", "heavy", 1.0, 0.0)] + \
               ([("quantile", "mid", 1.0, 0.0), ("quantile", "mid", 0.5, 0.0), ("quantile", "min", 1.0, 0.0)]
                if "quantile" in pdf_sets else []):
        print(cv.fmt(" ".join(map(str, key)), seg_metrics(ctx, preds[key])))
    # cross-fitted selection on the 'mid' scenario (and robustness on min/heavy)
    subj_w = {k: np.bincount(ctx.m_tr.row.values, weights=w, minlength=len(Y)) for k, w in ctx.W.items()}
    seg_terms = {}
    for key, p in preds.items():
        e = validate.wmae_terms(ctx.to_segments(p), ctx.m_tr[["sbp", "dbp"]].values)
        seg_terms[key] = e
    rows = ctx.m_tr.row.values
    families = sorted({k[0] for k in preds})
    res = {f: {s: [] for s in ["natural", "min", "mid", "heavy"]} for f in families + ["raw"]}
    raw_e = validate.wmae_terms(ctx.to_segments(mu), ctx.m_tr[["sbp", "dbp"]].values)
    for rep in range(30):
        half = rng.random(len(Y)) < 0.5
        fit_seg, ev_seg = half[rows], ~half[rows]
        for fam in families:
            keys = [k for k in preds if k[0] == fam]
            best = min(keys, key=lambda k: np.average(seg_terms[k][fit_seg], weights=ctx.W["mid"][fit_seg]))
            for s in res[fam]:
                res[fam][s].append(np.average(seg_terms[best][ev_seg], weights=ctx.W[s][ev_seg]))
        for s in res["raw"]:
            res["raw"][s].append(np.average(raw_e[ev_seg], weights=ctx.W[s][ev_seg]))
    print("\nCross-fitted (choose on half, evaluate on other half; mean over 30 splits):")
    for fam in ["raw"] + families:
        print(f"  {fam:10s} " + "  ".join(f"{s} {np.mean(v):.3f}" for s, v in res[fam].items()))
    # chosen settings on all data
    for fam in families:
        keys = [k for k in preds if k[0] == fam]
        best = min(keys, key=lambda k: np.average(seg_terms[k], weights=ctx.W["mid"]))
        print(f"  best-on-all {fam}: {best}")


if __name__ == "__main__":
    main()
