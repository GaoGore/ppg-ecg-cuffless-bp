"""D2: per-target posterior with level-dependent residual spread; BP-only vs
demographic-raked selection factors; cross-fitted choice of (scenario, tau, smoothing).
Residual distributions are re-estimated on the fit half only."""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppgbp import cv, shift, validate  # noqa: E402

R = validate.RESULTS / "oof"


def het_pdfs(mu, resid, fit, nb=5):
    """Residual KDE standardised by a level-dependent MAD (estimated on `fit`)."""
    out = []
    for j in range(2):
        m, r = mu[fit, j], resid[fit, j]
        edges = np.quantile(m, np.linspace(0, 1, nb + 1))
        b = np.clip(np.searchsorted(edges, m) - 1, 0, nb - 1)
        mad = np.array([np.median(np.abs(r[b == i])) for i in range(nb)])
        centers = np.array([np.median(m[b == i]) for i in range(nb)])
        s_all = np.interp(mu[:, j], centers, mad)
        M = np.median(mad)
        out.append(shift.residual_pdf(mu[:, j], r / s_all[fit] * M, scale=s_all / M))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", default="cat_core_rmse")
    ap.add_argument("--reps", type=int, default=40)
    args = ap.parse_args()
    ctx = cv.CVContext("core")
    Y = ctx.Y
    mu = np.load(R / f"{args.point}__subj.npy").mean(axis=0)
    resid = Y - mu
    y_seg = ctx.m_tr[["sbp", "dbp"]].values
    rows = ctx.m_tr.row.values
    fac = {}
    for sc in ("min", "mid", "heavy"):
        fac[("rake", sc)] = validate.bp_factors(ctx.m_tr, ctx.m_te, sc)
        rs, rd = shift.strata_ratios(ctx.m_tr, sc)
        fac[("bponly", sc)] = (rs / rs[1], rd / rd[1])
    grid = [(kind, sc, tau, sm) for (kind, sc) in fac for tau in (0.5, 0.75, 1.0, 1.25) for sm in (0.0, 4.0)]
    rng = np.random.default_rng(0)
    scen = ["natural", "min", "mid", "heavy"]
    raw = validate.wmae_terms(ctx.to_segments(mu), y_seg)
    res = {f: {s: [] for s in scen} for f in ("raw", "bponly", "rake", "both")}
    picks = {"bponly": [], "rake": [], "both": []}
    for rep in range(args.reps):
        half = rng.random(len(Y)) < 0.5
        pdfs = het_pdfs(mu, resid, half)                 # residual model from the fit half only
        terms = {}
        for (kind, sc, tau, sm) in grid:
            fs, fd = fac[(kind, sc)]
            p = shift.corrected_median(None, None, fs, fd, tau=tau, smooth_sd=sm, pdfs=pdfs)
            terms[(kind, sc, tau, sm)] = validate.wmae_terms(ctx.to_segments(p), y_seg)
        fit, ev = half[rows], ~half[rows]
        for s in scen:
            res["raw"][s].append(np.average(raw[ev], weights=ctx.W[s][ev]))
        for fam in ("bponly", "rake", "both"):
            keys = [k for k in terms if fam == "both" or k[0] == fam]
            best = min(keys, key=lambda k: np.average(terms[k][fit], weights=ctx.W["mid"][fit]))
            picks[fam].append(best)
            for s in scen:
                res[fam][s].append(np.average(terms[best][ev], weights=ctx.W[s][ev]))
    print(f"Cross-fitted ({args.reps} half splits), point model {args.point}:")
    for f, v in res.items():
        print(f"  {f:8s} " + "  ".join(f"{s} {np.mean(x):.3f}" for s, x in v.items()))
    for fam, p in picks.items():
        print(f"  picks {fam}:", Counter(p).most_common(4))


if __name__ == "__main__":
    main()
