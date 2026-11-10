"""D2: joint (SBP, DBP) weighted-kNN posterior with demographic-aware selection factors."""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppgbp import cv, shift, validate  # noqa: E402

R = validate.RESULTS / "oof"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", default="cat_core_rmse")
    args = ap.parse_args()
    ctx = cv.CVContext("core")
    Y = ctx.Y
    mu = np.load(R / f"{args.point}__subj.npy").mean(axis=0)
    g = validate.groups_of(ctx.S_tr.index.values)
    excl = g[:, None] == g[None, :]
    y_seg = ctx.m_tr[["sbp", "dbp"]].values
    scale = (Y - mu).std(axis=0)
    fac = {}
    for sc in ("min", "mid", "heavy"):
        fac[("rake", sc)] = validate.bp_factors(ctx.m_tr, ctx.m_te, sc)
        rs, rd = shift.strata_ratios(ctx.m_tr, sc)
        fac[("bponly", sc)] = (rs / rs[1], rd / rd[1])
        print(sc, "raked factors SBP", np.round(fac[("rake", sc)][0], 2), "DBP", np.round(fac[("rake", sc)][1], 2),
              "| BP-only", np.round(fac[("bponly", sc)][0], 2), np.round(fac[("bponly", sc)][1], 2))
    preds = {}
    for (kind, sc), (fs, fd) in fac.items():
        for tau in (0.5, 0.75, 1.0, 1.25):
            for k in (80, 150):
                for bw in (2.0, 4.0):
                    preds[(kind, sc, tau, k, bw)] = shift.joint_knn_posterior(mu, Y, mu, fs, fd, tau=tau, k=k, bw=bw,
                                                                             scale=scale, exclude=excl)
    terms = {key: validate.wmae_terms(ctx.to_segments(p), y_seg) for key, p in preds.items()}
    raw = validate.wmae_terms(ctx.to_segments(mu), y_seg)
    rows = ctx.m_tr.row.values
    rng = np.random.default_rng(0)
    scen = ["natural", "min", "mid", "heavy"]
    fams = {"raw": None, "rake": "rake", "bponly": "bponly", "rake_mid_tau1": ("rake", "mid", 1.0)}
    res = {f: {s: [] for s in scen} for f in fams}
    picks = []
    for rep in range(40):
        half = rng.random(len(Y)) < 0.5
        fit, ev = half[rows], ~half[rows]
        for f, sel in fams.items():
            if sel is None:
                e = raw
            else:
                keys = [kk for kk in terms if (kk[0] == sel if isinstance(sel, str) else kk[:3] == sel)]
                best = min(keys, key=lambda kk: np.average(terms[kk][fit], weights=ctx.W["mid"][fit]))
                if f == "rake":
                    picks.append(best)
                e = terms[best]
            for s in scen:
                res[f][s].append(np.average(e[ev], weights=ctx.W[s][ev]))
    print("Cross-fitted (40 half splits):")
    for f, v in res.items():
        print(f"  {f:14s} " + "  ".join(f"{s} {np.mean(x):.3f}" for s, x in v.items()))
    print("rake picks:", Counter(picks).most_common(5))
    for key in [("rake", "mid", 1.0, 150, 2.0), ("rake", "mid", 0.75, 150, 2.0), ("rake", "min", 1.0, 150, 2.0),
                ("bponly", "mid", 1.0, 150, 2.0), ("rake", "heavy", 1.0, 150, 2.0)]:
        print(cv.fmt(str(key), validate.evaluate(ctx.to_segments(preds[key]), ctx.m_tr, ctx.W)))


if __name__ == "__main__":
    main()
