"""D2 (continued): prior correction with p(y | prediction) from out-of-fold pairs.

Leave-one-subject-out conditional densities (duplicate recordings excluded
together), then 2-fold cross-fitted choice of (k, bw, prior, tau, age term).
"""
import argparse
import sys
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
    age = ctx.S_tr.age.values
    g = validate.groups_of(ctx.S_tr.index.values)
    excl = g[:, None] == g[None, :]
    y_seg = ctx.m_tr[["sbp", "dbp"]].values
    preds = {}
    for k in (60, 120, 200):
        for bw in (2.0, 4.0):
            for age_scale in (0.0, 0.3):
                pdfs = [shift.knn_conditional_pdf(mu[:, j], Y[:, j], mu[:, j], k=k, bw=bw,
                                                  extra_fit=age if age_scale else None,
                                                  extra_eval=age if age_scale else None,
                                                  extra_scale=age_scale, exclude=excl) for j in range(2)]
                for prior in ("min", "mid", "heavy"):
                    rs, rd = shift.strata_ratios(ctx.m_tr, prior)
                    for tau in (0.0, 0.5, 1.0, 1.25):
                        preds[(k, bw, age_scale, prior, tau)] = shift.corrected_median(None, None, rs, rd, tau=tau, pdfs=pdfs)
    terms = {key: validate.wmae_terms(ctx.to_segments(p), y_seg) for key, p in preds.items()}
    raw = validate.wmae_terms(ctx.to_segments(mu), y_seg)
    rows = ctx.m_tr.row.values
    rng = np.random.default_rng(0)
    scen = ["natural", "min", "mid", "heavy"]
    res = {"raw": {s: [] for s in scen}, "knn": {s: [] for s in scen}, "knn_tau0": {s: [] for s in scen}}
    picks = []
    for rep in range(40):
        half = rng.random(len(Y)) < 0.5
        fit, ev = half[rows], ~half[rows]
        best = min(terms, key=lambda kk: np.average(terms[kk][fit], weights=ctx.W["mid"][fit]))
        best0 = min([kk for kk in terms if kk[4] == 0.0], key=lambda kk: np.average(terms[kk][fit], weights=ctx.W["mid"][fit]))
        picks.append(best)
        for s in scen:
            res["raw"][s].append(np.average(raw[ev], weights=ctx.W[s][ev]))
            res["knn"][s].append(np.average(terms[best][ev], weights=ctx.W[s][ev]))
            res["knn_tau0"][s].append(np.average(terms[best0][ev], weights=ctx.W[s][ev]))
    print("Cross-fitted (40 half splits):")
    for fam, v in res.items():
        print(f"  {fam:9s} " + "  ".join(f"{s} {np.mean(x):.3f}±{np.std(x):.3f}" for s, x in v.items()))
    from collections import Counter
    print("most common picks:", Counter(picks).most_common(6))
    best = min(terms, key=lambda kk: np.average(terms[kk], weights=ctx.W["mid"]))
    print("best on all:", best)
    print(cv.fmt("knn best (all data)", validate.evaluate(ctx.to_segments(preds[best]), ctx.m_tr, ctx.W)))
    for key in [(120, 2.0, 0.0, "mid", 1.0), (120, 2.0, 0.0, "min", 1.0), (120, 2.0, 0.3, "mid", 1.0), (200, 4.0, 0.0, "mid", 1.0)]:
        if key in preds:
            print(cv.fmt(str(key), validate.evaluate(ctx.to_segments(preds[key]), ctx.m_tr, ctx.W)))


if __name__ == "__main__":
    main()
