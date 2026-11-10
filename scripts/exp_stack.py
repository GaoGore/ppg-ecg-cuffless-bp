"""Cross-fitted stacking of saved subject-level OOF predictions.

For each random half split: per-target non-negative linear stacking (with
intercept, plain MSE on the fit half) -> shift corrector fitted on the fit
half -> scored on the other half under every scenario. Compared on the same
splits with a fixed reference blend.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppgbp import cv, validate  # noqa: E402
from ppgbp.postprocess import ShiftConfig, ShiftCorrector  # noqa: E402

R = validate.RESULTS / "oof"
L = lambda n: np.load(R / f"{n}__subj.npy").mean(axis=0)            # noqa: E731
KEYS = ["natural", "min_bp", "min_dc", "mid_dc"]


def stack_fit(P, Y):
    return [LinearRegression(positive=True).fit(P[:, :, j], Y[:, j]) for j in range(2)]


def stack_apply(ms, P):
    return np.column_stack([ms[j].predict(P[:, :, j]) for j in range(2)])


def main():
    ctx = cv.CVContext("demo")
    nn_b2 = np.mean([L(f"b2_r5_s{s}") for s in (0, 1000, 2000)], axis=0)
    cands = {n: L(n) for n in ["r5_cat", "cat_core_dyn_r5", "cat_full_rmse", "cat_core_mae", "cat_core_d4", "cat_core_d6",
                               "cat_core_tailw0.25", "cat_core_agegap", "lin_core_300", "huber_core", "r5_nn", "cat_demo"]}
    cands["nn_b2"] = nn_b2
    sets = {
        "fixed reference (cat .7 + nn_b2 .3)": None,
        "stack: r5_cat + nn_b2": ["r5_cat", "nn_b2"],
        "stack: cat variants + nn_b2": ["r5_cat", "cat_core_dyn_r5", "cat_full_rmse", "cat_core_mae", "cat_core_d4", "nn_b2"],
        "stack: + linear/huber": ["r5_cat", "cat_core_dyn_r5", "cat_full_rmse", "cat_core_mae", "cat_core_d4", "nn_b2",
                                  "lin_core_300", "huber_core"],
        "stack: all": list(cands),
    }
    age = ctx.S_tr.age.values
    rows = ctx.m_tr.row.values
    y_seg = ctx.m_tr[["sbp", "dbp"]].values
    rng = np.random.default_rng(0)
    splits = [rng.random(len(ctx.Y)) < 0.5 for _ in range(30)]
    cfg = ShiftConfig()
    print(f"{'':34s}" + "".join(f"{k:>9s}" for k in KEYS) + "   (corrected, cross-fitted)")
    for name, members in sets.items():
        res = {k: [] for k in KEYS}
        for half in splits:
            if members is None:
                mu = 0.7 * cands["r5_cat"] + 0.3 * nn_b2
            else:
                P = np.stack([cands[m] for m in members], axis=1)          # (n, M, 2)
                mu = stack_apply(stack_fit(P[half], ctx.Y[half]), P)
            corr = ShiftCorrector(cfg).fit(mu[half], ctx.Y[half], ctx.m_tr, ctx.m_te, age=age[half])
            adj = mu.copy()
            adj[~half] = corr.transform(mu[~half], age=age[~half])
            e = validate.wmae_terms(ctx.to_segments(adj), y_seg)
            ev = ~half[rows]
            for k in KEYS:
                res[k].append(np.average(e[ev], weights=ctx.W[k][ev]))
        print(f"{name:34s}" + "".join(f"{np.mean(v):9.3f}" for v in res.values()))
    P = np.stack([cands[m] for m in sets["stack: all"]], axis=1)
    ms = stack_fit(P, ctx.Y)
    for j, t in enumerate(["SBP", "DBP"]):
        w = dict(zip(sets["stack: all"], ms[j].coef_))
        print(f"weights on all data, {t}: " + ", ".join(f"{k} {v:.2f}" for k, v in w.items() if v > 0.005)
              + f"  (intercept {ms[j].intercept_:.1f})")


if __name__ == "__main__":
    main()
