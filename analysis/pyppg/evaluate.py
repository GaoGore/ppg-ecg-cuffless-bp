# Does pyPPG add information beyond our own features? Same CatBoost, same subject folds (5-fold x 3), subject-level
# pooling as ppgbp/tables.py (median over a subject's segments). Models:
#   demo            demographics only
#   core            our main feature set (demographics + consensus / cross-channel / ECG / rhythm / motion)
#   demo+pp_all     demographics + pyPPG, median over the 12 IR/R/G channels (102 biomarkers x median/IQR)
#   demo+pp_wl      demographics + pyPPG per wavelength IR / R / G (median over the 4 positions)
#   core+pp_all, core+pp_wl
#   demo+pp_shuf / core+pp_shuf   CONTROL: the same pyPPG table with subject rows permuted (seed 0) -> what adding
#                                 that many columns at rsm 0.2 does by itself (dilution); the pyPPG effect is vs this
# Reported: subject-level OOF R^2 (all / ID<1000 / ID>=1000), segment wMAE natural and target-distribution scenario weights,
# paired subject bootstrap of the differences that matter, also within ID<1000; dc weights also without 0665 / top-3.
# Plus vs the reference blend (0.5 r5_cat + 0.25 template nets + 0.25 raw nets, OOF from results/oof):
# pyPPG partial correlation with the blend's residuals in ID<1000 vs a permutation threshold, a cross-fitted residual
# model, and blend + 0.25 x (core+pp_all) vs blend + 0.25 x core.
#
# PRE-DECLARED READING (written 2026-11-10 before any model was fitted):
#   primary   ID<1000 natural wMAE, core+pp_all - core+pp_shuf: 95% CI below 0 -> pyPPG adds information beyond
#             our core features in the younger recruitment cohort; CI covering 0 -> no detectable information.
#   material  strict rule of the project: CI below 0 on BOTH min_dc and mid_dc and point <= -0.2 -> would matter
#             under the target BP distribution; CI below 0 but point > -0.2 -> real but small; dc results must survive dropping 0665 and the
#             top-3 weighted subjects, otherwise "undecided".
#   blend     residual max |rho| above the permutation 95% threshold AND blend+0.25(core+pp) better than blend+0.25 core
#             (CI below 0, natural) -> pyPPG would improve our reference point model.
#   Several views x weightings form a selection set: only the pre-declared cells above are read, not the best cell.
# Environment: pyPPG 1.0.73 core modules run on numpy 2.2.6 / pandas 2.2.3 / scipy 1.15.3 (not its 2022 pins;
# pandas 3 silently breaks its chained assignments), np.NaN re-added.
# usage: python evaluate.py [--fit] [model ...]   (fits missing models; OOF cached in analysis/pyppg/out/)
_ROOT = __import__('os').environ.get('PPGBP_ROOT') or str(__import__('pathlib').Path(__file__).resolve().parents[2])  # repository root
import sys, glob, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, _ROOT)
import numpy as np, pandas as pd
from pathlib import Path
from ppgbp import cv, validate, tables
from ppgbp.models.gbdt import CatModel
HERE = Path(__file__).resolve().parent; OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
REPO = HERE.parents[1]

def pyppg_views(ctx):
    """subject tables of the pyPPG views, aligned to ctx.S_tr / ctx.S_te"""
    P = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(REPO / "cache/pyppg/*.csv")))], ignore_index=True)
    chans = {w: [f"PPG_{w}_{i}" for i in (1, 2, 3, 4)] for w in ("IR", "R", "G", "B")}
    bs = [c.split("__", 1)[1] for c in P.columns if c.startswith("PPG_IR_1__") and not c.endswith("__nbeats")]
    A = np.stack([P[[f"{ch}__{b}" for b in bs]].to_numpy(float) for w in chans for ch in chans[w]], axis=1)   # (seg, 16, B)
    A[~np.isfinite(A)] = np.nan
    ix = {ch: i for i, ch in enumerate(c for w in chans for c in chans[w])}
    cols = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for w in ("IR", "R", "G"):
            M = np.nanmedian(A[:, [ix[c] for c in chans[w]]], axis=1)
            cols.update({f"pp_{w}__{b}": M[:, k] for k, b in enumerate(bs)})
        M = np.nanmedian(A[:, [ix[c] for w in ("IR", "R", "G") for c in chans[w]]], axis=1)
        cols.update({f"pp_all__{b}": M[:, k] for k, b in enumerate(bs)})
    D = pd.DataFrame(cols); D["seg_id"] = P.seg_id.values
    meta = pd.concat([ctx.m_tr[["seg_id", "sid"]], ctx.m_te[["seg_id", "sid"]]])
    D = D.merge(meta, on="seg_id", how="inner", validate="one_to_one")
    S = D.drop(columns="seg_id").groupby("sid").median()
    S.replace([np.inf, -np.inf], np.nan, inplace=True)
    return S, [c for c in S.columns if c.startswith("pp_all__")], [c for c in S.columns if c.startswith(("pp_IR__", "pp_R__", "pp_G__"))]

def main():
    argv = sys.argv[1:]; fit = "--fit" in argv
    ctx = cv.CVContext("core")
    S, pp_all, pp_wl = pyppg_views(ctx)
    miss = [s for s in ctx.S_tr.index.tolist() + ctx.S_te.index.tolist() if s not in S.index]
    print(f"pyPPG subject table {S.shape}; subjects without pyPPG rows: {len(miss)}")
    ctx.S_tr = ctx.S_tr.join(S); ctx.S_te = ctx.S_te.join(S)
    rng = np.random.default_rng(0); sh = {}
    for SS in (ctx.S_tr, ctx.S_te):                       # control: pyPPG rows permuted among subjects of the same split
        V = SS[pp_all].to_numpy()[rng.permutation(len(SS))]
        for k, c in enumerate(pp_all): SS[c.replace("pp_all__", "ppshuf__")] = V[:, k]
    pp_shuf = [c.replace("pp_all__", "ppshuf__") for c in pp_all]
    core = ctx.cols; demo = tables.DEMO
    MODELS = {"demo": demo, "core": core, "demo+pp_all": demo + pp_all, "demo+pp_wl": demo + pp_wl,
              "core+pp_all": core + pp_all, "core+pp_wl": core + pp_wl,
              "demo+pp_shuf": demo + pp_shuf, "core+pp_shuf": core + pp_shuf}
    want = [a for a in argv if a in MODELS] or list(MODELS)
    if miss:                                              # extraction not finished: only the models without pyPPG columns
        want = [a for a in want if "pp" not in a]; print(f"incomplete pyPPG table -> only {want}")
    res = {}
    for name in want:
        f = OUT / f"{name}.npz"
        if f.exists():
            z = np.load(f); res[name] = {k: z[k] for k in z.files}; continue
        if not fit: continue
        t = time.time()
        r = cv.run(ctx, name, lambda seed: CatModel(seed=seed), cols=MODELS[name], log=False, verbose=False)
        np.savez(f, oof=r["oof"], test=r["test"], oof_subj=r["oof_subj"], oof_rep=r["oof_rep"])
        res[name] = {"oof": r["oof"], "test": r["test"], "oof_subj": r["oof_subj"], "oof_rep": r["oof_rep"]}
        print(f"fitted {name} ({len(MODELS[name])} cols) in {time.time() - t:.0f}s", flush=True)
    report(ctx, res)
    sanity(ctx, res)
    if not miss: vs_blend(ctx, res, pp_all, pp_wl)

def sanity(ctx, res):
    """the harness should reproduce the saved cat_core_rmse run (scripts/exp_base.py): same OOF natural wMAE"""
    if "core" not in res: return
    e = validate.evaluate(res["core"]["oof"], ctx.m_tr, {"natural": ctx.W["natural"]})["wmae_natural"]
    f = validate.RESULTS / "oof" / "cat_core_rmse__subj.npy"
    if f.exists():
        ref = validate.evaluate(ctx.to_segments(np.load(f).mean(0)), ctx.m_tr, {"natural": ctx.W["natural"]})["wmae_natural"]
        print(f"\nsanity: core OOF natural wMAE {e:.3f} vs saved cat_core_rmse run {ref:.3f}")

def blend_oof(ctx):
    L = lambda ns: np.mean([np.load(validate.RESULTS / "oof" / f"{n}__subj.npy").mean(0) for n in ns], 0)
    tm = [f"b2_r5_s{s}" for s in (0, 1000, 2000)] + [f"b4_r5_s{s}" for s in (3000, 4000, 5000, 6000)] + [f"b5_r5_s{s}" for s in (0, 1000, 2000, 3000, 4000, 5000, 6000)]
    return 0.5 * L(["r5_cat"]) + 0.25 * L(tm) + 0.25 * L([f"raw_r5_s{s}" for s in (0, 1000, 2000, 3000)])

def vs_blend(ctx, res, pp_all, pp_wl, n_perm=500):
    from scipy.stats import spearmanr, rankdata
    B = blend_oof(ctx); Y = ctx.S_tr[["sbp", "dbp"]].values; young = ctx.S_tr.index.astype(int).values < 1000
    e = validate.evaluate(ctx.to_segments(B), ctx.m_tr, {"natural": ctx.W["natural"]})["wmae_natural"]
    print(f"\nreference blend OOF natural wMAE {e:.3f}; R2 SBP/DBP ID<1000 {np.round(r2(Y[young], B[young]), 3)}")
    X = ctx.S_tr.loc[young, pp_wl + pp_all]; X = X.loc[:, X.notna().mean() > 0.9]
    Xr = np.column_stack([rankdata(X[c].fillna(X[c].median())) for c in X.columns]); Xr = (Xr - Xr.mean(0)) / Xr.std(0)
    rng = np.random.default_rng(1)
    for j, lab in ((0, "SBP"), (1, "DBP")):
        rr = rankdata(Y[young, j] - B[young, j]); rr = (rr - rr.mean()) / rr.std()
        rho = np.abs(Xr.T @ rr) / len(rr); top = np.argsort(-rho)[:3]
        null = [np.max(np.abs(Xr.T @ rng.permutation(rr)) / len(rr)) for _ in range(n_perm)]
        print(f"  blend residual {lab} (ID<1000, {X.shape[1]} pyPPG cols): max |rho| {rho.max():.3f} vs permutation 95% {np.percentile(null, 95):.3f}; "
              f"top: " + ", ".join(f"{X.columns[k]} {rho[k]:.3f}" for k in top))
    # cross-fitted residual model on pyPPG only (subject folds of repeat 0), ID<1000
    f = ctx.subj_folds[0][young]; R = Y[young] - B[young]; P = np.zeros_like(R)
    for k in range(ctx.n_splits):
        tr, va = f != k, f == k
        m = CatModel(seed=k).fit(X.values[tr], R[tr]); P[va] = m.predict(X.values[va])
    print(f"  cross-fitted pyPPG residual model, OOF R2 on blend residuals SBP/DBP: {np.round(r2(R, P), 3)}")
    for name in ("core+pp_all", "core+pp_wl"):
        if name not in res or "core" not in res: continue
        a = ctx.to_segments(0.75 * B + 0.25 * res[name]["oof_subj"]); b = ctx.to_segments(0.75 * B + 0.25 * res["core"]["oof_subj"])
        ys = (ctx.m_tr.sid.astype(int).values < 1000).astype(float)
        out = [f"{lab} {d:+.3f} [{lo:+.3f},{hi:+.3f}]" for lab, w in (("natural", ctx.W["natural"]), ("ID<1000", ctx.W["natural"] * ys), ("min_dc", ctx.W["min_dc"]), ("mid_dc", ctx.W["mid_dc"]))
               for d, lo, hi in [validate.paired_bootstrap(a, b, ctx.m_tr, w)]]
        print(f"  blend+0.25({name}) - blend+0.25(core): " + " | ".join(out))

def r2(y, p):
    return 1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)

def report(ctx, res):
    Y = ctx.S_tr[["sbp", "dbp"]].values; young = ctx.S_tr.index.astype(int).values < 1000
    yseg = (ctx.m_tr.sid.astype(int).values < 1000).astype(float)
    rows = []
    for name, r in res.items():
        p = r["oof_subj"]; e = validate.evaluate(r["oof"], ctx.m_tr, {k: ctx.W[k] for k in ("natural", "mid_dc", "wide_dc")})
        a, y_, o = r2(Y, p), r2(Y[young], p[young]), r2(Y[~young], p[~young])
        rows.append({"model": name, "R2 SBP": a[0], "R2 DBP": a[1], "R2 SBP <1000": y_[0], "R2 DBP <1000": y_[1],
                     "R2 SBP >=1000": o[0], "R2 DBP >=1000": o[1], "wMAE nat": e["wmae_natural"], "wMAE mid_dc": e["wmae_mid_dc"],
                     "wMAE wide_dc": e["wmae_wide_dc"], "MAE S": e["mae_sbp"], "MAE D": e["mae_dbp"]})
    T = pd.DataFrame(rows).set_index("model")
    pd.set_option("display.width", 250)
    print("\nsubject-level OOF R^2 and segment wMAE (lower = better)\n" + T.round(3).to_string())
    print("\npaired subject bootstrap, wMAE(a) - wMAE(b) [95% CI] (negative = a better)")
    sid = ctx.m_tr.sid.values
    def drop(w, k):                                   # dc weights without 0665 (k=0) or without the k largest-weight subjects
        w = w.copy()
        if k == 0: w[sid == "0665"] = 0; return w
        top = pd.Series(w).groupby(sid).sum().nlargest(k).index; w[np.isin(sid, top)] = 0; return w
    for a, b in (("core+pp_all", "core+pp_shuf"), ("core+pp_wl", "core+pp_shuf"), ("core+pp_shuf", "core"),
                 ("demo+pp_all", "demo+pp_shuf"), ("demo+pp_wl", "demo+pp_shuf"), ("demo+pp_all", "core"), ("core", "demo")):
        if a not in res or b not in res: continue
        out = []
        for lab, w in (("natural", ctx.W["natural"]), ("ID<1000", ctx.W["natural"] * yseg), ("min_dc", ctx.W["min_dc"]), ("mid_dc", ctx.W["mid_dc"]),
                       ("mid_dc-0665", drop(ctx.W["mid_dc"], 0)), ("mid_dc-top3", drop(ctx.W["mid_dc"], 3))):
            d, lo, hi = validate.paired_bootstrap(res[a]["oof"], res[b]["oof"], ctx.m_tr, w)
            out.append(f"{lab} {d:+.3f} [{lo:+.3f},{hi:+.3f}]")
        print(f"  {a:12s} - {b:12s}: " + " | ".join(out))
    T.to_csv(OUT / "summary.csv")

if __name__ == "__main__":
    main()
