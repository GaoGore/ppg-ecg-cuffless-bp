"""Subject-grouped CV and target-distribution scenario weighting.

The target population follows an AAMI/ISO 81060-2-style BP distribution (quota-based), so the plain CV
average over the training distribution understates errors at the BP extremes. Each scenario reweights
training segments (raking / iterative proportional fitting) so that the weighted label distribution meets
a set of SBP/DBP stratum shares and the demographic margins (age band, sex, medication) of the unlabeled
hold-out inputs.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import io

RESULTS = io.PROJECT_DIR / "results"
DUPLICATE_GROUPS = {"0729": "0724"}          # identical recordings and labels

SBP_EDGES = ("<=100", "100-140", "140-160", ">=160")
DBP_EDGES = ("<=60", "60-85", "85-100", ">=100")
SCENARIOS = {
    #            SBP strata shares            DBP strata shares
    "min":   ([0.06, 0.74, 0.15, 0.05], [0.05, 0.75, 0.15, 0.05]),
    "mid":   ([0.08, 0.67, 0.18, 0.07], [0.07, 0.66, 0.20, 0.07]),
    "heavy": ([0.10, 0.55, 0.25, 0.10], [0.08, 0.57, 0.25, 0.10]),
    # Two-tailed: the quotas are minimums, so the low-BP share is kept as well; _lo / _hi bracket the
    # high tails (lower end = AAMI minimum 20%). Validation design only.
    "wide":    ([0.19, 0.55, 0.19, 0.07], [0.08, 0.65, 0.20, 0.07]),
    "wide_lo": ([0.19, 0.61, 0.15, 0.05], [0.08, 0.72, 0.15, 0.05]),
    "wide_hi": ([0.19, 0.43, 0.28, 0.10], [0.08, 0.57, 0.26, 0.09]),
}
PRIMARY = "mid"


def sbp_stratum(s):
    s = np.asarray(s)
    return np.select([s <= 100, s < 140, s < 160], [0, 1, 2], 3)


def dbp_stratum(d):
    d = np.asarray(d)
    return np.select([d <= 60, d < 85, d < 100], [0, 1, 2], 3)


def age_band(a):
    return np.digitize(np.asarray(a), [30, 45, 60], right=True)


def groups_of(sids) -> np.ndarray:
    return np.array([DUPLICATE_GROUPS.get(s, s) for s in sids])


def folds(meta_tr: pd.DataFrame, n_splits=5, repeats=3, seed=0) -> list[np.ndarray]:
    """Per repeat, a fold id for every training segment (subjects never split)."""
    subj = meta_tr.groupby("sid").agg(sbp=("sbp", "mean"), dbp=("dbp", "mean"))
    strat = sbp_stratum(subj.sbp) * 2 + (subj.dbp.values >= 85)
    g = groups_of(subj.index)
    out = []
    for r in range(repeats):
        sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        fold_of_subj = np.empty(len(subj), int)
        for k, (_, va) in enumerate(sgk.split(np.zeros(len(subj)), strat, g)):
            fold_of_subj[va] = k
        mp = dict(zip(subj.index, fold_of_subj))
        out.append(meta_tr.sid.map(mp).values)
    return out


def rake(margins, iters=300, return_factors=False):
    """Iterative proportional fitting. Optionally also returns, per margin, the
    cumulative multiplicative factor of each category (the weights are their
    product up to a constant)."""
    w = np.ones(len(margins[0][0]))
    factors = [np.ones(len(t)) for _, t in margins]
    for _ in range(iters):
        for m, (cat, target) in enumerate(margins):
            target = np.asarray(target, float)
            cur = np.bincount(cat, weights=w, minlength=len(target)) / w.sum()
            upd = target / np.maximum(cur, 1e-12)
            w *= upd[cat]
            factors[m] *= upd
    w = w / w.mean()
    return (w, factors) if return_factors else w


def bp_factors(meta_tr: pd.DataFrame, meta_te: pd.DataFrame, scenario: str) -> tuple[np.ndarray, np.ndarray]:
    """Selection factors for SBP and DBP strata implied by a scenario after
    joint raking with the hold-out demographic margins (normalised so the
    middle stratum is 1). These are the BP part of the density ratio
    p_target(y, demo) / p_train(y, demo)."""
    te_subj = meta_te.drop_duplicates("sid")
    ts, td = SCENARIOS[scenario]
    margins = [(sbp_stratum(meta_tr.sbp), ts), (dbp_stratum(meta_tr.dbp), td),
               (age_band(meta_tr.age), np.bincount(age_band(te_subj.age), minlength=4) / len(te_subj)),
               (meta_tr.male.astype(int).values, np.bincount(te_subj.male.astype(int), minlength=2) / len(te_subj)),
               (meta_tr.med_his.astype(int).values, np.bincount(te_subj.med_his.astype(int), minlength=2) / len(te_subj))]
    _, f = rake(margins, return_factors=True)
    return f[0] / f[0][1], f[1] / f[1][1]


def scenario_weights(meta_tr: pd.DataFrame, meta_te: pd.DataFrame) -> dict[str, np.ndarray]:
    """Segment weights for each scenario (natural = 1)."""
    te_subj = meta_te.drop_duplicates("sid")
    demo_margins = [
        (age_band(meta_tr.age), np.bincount(age_band(te_subj.age), minlength=4) / len(te_subj)),
        (meta_tr.male.astype(int).values, np.bincount(te_subj.male.astype(int), minlength=2) / len(te_subj)),
        (meta_tr.med_his.astype(int).values, np.bincount(te_subj.med_his.astype(int), minlength=2) / len(te_subj)),
    ]
    sb, db = sbp_stratum(meta_tr.sbp), dbp_stratum(meta_tr.dbp)
    W = {"natural": np.ones(len(meta_tr))}
    for name, (ts, td) in SCENARIOS.items():
        W[name] = rake([(sb, ts), (db, td)] + demo_margins)
    # BP quotas only, with no assumption about how the extra extremes relate to demographics
    for name, (ts, td) in SCENARIOS.items():
        W[name + "_bp"] = rake([(sb, ts), (db, td)])
    # + recruitment-cohort mix (subject-ID ranges) of the hold-out set. Used for validation
    # weighting only, never as a model input.
    coh_tr = cohort_of(meta_tr.sid)
    coh_margin = (coh_tr, np.bincount(cohort_of(te_subj.sid), minlength=len(COHORT_EDGES) + 1) / len(te_subj))
    for name, (ts, td) in SCENARIOS.items():
        W[name + "_dc"] = rake([(sb, ts), (db, td)] + demo_margins + [coh_margin])
    # finer design: 10-year age bands, BMI tertiles and 100-wide subject-ID bands of the hold-out set
    # (closer to the hold-out mix, fewer effective subjects). Validation weighting only.
    fine = fine_margins(meta_tr, te_subj)
    for name, (ts, td) in SCENARIOS.items():
        W[name + "_dcf"] = rake([(sb, ts), (db, td)] + fine)
    return W


COHORT_EDGES = (200, 400, 600, 800, 1000)
FINE_AGE_EDGES = (25, 35, 45, 55, 65)
FINE_COHORT_EDGES = tuple(range(100, 1001, 100))


def fine_margins(meta_tr: pd.DataFrame, te_subj: pd.DataFrame) -> list:
    def bmi(m):
        return np.asarray(m.weight, float) / (np.asarray(m.height, float) / 100) ** 2
    bmi_cuts = np.quantile(bmi(meta_tr.drop_duplicates("sid")), [1 / 3, 2 / 3])
    cats = [(lambda m: np.digitize(np.asarray(m.age), FINE_AGE_EDGES, right=True), len(FINE_AGE_EDGES) + 1),
            (lambda m: np.digitize(bmi(m), bmi_cuts), 3),
            (lambda m: np.asarray(m.male).astype(int), 2),
            (lambda m: np.asarray(m.med_his).astype(int), 2),
            (lambda m: np.digitize(np.asarray(m.sid).astype(int), FINE_COHORT_EDGES, right=True), len(FINE_COHORT_EDGES) + 1)]
    return [(f(meta_tr), np.bincount(f(te_subj), minlength=k) / len(te_subj)) for f, k in cats]


def cohort_of(sids) -> np.ndarray:
    return np.digitize(np.asarray(sids).astype(int), COHORT_EDGES, right=True)


def wmae_terms(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 0.6 * np.abs(pred[:, 0] - y[:, 0]) + 0.4 * np.abs(pred[:, 1] - y[:, 1])


def evaluate(pred: np.ndarray, meta_tr: pd.DataFrame, W: dict) -> dict:
    """Scenario wMAEs plus per-stratum bias/MAE (natural weights) for one OOF set."""
    y = meta_tr[["sbp", "dbp"]].values
    e = wmae_terms(pred, y)
    out = {f"wmae_{k}": float(np.average(e, weights=w)) for k, w in W.items()}
    out["mae_sbp"] = float(np.abs(pred[:, 0] - y[:, 0]).mean())
    out["mae_dbp"] = float(np.abs(pred[:, 1] - y[:, 1]).mean())
    sb, db = sbp_stratum(y[:, 0]), dbp_stratum(y[:, 1])
    for i, lab in enumerate(SBP_EDGES):
        m = sb == i
        out[f"bias_sbp{lab}"] = float((pred[m, 0] - y[m, 0]).mean())
    for i, lab in enumerate(DBP_EDGES):
        m = db == i
        out[f"bias_dbp{lab}"] = float((pred[m, 1] - y[m, 1]).mean())
    return out


def paired_bootstrap(pred_a, pred_b, meta_tr, w, n=2000, seed=0) -> tuple[float, float, float]:
    """Mean and 95% CI of wMAE(a) - wMAE(b) under weights w, resampling subjects."""
    y = meta_tr[["sbp", "dbp"]].values
    g = groups_of(meta_tr.sid.values)
    df = pd.DataFrame({"g": g, "w": w, "ea": w * wmae_terms(pred_a, y), "eb": w * wmae_terms(pred_b, y)})
    s = df.groupby("g")[["w", "ea", "eb"]].sum().values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n, len(s)))
    tot = s[idx].sum(axis=1)
    diff = (tot[:, 1] - tot[:, 2]) / tot[:, 0]
    point = (s[:, 1].sum() - s[:, 2].sum()) / s[:, 0].sum()
    return float(point), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


def log_result(name: str, metrics: dict, oof: np.ndarray | None = None, test_pred: np.ndarray | None = None,
               notes: str = "") -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "oof").mkdir(exist_ok=True)
    if oof is not None:
        np.save(RESULTS / "oof" / f"{name}.npy", oof)
    if test_pred is not None:
        np.save(RESULTS / "oof" / f"{name}__test.npy", test_pred)
    row = {"name": name, **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}, "notes": notes}
    path = RESULTS / "summary.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def summary_table() -> pd.DataFrame:
    rows = [json.loads(l) for l in open(RESULTS / "summary.jsonl")]
    df = pd.DataFrame(rows).drop_duplicates("name", keep="last")
    cols = ["name", "wmae_natural", "wmae_min", "wmae_mid", "wmae_heavy", "mae_sbp", "mae_dbp"]
    return df[cols + [c for c in df.columns if c.startswith("bias_")] + ["notes"]]
