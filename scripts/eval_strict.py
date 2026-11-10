"""Strict comparison of candidate blends against a reference recipe (CatBoost .7 + old net .3).

Each candidate: CatBoost (r5_cat) blended with an average of network runs, shift-corrected
(corrector fitted on all OOF predictions, same for every candidate). For each scenario the
paired subject bootstrap (3,000 resamples) gives the 95% interval of
wMAE(candidate) - wMAE(reference). A candidate counts as better only if the upper bound is
below 0 on both cohort-matched scenarios (min_dc and mid_dc).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppgbp import cv, validate  # noqa: E402
from ppgbp.postprocess import ShiftConfig, ShiftCorrector  # noqa: E402

R = validate.RESULTS / "oof"


def runs(names):
    have = [n for n in names if (R / f"{n}__subj.npy").exists()]
    return np.mean([np.load(R / f"{n}__subj.npy").mean(axis=0) for n in have], axis=0) if have else None, have


def main():
    ctx = cv.CVContext("demo")
    age = ctx.S_tr.age.values
    cat = np.load(R / "r5_cat__subj.npy").mean(axis=0)
    fam = {
        "old net": ["r5_nn"],
        "v2 4-block, 3 local seeds": [f"b2_r5_s{s}" for s in (0, 1000, 2000)],
        "v2 4-block, 4 remote seeds": [f"b4_r5_s{s}" for s in (3000, 4000, 5000, 6000)],
        "v2 4-block, 7 seeds": [f"b2_r5_s{s}" for s in (0, 1000, 2000)] + [f"b4_r5_s{s}" for s in (3000, 4000, 5000, 6000)],
        "v2 5-block, 7 seeds": [f"b5_r5_s{s}" for s in (0, 1000, 2000, 3000, 4000, 5000, 6000)],
        "raw-waveform net": [f"raw_r5_s{s}" for s in (0, 1000, 2000, 3000)],
    }
    nets = {k: runs(v) for k, v in fam.items()}
    for k, (_, have) in nets.items():
        print(f"{k:28s} runs found: {len(have)}")
    blends = {"reference: cat .7 + old net .3": 0.7 * cat + 0.3 * nets["old net"][0],
              "cat .7 + v2 4-block (3 seeds) .3": 0.7 * cat + 0.3 * nets["v2 4-block, 3 local seeds"][0]}
    if nets["v2 4-block, 4 remote seeds"][0] is not None:
        t7, f7 = nets["v2 4-block, 7 seeds"][0], nets["v2 5-block, 7 seeds"][0]
        blends["cat .7 + v2 4-block (7 seeds) .3"] = 0.7 * cat + 0.3 * t7
        blends["cat .7 + v2 5-block (7 seeds) .3"] = 0.7 * cat + 0.3 * f7
        blends["cat .7 + v2 both depths (14 seeds) .3"] = 0.7 * cat + 0.3 * (t7 + f7) / 2
        blends["cat .7 + (old + 4-block + 5-block)/3 .3"] = 0.7 * cat + 0.3 * (nets["old net"][0] + t7 + f7) / 3
        if nets["raw-waveform net"][0] is not None:
            raw = nets["raw-waveform net"][0]
            blends["cat .7 + raw net .3"] = 0.7 * cat + 0.3 * raw
            blends["cat .6 + v2 both depths .2 + raw .2"] = 0.6 * cat + 0.2 * (t7 + f7) / 2 + 0.2 * raw
            blends["cat .5 + v2 both depths .25 + raw .25"] = 0.5 * cat + 0.25 * (t7 + f7) / 2 + 0.25 * raw
            blends["cat .7 + (old + 4b + 5b + raw)/4 .3"] = 0.7 * cat + 0.3 * (nets["old net"][0] + t7 + f7 + raw) / 4
    corr = {}
    for k, mu in blends.items():
        c = ShiftCorrector(ShiftConfig()).fit(mu, ctx.Y, ctx.m_tr, ctx.m_te, age=age)
        corr[k] = ctx.to_segments(c.transform(mu, age=age))
    ref = corr["reference: cat .7 + old net .3"]
    y = ctx.m_tr[["sbp", "dbp"]].values
    print(f"\n{'candidate':46s} " + " | ".join(f"{s:^27s}" for s in ("natural", "min_dc", "mid_dc")) + " | verdict")
    for k, p in corr.items():
        cells, better = [], True
        for sc in ("natural", "min_dc", "mid_dc"):
            level = np.average(validate.wmae_terms(p, y), weights=ctx.W[sc])
            if k.startswith("reference"):
                cells.append(f"{level:6.3f}{'':21s}")
                continue
            pt, lo, hi = validate.paired_bootstrap(p, ref, ctx.m_tr, ctx.W[sc], n=3000)
            cells.append(f"{level:6.3f} {pt:+.3f} [{lo:+.3f},{hi:+.3f}]")
            if sc != "natural" and hi >= 0:
                better = False
        verdict = "reference" if k.startswith("reference") else ("BETTER" if better else "no clear gain")
        print(f"{k:46s} " + " | ".join(cells) + f" | {verdict}")


if __name__ == "__main__":
    main()
