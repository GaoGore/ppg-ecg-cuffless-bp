"""Reference numbers for judging calibration-free PPG/ECG BP estimates.

1. Per-user calibration baseline: take each subject's round-1 label as the "calibration" value and predict
   the later rounds with it, using no signal at all. Rounds >= 3 share no cuff reading with round 1 (each
   round's label is the mean of the readings before and after it), so they give an honest number. The model's
   own wMAE on the same rows is printed next to it.
2. ISO 81060-2 criterion 1 (mean error within +/-5 mmHg, SD at most 8 mmHg) for out-of-fold predictions, under
   the training distribution ("natural") and under the AAMI/ISO-style target distribution ("mid": BP stratum
   quotas raked jointly with the hold-out set's age, sex and medication margins). This is only an indicative
   comparison: the standard prescribes its own validation protocol.
3. Mean SBP error in each SBP stratum (<=100, 100-140, 140-160, >=160 mmHg), i.e. where the errors come from.

  .venv/bin/python scripts/calibration_iso.py [oof ...]
  oof = a subject-level OOF array <name>__subj.npy (repeats x subjects x 2) or a row-level .npz with an "oof"
  array (analysis/pyppg/out/). Default: results/oof/r5_cat__subj.npy, the uncorrected CatBoost of
  scripts/final_pipeline.py --repeats 5 --tag r5.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ppgbp import cv, validate  # noqa: E402

WEIGHTS = ("natural", "mid")
SBP_BANDS = ("<=100", "100-140", "140-160", ">=160")      # validate.sbp_stratum 0..3


def wmae(e: np.ndarray) -> float:
    return 0.6 * np.abs(e[:, 0]).mean() + 0.4 * np.abs(e[:, 1]).mean()


def calibration_baseline(m_tr: pd.DataFrame) -> None:
    r1 = m_tr[m_tr["round"] == 1].set_index("sid")[["sbp", "dbp"]]
    for lab, sel in (("round 2 (shares one cuff reading with round 1, optimistic)", m_tr["round"] == 2),
                     ("rounds >= 3 (no shared cuff reading)", m_tr["round"] >= 3)):
        t = m_tr[sel].dropna(subset=["sbp"])
        e = t[["sbp", "dbp"]].values - r1.loc[t.sid].values
        mae = np.abs(e).mean(0)
        print(f"calibrate on round 1, predict {lab}: n={len(t)}, MAE SBP {mae[0]:.2f} DBP {mae[1]:.2f}, "
              f"wMAE {wmae(e):.2f}, error SD SBP {e[:, 0].std():.2f} DBP {e[:, 1].std():.2f}")
    g = m_tr.groupby("sid")[["sbp", "dbp"]]
    print(f"between-subject SD of subject means: SBP {g.mean().sbp.std():.1f}, DBP {g.mean().dbp.std():.1f}; "
          f"mean within-subject SD: SBP {g.std().sbp.mean():.1f}, DBP {g.std().dbp.mean():.1f}")


def load_oof(ctx, f: Path) -> np.ndarray:
    """row-level (segments x [SBP, DBP]) OOF predictions from either file type"""
    if f.suffix == ".npz":
        return np.load(f)["oof"]
    if f.name.endswith("__subj.npy"):
        return ctx.to_segments(np.load(f).mean(0))      # mean over the CV repeats, then back to rows
    raise SystemExit(f"{f}: expected <name>__subj.npy (repeats x subjects x 2) or a .npz with an 'oof' array")


def model_report(ctx, oof: np.ndarray, name: str) -> None:
    m = ctx.m_tr; y = m[["sbp", "dbp"]].values
    ok = ~np.isnan(y).any(1); e = oof - y
    late = ok & (m["round"].values >= 3)
    print(f"{name}: wMAE on rounds >= 3 (same rows as the calibration baseline) {wmae(e[late]):.2f}, "
          f"all rows {wmae(e[ok]):.2f}")
    for wn in WEIGHTS:
        w = ctx.W[wn][ok]; eo = e[ok]
        mu = np.average(eo, axis=0, weights=w); sd = np.sqrt(np.average((eo - mu) ** 2, axis=0, weights=w))
        met = (np.abs(mu) <= 5) & (sd <= 8)
        print(f"{name} [{wn}] mean error SBP {mu[0]:+.1f} DBP {mu[1]:+.1f} | SD SBP {sd[0]:.1f} DBP {sd[1]:.1f} | "
              f"criterion 1 met: SBP {met[0]}, DBP {met[1]}")
    sb = validate.sbp_stratum(m.sbp.values)
    print(f"{name} mean SBP error by SBP stratum (unweighted): " + ", ".join(
        f"{b} {e[ok & (sb == j), 0].mean():+.1f} (n={int((ok & (sb == j)).sum())})" for j, b in enumerate(SBP_BANDS)))


def main():
    ctx = cv.CVContext("core")
    calibration_baseline(ctx.m_tr)
    files = sys.argv[1:] or [str(validate.RESULTS / "oof" / "r5_cat__subj.npy")]
    for f in map(Path, files):
        if f.exists():
            model_report(ctx, load_oof(ctx, f), f.name.removesuffix("__subj.npy").removesuffix(".npz"))
        else:
            print(f"{f} not found (run scripts/final_pipeline.py --repeats 5 --tag r5, or analysis/pyppg/evaluate.py --fit)")


if __name__ == "__main__":
    main()
