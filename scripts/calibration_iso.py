"""Two reference numbers for judging calibration-free PPG/ECG BP estimates.

1. Per-user calibration baseline: take each subject's round-1 label as the "calibration" value and predict
   the later rounds with it, using no signal at all. Rounds >= 3 share no cuff reading with round 1 (each
   round's label is the mean of the readings before and after it), so they give an honest number.
2. ISO 81060-2 criterion 1 (mean error within +/-5 mmHg, SD at most 8 mmHg) for out-of-fold predictions,
   under the training distribution and under an AAMI-style target distribution (scenario weights).

  .venv/bin/python scripts/calibration_iso.py [oof.npz ...]   (default: analysis/pyppg/out/core.npz)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ppgbp import cv  # noqa: E402


def calibration_baseline(m_tr: pd.DataFrame) -> None:
    r1 = m_tr[m_tr["round"] == 1].set_index("sid")[["sbp", "dbp"]]
    for lab, sel in (("round 2 (shares one cuff reading with round 1, optimistic)", m_tr["round"] == 2),
                     ("rounds >= 3 (no shared cuff reading)", m_tr["round"] >= 3)):
        t = m_tr[sel].dropna(subset=["sbp"])
        e = t[["sbp", "dbp"]].values - r1.loc[t.sid].values
        mae = np.abs(e).mean(0)
        print(f"calibrate on round 1, predict {lab}: n={len(t)}, MAE SBP {mae[0]:.2f} DBP {mae[1]:.2f}, "
              f"wMAE {0.6 * mae[0] + 0.4 * mae[1]:.2f}, error SD SBP {e[:, 0].std():.2f} DBP {e[:, 1].std():.2f}")
    g = m_tr.groupby("sid")[["sbp", "dbp"]]
    print(f"between-subject SD of subject means: SBP {g.mean().sbp.std():.1f}, DBP {g.mean().dbp.std():.1f}; "
          f"mean within-subject SD: SBP {g.std().sbp.mean():.1f}, DBP {g.std().dbp.mean():.1f}")


def iso_criterion1(ctx, oof: np.ndarray, name: str) -> None:
    y = ctx.m_tr[["sbp", "dbp"]].values
    for wn in ("natural", "mid_dc"):
        w = ctx.W[wn]; e = oof - y
        m = np.average(e, axis=0, weights=w); sd = np.sqrt(np.average((e - m) ** 2, axis=0, weights=w))
        ok = (np.abs(m) <= 5) & (sd <= 8)
        print(f"{name} [{wn}] mean error SBP {m[0]:+.1f} DBP {m[1]:+.1f} | SD SBP {sd[0]:.1f} DBP {sd[1]:.1f} | "
              f"criterion 1 met: SBP {ok[0]}, DBP {ok[1]}")


def main():
    ctx = cv.CVContext("core")
    calibration_baseline(ctx.m_tr)
    files = sys.argv[1:] or [str(ROOT / "analysis/pyppg/out/core.npz")]
    for f in files:
        if Path(f).exists():
            iso_criterion1(ctx, np.load(f)["oof"], Path(f).stem)
        else:
            print(f"{f} not found (run analysis/pyppg/evaluate.py --fit core first)")


if __name__ == "__main__":
    main()
