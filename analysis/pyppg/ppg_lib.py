# pyPPG wrapper for one PPG channel of one 1-min segment (500 Hz).
# pyPPG 1.0.73 pins 2022-era dependencies that do not install on Python 3.13. Its core modules work unchanged in a
# separate venv with numpy 2.2.6 / pandas 2.2.3 / scipy 1.15.3 / dotmap 1.3.30, then `pip install --no-deps pyPPG==1.0.73`.
# pandas 3 must NOT be used: copy-on-write silently breaks pyPPG's chained assignments (all fiducials come out empty).
# np.NaN (removed in numpy 2) is re-added below.
import warnings, time
import numpy as np
np.NaN = np.nan
import pandas as pd
from dotmap import DotMap
import pyPPG.preproc as PP, pyPPG.fiducials as FP, pyPPG.biomarkers as BM
from pyPPG import PPG, Fiducials
warnings.filterwarnings("ignore")
STATS = ("median", "iqr")                      # per-biomarker statistics over the beats of the segment kept as features

def run_channel(raw, fs=500):
    """raw intensity (L,) -> (dict of biomarker stats, number of beats). Pulse = inverted intensity, as ppgbp.beats."""
    x = raw.astype(float)
    if not np.isfinite(x).all() or np.ptp(x) <= 0: return {}, 0
    s = DotMap(); s.v = -(x - np.median(x)); s.fs = fs; s.start_sig = 0; s.end_sig = len(x); s.name = "seg"
    s.filtering = True; s.fL = 0.5000001; s.fH = 12; s.order = 4; s.sm_wins = {"ppg": 50, "vpg": 10, "apg": 10, "jpg": 10}
    prep = PP.Preprocess(fL=s.fL, fH=s.fH, order=s.order, sm_wins=s.sm_wins)
    s.ppg, s.vpg, s.apg, s.jpg = prep.get_signals(s=s)
    corr = pd.DataFrame(); corr.loc[0, ["on", "dn", "dp", "v", "w", "f"]] = True; s.correction = corr
    p = PPG(s=s, check_ppg_len=True)
    fid = FP.FpCollection(s=p).get_fiducials(s=p)
    if len(fid) < 5: return {}, len(fid)
    fp = Fiducials(fp=fid)
    _, vals, stats = BM.BmCollection(s=p, fp=fp).get_biomarkers()
    out = {}
    for grp, df in stats.items():
        for st in STATS:
            for k, v in df.loc[st].items():
                if k != "TimeStamp": out[f"{k}__{st}"] = float(v)
    return out, len(fid)
