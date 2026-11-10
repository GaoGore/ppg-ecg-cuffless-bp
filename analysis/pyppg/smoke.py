# smoke test: one training subject, round 1, all 16 channels -> timing, beats, plausibility of a few biomarkers
_ROOT = __import__('os').environ.get('PPGBP_ROOT') or str(__import__('pathlib').Path(__file__).resolve().parents[2])  # repository root
import sys, time
sys.path.insert(0, _ROOT); sys.path.insert(0, (_ROOT + "/analysis/pyppg"))
import numpy as np
from ppgbp import io
from ppg_lib import run_channel
p = io.subject_paths("train")[int(sys.argv[1]) if len(sys.argv) > 1 else 0]
subj = io.load_subject(p); rd = subj["rounds"][0]
print(p.stem, subj["demo"], "labels", subj["labels"][0])
for c, name in enumerate(io.PPG_CHANNELS):
    t = time.time(); f, nb = run_channel(rd["ppg"][c]); dt = time.time() - t
    show = {k: round(f[k], 3) for k in ("Tpi__median", "Tsys__median", "Tdia__median", "AGI__median", "IPR__median") if k in f}
    print(f"{name:9s} {dt:5.2f}s beats {nb:3d} n_feat {len(f):4d} {show}")
