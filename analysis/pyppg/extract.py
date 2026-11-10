# pyPPG biomarkers for every segment (train + test), all 16 PPG channels -> cache/pyppg/<split>_<sid>.csv (one row per segment).
# Columns: seg_id, <channel>__nbeats, <channel>__<biomarker>__<median|iqr>. Resumable: subjects with an output file are skipped.
# usage (pandas-2.2 venv, see ppg_lib.py): python extract.py [--workers N] [--limit K]
_ROOT = __import__('os').environ.get('PPGBP_ROOT') or str(__import__('pathlib').Path(__file__).resolve().parents[2])  # repository root
import sys, os, io as _io, time, contextlib, multiprocessing as mp
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from ppgbp import io
from ppg_lib import run_channel
OUT = io.PROJECT_DIR / "cache" / "pyppg"

def do_subject(path):
    out = OUT / f"{path.parent.name}_{path.stem}.csv"
    if out.exists(): return path.stem, 0.0, 0
    t = time.time(); subj = io.load_subject(path); rows = []; fails = 0
    for r, rd in enumerate(subj["rounds"], start=1):
        row = {"seg_id": io.seg_id(subj["sid"], r)}
        for c, ch in enumerate(io.PPG_CHANNELS):
            try:
                with contextlib.redirect_stdout(_io.StringIO()): f, nb = run_channel(rd["ppg"][c])
            except Exception:
                f, nb = {}, -1; fails += 1
            row[f"{ch}__nbeats"] = nb
            row.update({f"{ch}__{k}": v for k, v in f.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False)
    return path.stem, time.time() - t, fails

if __name__ == "__main__":
    argv = sys.argv; nw = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 12
    paths = io.subject_paths("train") + io.subject_paths("test")
    if "--limit" in argv: paths = paths[::max(1, len(paths) // int(argv[argv.index("--limit") + 1]))][: int(argv[argv.index("--limit") + 1])]
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time(); done = 0; fails = 0
    with mp.get_context("fork").Pool(nw) as pool:
        for sid, dt, fl in pool.imap_unordered(do_subject, paths, chunksize=1):
            done += 1; fails += fl
            if done % 25 == 0 or done == len(paths):
                print(f"{done}/{len(paths)} subjects, {time.time() - t0:.0f}s, channel failures {fails}", flush=True)
    print("finished", flush=True)
