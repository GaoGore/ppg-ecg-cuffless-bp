"""Train one BeatNet v2 configuration with CV and score it like scripts/search.py.

  .venv/bin/python scripts/run_nn.py --name b6_r3 --repeats 3 --kw '{"n_blocks": 6}'
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppgbp import cv, postprocess, tables, validate  # noqa: E402
from ppgbp.models.beatnet2 import BeatNet2Model  # noqa: E402
from ppgbp.postprocess import ShiftConfig  # noqa: E402

R = validate.RESULTS / "oof"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--kw", default="{}")
    ap.add_argument("--model", default="beatnet2", choices=["beatnet2", "rawnet"])
    args = ap.parse_args()
    kw = json.loads(args.kw)
    ctx = cv.CVContext("core", repeats=args.repeats)
    t0 = time.time()
    if args.model == "rawnet":
        from ppgbp.models.rawnet import RawNetModel as Model
    else:
        Model = BeatNet2Model
    out = cv.run(ctx, args.name, lambda seed: Model(seed=seed + args.seed_offset, **kw), cols=tables.DEMO,
                 verbose=False, notes=f"{args.model} {args.kw}")
    cat = np.load(R / "r5_cat__subj.npy").mean(axis=0)
    for label, mu in [("alone", out["oof_subj"]), ("blend cat .7", 0.7 * cat + 0.3 * out["oof_subj"])]:
        r = postprocess.crossfit_eval(ctx, mu, ShiftConfig(), reps=30)
        print(f"[{args.name}] {label:13s} objective {0.5 * (r['cf_min_dc'] + r['cf_mid_dc']):.3f} "
              f"(min_dc {r['cf_min_dc']:.3f}, mid_dc {r['cf_mid_dc']:.3f}, natural {r['cf_natural']:.3f})", flush=True)
    print(f"[{args.name}] {time.time() - t0:.0f}s")
