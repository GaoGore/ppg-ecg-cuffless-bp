"""D4 v2 experiments: round-level head, auxiliary targets, contrastive pretraining.

Scores the network alone and blended with CatBoost (r5_cat), before and after
the shift correction, on the cohort-matched scenarios, against the previous
network (beatnet_e60_r3, same 3 repeats).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppgbp import cv, postprocess, tables, validate  # noqa: E402
from ppgbp.models.beatnet2 import BeatNet2Model  # noqa: E402
from ppgbp.postprocess import ShiftConfig  # noqa: E402

R = validate.RESULTS / "oof"
KEYS = ["natural", "min_bp", "min_dc", "mid_dc"]


def line(name, r):
    return (f"{name:40s} raw " + " ".join(f"{r['raw_' + k]:6.3f}" for k in KEYS)
            + "  | corrected " + " ".join(f"{r['cf_' + k]:6.3f}" for k in KEYS))


def report(ctx, label, nn_oof, seg_delta=None):
    cat = np.load(R / "r5_cat__subj.npy").mean(axis=0)
    cfg = ShiftConfig()
    print(line(f"{label} alone", postprocess.crossfit_eval(ctx, nn_oof, cfg, reps=30, seg_delta=seg_delta)))
    for w in (0.7, 0.5):
        print(line(f"{label} + cat (w_cat={w})", postprocess.crossfit_eval(ctx, w * cat + (1 - w) * nn_oof, cfg, reps=30,
                                                                         seg_delta=seg_delta)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--ssl_epochs", type=int, default=0)
    ap.add_argument("--seg_w", type=float, default=0.5)
    ap.add_argument("--aux_w", type=float, default=0.3)
    ap.add_argument("--enc_lr_mult", type=float, default=1.0)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--baseline", action="store_true", help="also print the previous network for reference")
    args = ap.parse_args()
    ctx = cv.CVContext("core", repeats=args.repeats)
    if args.baseline:
        report(ctx, "old net (beatnet_e60_r3)", np.load(R / "beatnet_e60_r3__subj.npy").mean(axis=0))
    t0 = time.time()
    kw = dict(epochs=args.epochs, ssl_epochs=args.ssl_epochs, seg_w=args.seg_w, aux_w=args.aux_w,
              enc_lr_mult=args.enc_lr_mult)
    out = cv.run(ctx, args.name, lambda seed: BeatNet2Model(seed=seed + args.seed_offset, **kw), cols=tables.DEMO,
                 notes=str(kw))
    print(f"[{args.name}] trained in {time.time() - t0:.0f}s")
    report(ctx, args.name, out["oof_subj"])
    sd = np.load(R / f"{args.name}__segdelta.npy")
    report(ctx, args.name + " +round head", out["oof_subj"], seg_delta=sd)
