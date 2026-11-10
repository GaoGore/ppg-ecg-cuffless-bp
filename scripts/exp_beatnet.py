"""D4: beat-template network CV (subject-level), scored before/after the shift correction."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppgbp import cv, postprocess, tables  # noqa: E402
from ppgbp.models.beatnet import BeatNetModel  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="beatnet")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--age_w", type=float, default=0.3)
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()
    ctx = cv.CVContext("core", repeats=args.repeats)
    t0 = time.time()
    out = cv.run(ctx, args.name, lambda seed: BeatNetModel(seed=seed, epochs=args.epochs, width=args.width,
                                                           lr=args.lr, age_w=args.age_w),
                 cols=tables.DEMO, notes=f"epochs={args.epochs} width={args.width} lr={args.lr} age_w={args.age_w}")
    print(postprocess.fmt_cf(args.name, postprocess.crossfit_eval(ctx, out["oof_subj"])))
    print(f"total {time.time() - t0:.0f}s")
