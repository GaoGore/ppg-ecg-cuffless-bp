"""End-to-end: (cache must exist) -> CatBoost + BeatNet subject models over repeated
CV -> blend -> shift-aware correction -> predictions CSV for the hold-out rows.

  .venv/bin/python scripts/build_cache.py
  .venv/bin/python scripts/final_pipeline.py --out predictions/final.csv

Test predictions average every fold model of every repeat; the shift corrector
is fitted on the blended out-of-fold predictions of the training subjects.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppgbp import cv, postprocess, predictions, tables, validate  # noqa: E402
from ppgbp.models.beatnet import BeatNetModel  # noqa: E402
from ppgbp.models.gbdt import CatModel  # noqa: E402
from ppgbp.postprocess import ShiftConfig, ShiftCorrector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--w_cat", type=float, default=0.7)
    ap.add_argument("--nn_epochs", type=int, default=60)
    ap.add_argument("--scenario", default="mid")
    ap.add_argument("--factors", default="bponly")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--scale_model", default="loglin")
    ap.add_argument("--smooth", type=float, default=4.0)
    ap.add_argument("--reuse", action="store_true", help="reuse saved OOF/test predictions if present")
    ap.add_argument("--tag", default="final")
    args = ap.parse_args()
    t0 = time.time()
    ctx = cv.CVContext("core", repeats=args.repeats)
    R = validate.RESULTS / "oof"
    runs = {}
    for name, fn, cols in [
        (f"{args.tag}_cat", lambda seed: CatModel(seed=seed), None),
        (f"{args.tag}_nn", lambda seed: BeatNetModel(seed=seed, epochs=args.nn_epochs), tables.DEMO),
    ]:
        if args.reuse and (R / f"{name}__subj.npy").exists():
            runs[name] = (np.load(R / f"{name}__subj.npy").mean(axis=0), np.load(R / f"{name}__subj_test.npy"))
        else:
            out = cv.run(ctx, name, fn, cols=cols)
            runs[name] = (out["oof_subj"], np.load(R / f"{name}__subj_test.npy"))
    (cat_oof, cat_te), (nn_oof, nn_te) = runs[f"{args.tag}_cat"], runs[f"{args.tag}_nn"]
    w = args.w_cat
    mu_oof, mu_te = w * cat_oof + (1 - w) * nn_oof, w * cat_te + (1 - w) * nn_te
    cfg = ShiftConfig(scenario=args.scenario, factors=args.factors, tau=args.tau, smooth_sd=args.smooth,
                      scale_model=args.scale_model)
    cf = postprocess.crossfit_eval(ctx, mu_oof, cfg)
    print(postprocess.fmt_cf("blend (cross-fitted)", cf))
    corr = ShiftCorrector(cfg).fit(mu_oof, ctx.Y, ctx.m_tr, ctx.m_te, age=ctx.S_tr.age.values)
    pred = ctx.to_segments(corr.transform(mu_te, age=ctx.S_te.age.values), "test")
    df = predictions.write(pred, ctx.m_te, Path(args.out))
    print(f"wrote {args.out}: SBP {df.Predicted_SBP.min():.1f}-{df.Predicted_SBP.max():.1f} (mean {df.Predicted_SBP.mean():.1f}), "
          f"DBP {df.Predicted_DBP.min():.1f}-{df.Predicted_DBP.max():.1f} (mean {df.Predicted_DBP.mean():.1f}); {time.time() - t0:.0f}s")
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {"args": vars(args), "crossfit": cf, "shift": vars(cfg)}, indent=1))


if __name__ == "__main__":
    main()
