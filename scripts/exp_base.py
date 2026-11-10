"""Base-model comparison, each scored before and after the shift correction (cross-fitted)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppgbp import cv, postprocess, tables  # noqa: E402
from ppgbp.models.gbdt import CatModel  # noqa: E402
from ppgbp.models.linear import LinModel  # noqa: E402

EXPS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["core_rmse", "core_mae", "lin300", "full_rmse"]


def report(ctx, name, out):
    mu = out["oof_subj"]
    print(postprocess.fmt_cf(name, postprocess.crossfit_eval(ctx, mu)), flush=True)


ctx_core = cv.CVContext("core")
for e in EXPS:
    if e == "core_rmse":
        out = cv.run(ctx_core, "cat_core_rmse", lambda seed: CatModel(seed=seed), verbose=False)
        report(ctx_core, e, out)
    elif e == "core_mae":
        out = cv.run(ctx_core, "cat_core_mae", lambda seed: CatModel(loss="MAE", seed=seed), verbose=False)
        report(ctx_core, e, out)
    elif e == "core_d4":
        out = cv.run(ctx_core, "cat_core_d4", lambda seed: CatModel(seed=seed, depth=4, iterations=2500, learning_rate=0.02), verbose=False)
        report(ctx_core, e, out)
    elif e == "core_d6":
        out = cv.run(ctx_core, "cat_core_d6", lambda seed: CatModel(seed=seed, depth=6, iterations=1200), verbose=False)
        report(ctx_core, e, out)
    elif e == "lin300":
        out = cv.run(ctx_core, "lin_core_300", lambda seed: LinModel(alpha=300), verbose=False)
        report(ctx_core, e, out)
    elif e == "lin1000":
        out = cv.run(ctx_core, "lin_core_1000", lambda seed: LinModel(alpha=1000), verbose=False)
        report(ctx_core, e, out)
    elif e == "huber":
        out = cv.run(ctx_core, "huber_core", lambda seed: LinModel(alpha=30, huber=True), verbose=False)
        report(ctx_core, e, out)
    elif e == "full_rmse":
        ctx_full = cv.CVContext("full")
        out = cv.run(ctx_full, "cat_full_rmse", lambda seed: CatModel(seed=seed, rsm=0.1), verbose=False)
        report(ctx_full, e, out)
    elif e == "demo":
        out = cv.run(ctx_core, "cat_demo", lambda seed: CatModel(seed=seed, rsm=1.0, iterations=800, depth=4),
                     cols=tables.DEMO, verbose=False)
        report(ctx_core, e, out)
