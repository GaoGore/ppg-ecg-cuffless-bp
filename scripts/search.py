"""Hyper-parameter / architecture search, judged on the cohort-matched validation.

Each candidate is trained with 3 CV repeats and scored inside the blend with
the other model family held fixed (CatBoost 0.7 / network 0.3), after the
shift correction, cross-fitted: objective = mean of the min_dc and mid_dc
scenario wMAEs. Results are appended to results/search_<family>.csv.

  .venv/bin/python scripts/search.py cat
  .venv/bin/python scripts/search.py nn
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppgbp import cv, postprocess, tables, validate  # noqa: E402
from ppgbp.models.gbdt import CatModel  # noqa: E402
from ppgbp.postprocess import ShiftConfig  # noqa: E402

R = validate.RESULTS / "oof"
L = lambda n: np.load(R / f"{n}__subj.npy").mean(axis=0)            # noqa: E731
KEYS = ["natural", "min_bp", "min_dc", "mid_dc"]


def score(ctx, mu, reps=30):
    r = postprocess.crossfit_eval(ctx, mu, ShiftConfig(), reps=reps)
    out = {f"cf_{k}": r[f"cf_{k}"] for k in KEYS}
    out["raw_natural"] = r["raw_natural"]
    out["objective"] = 0.5 * (r["cf_min_dc"] + r["cf_mid_dc"])
    return out


def cat_configs(n=16, seed=0):
    rng = np.random.default_rng(seed)
    grid = []
    for _ in range(200):
        depth = int(rng.choice([4, 5, 6, 7]))
        lr, it = [(0.015, 3000), (0.03, 1500), (0.05, 900)][rng.integers(3)]
        if depth == 7 and lr == 0.015:
            continue
        cfg = dict(depth=depth, learning_rate=lr, iterations=it, l2_leaf_reg=float(rng.choice([1, 3, 5, 10, 20])),
                   rsm=float(rng.choice([0.1, 0.2, 0.35, 0.5])), subsample=float(rng.choice([0.6, 0.8, 1.0])),
                   loss=str(rng.choice(["RMSE", "MAE", "Huber:delta=8"])),
                   grow_policy=str(rng.choice(["SymmetricTree", "Depthwise"])))
        if cfg not in grid:
            grid.append(cfg)
        if len(grid) == n:
            break
    return grid


NN_CONFIGS = [
    {"width": 64},
    {"kernel": 11},
    {"n_blocks": 5},
    {"pool": "mean"},
    {"drop": 0.3, "wd": 1e-2},
    {"epochs": 90, "lr": 1e-3},
    {"aux_w": 0.6, "seg_w": 1.0},
]


def main():
    family = sys.argv[1]
    ctx = cv.CVContext("core", repeats=3)
    out_csv = validate.RESULTS / f"search_{family}.csv"
    done = set(pd.read_csv(out_csv).cfg) if out_csv.exists() else set()
    nn_partner = np.mean([L(f"b2_r5_s{s}") for s in (0, 1000, 2000)], axis=0)
    cat_partner = L("r5_cat")
    if family == "cat":
        cands = [("baseline", {})] + [(f"c{i:02d}", c) for i, c in enumerate(cat_configs())]
    else:
        from ppgbp.models.beatnet2 import BeatNet2Model
        cands = [("baseline", None)] + [(f"n{i:02d}", c) for i, c in enumerate(NN_CONFIGS)]
    for tag, cfg in cands:
        key = json.dumps(cfg, sort_keys=True)
        if key in done:
            continue
        t0 = time.time()
        if family == "cat":
            params = dict(cfg)
            loss = params.pop("loss", "RMSE")
            if params.get("grow_policy") == "Depthwise":
                params["min_data_in_leaf"] = 10
            params["thread_count"] = 8
            out = cv.run(ctx, f"search_cat_{tag}", lambda seed, p=params, l=loss: CatModel(loss=l, seed=seed, **p),
                         verbose=False, notes=key)
            mu_alone, mu_blend = out["oof_subj"], 0.7 * out["oof_subj"] + 0.3 * nn_partner
        else:
            if cfg is None:                       # the current network, already trained with 3 repeats
                oof = L("b2_auxseg")
            else:
                out = cv.run(ctx, f"search_nn_{tag}", lambda seed, c=cfg: BeatNet2Model(seed=seed, **c),
                             cols=tables.DEMO, verbose=False, notes=key)
                oof = out["oof_subj"]
            mu_alone, mu_blend = oof, 0.7 * cat_partner + 0.3 * oof
        a, b = score(ctx, mu_alone), score(ctx, mu_blend)
        row = {"tag": tag, "cfg": key, "minutes": round((time.time() - t0) / 60, 1),
               **{f"alone_{k}": round(v, 4) for k, v in a.items()}, **{f"blend_{k}": round(v, 4) for k, v in b.items()}}
        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
        print(f"[{family}] {tag} {key}: blend objective {b['objective']:.3f} (min_dc {b['cf_min_dc']:.3f}, "
              f"mid_dc {b['cf_mid_dc']:.3f}, natural {b['cf_natural']:.3f}) | alone {a['objective']:.3f} | "
              f"{row['minutes']} min", flush=True)


if __name__ == "__main__":
    main()
