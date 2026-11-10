"""Model tables: subject-level pooled features and targets.

Within-subject BP varies little (SD ~2 mmHg) compared with between-subject BP
(SD ~15 mmHg), so models are fitted per subject on features pooled (median)
over all of the subject's segments; segment predictions add a per-round offset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import cache

DEMO = ["age", "male", "height", "weight", "bmi", "bsa", "med_his", "med_today"]


def _feature_groups(cols: list[str]) -> dict[str, list[str]]:
    per_channel = [c for c in cols if c.startswith("PPG_")]
    consensus = [c for c in cols if c.split("__")[0] in ("IR", "R", "G", "B", "all")]
    cross = [c for c in cols if c.startswith(("x_", "xpos", "xc_"))]
    ecg = [c for c in cols if c.startswith("ecg__") or c.startswith("ecgq_") or c == "ecg_ok"]
    rhythm = [c for c in cols if c.startswith("rhy__") or c in ("hr_ppg", "n_beats")]
    motion = [c for c in cols if c.startswith("acc__")]
    return {"per_channel": per_channel, "consensus": consensus, "cross": cross, "ecg": ecg,
            "rhythm": rhythm, "motion": motion}


FEATURE_SETS = {
    "demo": [],
    "core": ["consensus", "cross", "ecg", "rhythm", "motion"],
    "full": ["per_channel", "consensus", "cross", "ecg", "rhythm", "motion"],
}


def load_tables(feature_set: str = "core", extra: tuple = ()):
    """Returns dict with subject tables S_tr/S_te (features + targets), segment
    meta for train/test with a 'row' index into the subject tables, and the
    list of feature columns."""
    meta, feats = cache.load()
    meta = meta.copy()
    meta["bmi"] = meta.weight / (meta.height / 100) ** 2
    meta["bsa"] = np.sqrt(meta.height * meta.weight / 3600)          # Mosteller
    groups = _feature_groups([c for c in feats.columns if c != "seg_id"])
    sig_cols = [c for g in FEATURE_SETS[feature_set] for c in groups[g]]
    D = pd.concat([meta, feats[sig_cols]], axis=1)
    if "dyn" in extra:                                   # beat-to-beat dynamics (ppgbp/dynamics.py)
        dyn = pd.read_parquet(cache.CACHE_DIR / "dynamics.parquet")
        dcols = [c for c in dyn.columns if c != "seg_id"]
        D = D.merge(dyn, on="seg_id", how="left", validate="one_to_one")
        sig_cols = sig_cols + dcols
    agg = {c: "first" for c in DEMO + ["split", "n_rounds"]}
    agg.update({c: "median" for c in sig_cols})
    agg.update({"sbp": "mean", "dbp": "mean"})
    S = D.groupby("sid", sort=True).agg(agg)
    S["ecg_ok_frac"] = feats["ecg_ok"].groupby(meta["sid"].values).mean()
    cols = DEMO + sig_cols + (["ecg_ok_frac"] if sig_cols else [])
    if extra and (cache.CACHE_DIR / "agegap.parquet").exists() and "agegap" in extra:
        ag = pd.read_parquet(cache.CACHE_DIR / "agegap.parquet").set_index("sid")
        S = S.join(ag)
        cols += ["age_sig", "age_gap"]
    S_tr, S_te = S[S.split == "train"].copy(), S[S.split == "test"].copy()
    m_tr = meta[meta.split == "train"].reset_index(drop=True)
    m_te = meta[meta.split == "test"].reset_index(drop=True)
    m_tr["row"] = S_tr.index.get_indexer(m_tr.sid)
    m_te["row"] = S_te.index.get_indexer(m_te.sid)
    return {"S_tr": S_tr, "S_te": S_te, "m_tr": m_tr, "m_te": m_te, "cols": cols}


def round_offsets(m_tr: pd.DataFrame, max_round: int = 5, shrink: float = 20.0) -> np.ndarray:
    """(max_round+1, 2) mean deviation of a round's label from its subject mean,
    shrunk towards 0 for rarely seen rounds."""
    dev = m_tr[["sbp", "dbp"]].values - m_tr.groupby("sid")[["sbp", "dbp"]].transform("mean").values
    off = np.zeros((max_round + 1, 2))
    for r in range(1, max_round + 1):
        m = (m_tr["round"] == r).values
        if m.any():
            off[r] = dev[m].sum(axis=0) / (m.sum() + shrink)
    return off
