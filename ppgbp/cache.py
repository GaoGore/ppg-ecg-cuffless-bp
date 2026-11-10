"""Parallel extraction of segment features and NN templates into cache/."""
from __future__ import annotations

import hashlib
import json
import os
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import beats as bt
from . import features as ft
from . import io

CACHE_DIR = Path(os.environ.get("PPGBP_CACHE", io.PROJECT_DIR / "cache"))
VERSION = "beats-v1"
K_SUBSETS = 16
SUBSET_FRAC = 0.5


def _seed(seg_id: str) -> int:
    return int(hashlib.sha1(seg_id.encode()).hexdigest()[:8], 16)


def process_subject(path: Path) -> list[dict]:
    split = path.parent.name
    subj = io.load_subject(path)
    rows = []
    for r, rd in enumerate(subj["rounds"], start=1):
        sid = io.seg_id(subj["sid"], r)
        meta = {"split": split, "sid": subj["sid"], "round": r, "n_rounds": len(subj["rounds"]), "seg_id": sid,
                "sbp": subj["labels"][r - 1, 0], "dbp": subj["labels"][r - 1, 1], **subj["demo"]}
        seg = bt.process_segment(rd["ppg"], rd["ecg"], rd["acc"])
        feats = ft.segment_features(seg, subj["demo"], rd["acc"])
        tmpl = np.zeros((K_SUBSETS + 1, 17, ft.NN_LEN), np.float16)
        mask = np.zeros(17, bool)
        if seg.get("ok"):
            full, mask = ft.nn_template(seg["templates"], seg)
            tmpl[0] = full
            rng = np.random.default_rng(_seed(sid))
            for j, T in enumerate(bt.subset_templates(seg, K_SUBSETS, SUBSET_FRAC, rng), start=1):
                x, _ = ft.nn_template(T, seg)
                tmpl[j] = x
            tmpl[:, ~mask] = 0
        rows.append({"meta": meta, "feats": feats, "tmpl": tmpl, "mask": mask})
    return rows


def build(workers: int = 12, splits=("train", "test")) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    paths = [p for s in splits for p in io.subject_paths(s)]
    t0 = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(process_subject, paths, chunksize=2)
    rows = [r for rr in res for r in rr]
    meta = pd.DataFrame([r["meta"] for r in rows])
    feats = pd.DataFrame([r["feats"] for r in rows])
    feats.insert(0, "seg_id", meta["seg_id"].values)
    meta.to_parquet(CACHE_DIR / "meta.parquet", index=False)
    feats.to_parquet(CACHE_DIR / "features.parquet", index=False)
    np.save(CACHE_DIR / "nn_templates.npy", np.stack([r["tmpl"] for r in rows]))
    np.save(CACHE_DIR / "nn_masks.npy", np.stack([r["mask"] for r in rows]))
    info = {"version": VERSION, "n_segments": len(rows), "n_features": feats.shape[1] - 1,
            "k_subsets": K_SUBSETS, "subset_frac": SUBSET_FRAC, "seconds": round(time.time() - t0, 1)}
    (CACHE_DIR / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_parquet(CACHE_DIR / "meta.parquet")
    feats = pd.read_parquet(CACHE_DIR / "features.parquet")
    assert (meta.seg_id.values == feats.seg_id.values).all()
    return meta, feats


def load_templates(mmap: bool = True) -> tuple[np.ndarray, np.ndarray]:
    return (np.load(CACHE_DIR / "nn_templates.npy", mmap_mode="r" if mmap else None),
            np.load(CACHE_DIR / "nn_masks.npy"))


def _dyn_subject(path: Path) -> list[dict]:
    from . import dynamics
    subj = io.load_subject(path)
    rows = []
    for r, rd in enumerate(subj["rounds"], start=1):
        seg = bt.process_segment(rd["ppg"], rd["ecg"], rd["acc"])
        f = dynamics.dynamics_features(seg)
        rows.append({"seg_id": io.seg_id(subj["sid"], r), **{f"dyn__{k}": v for k, v in f.items()}})
    return rows


def build_dynamics(workers: int = 12, splits=("train", "test")) -> pd.DataFrame:
    """Beat-to-beat dynamics features for every segment -> cache/dynamics.parquet."""
    paths = [p for s in splits for p in io.subject_paths(s)]
    t0 = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_dyn_subject, paths, chunksize=2)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_parquet(CACHE_DIR / "dynamics.parquet", index=False)
    print(f"dynamics: {df.shape[0]} segments x {df.shape[1] - 1} features in {time.time() - t0:.0f}s")
    return df


RAW_FS = 125


def raw_paths(fs: int = RAW_FS) -> tuple[Path, Path]:
    return CACHE_DIR / f"raw{fs}.npy", CACHE_DIR / f"raw{fs}_mask.npy"


def _raw_subject(path: Path, fs: int = RAW_FS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per segment: (17, 60*fs) float16 robust-normalised pulses + ECG, and a channel mask."""
    from scipy.signal import resample_poly, sosfiltfilt
    subj = io.load_subject(path)
    out = []
    for rd in subj["rounds"]:
        pulses, _ = bt.ppg_pulses(rd["ppg"])
        ecg = rd["ecg"]
        ecg_f = sosfiltfilt(bt._sos(0.5, 40.0), ecg - np.median(ecg)) if np.isfinite(ecg).all() else np.full_like(ecg, np.nan)
        X = np.vstack([pulses, ecg_f[None, :]])
        X = resample_poly(np.nan_to_num(X), 1, io.FS // fs, axis=1)
        mask = np.isfinite(np.vstack([pulses, ecg_f[None, :]])).all(axis=1)
        med = np.median(X, axis=1, keepdims=True)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=1))[:, None]
        X = np.clip((X - med) / np.where(iqr > 0, iqr, 1.0), -10, 10)
        X[~mask] = 0
        out.append((X.astype(np.float16), mask))
    return out


def build_raw(workers: int = 12, splits=("train", "test"), fs: int = RAW_FS) -> None:
    """Band-passed, robust-normalised raw signals of every segment at `fs` Hz (meta order)."""
    from functools import partial
    paths = [p for s in splits for p in io.subject_paths(s)]
    t0 = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(partial(_raw_subject, fs=fs), paths, chunksize=2)
    X = np.stack([x for rr in res for x, _ in rr])
    M = np.stack([m for rr in res for _, m in rr])
    px, pm = raw_paths(fs)
    np.save(px, X)
    np.save(pm, M)
    print(f"raw cache {fs} Hz: {X.shape} float16 ({X.nbytes / 1e9:.2f} GB) in {time.time() - t0:.0f}s")
