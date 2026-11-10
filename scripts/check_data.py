"""Sanity checks on the raw data: labels, round ordering, row IDs, duplicates."""
import hashlib
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppgbp import io  # noqa: E402


def reference_labels(path):
    """Independent re-implementation of the target rule (label = mean of the cuff readings before and after a round)."""
    from scipy.io import loadmat
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    rk = [f for f in m["data"]._fieldnames if f.lower().startswith("round")]
    rk = sorted(rk, key=lambda r: int("".join(ch for ch in r if ch.isdigit())))
    s = np.asarray(m["sbpdbp"], dtype=float)
    s = s.reshape(-1, 2) if s.ndim == 1 else s
    out = []
    for i, _ in enumerate(rk, start=1):
        if i < s.shape[0]:
            out.append(((s[i - 1, 0] + s[i, 0]) / 2.0, (s[i - 1, 1] + s[i, 1]) / 2.0))
        else:
            out.append((np.nan, np.nan))
    return rk, np.array(out)


def segment_hashes(path):
    subj = io.load_subject(path)
    rows = []
    for r, rd in enumerate(subj["rounds"], start=1):
        h = hashlib.sha1(np.ascontiguousarray(rd["ppg"]).tobytes() + np.ascontiguousarray(rd["ecg"]).tobytes()).hexdigest()
        rows.append((path.parent.name, subj["sid"], r, rd["key"], h, rd["ppg"].shape[1], rd["acc"].shape[1]))
    return rows


def main():
    train, test = io.subject_paths("train"), io.subject_paths("test")
    print(f"train files {len(train)}, test files {len(test)}")
    # 1) labels vs an independent re-implementation of the target rule
    n_seg, bad, nonstd_keys = 0, 0, []
    for p in train:
        light = io.load_light(p)
        rk, lab = reference_labels(p)
        assert light["keys"] == rk, p
        n_seg += len(rk)
        if not np.allclose(light["labels"], lab, equal_nan=True):
            bad += 1
        nonstd_keys += [k for k in rk if not k[5:].isdigit()]
    print(f"train segments {n_seg}; label mismatches {bad}; non-standard round keys {len(nonstd_keys)} e.g. {nonstd_keys[:3]}")
    # 2) hold-out row IDs are unique
    ids = [io.seg_id(io.load_light(p)["sid"], r) for p in test for r in range(1, len(io.load_light(p)["keys"]) + 1)]
    print(f"hold-out segments {len(ids)}; unique IDs {len(set(ids)) == len(ids)}")
    # 3) exact duplicate recordings (train and test)
    with mp.get_context("fork").Pool(12) as pool:
        rows = [r for rr in pool.map(segment_hashes, train + test, chunksize=4) for r in rr]
    df = pd.DataFrame(rows, columns=["split", "sid", "round", "key", "hash", "len", "acc_len"])
    dup = df[df.duplicated("hash", keep=False)].sort_values(["hash", "split", "sid"])
    print(f"duplicate segments: {len(dup)}")
    if len(dup):
        print(dup.groupby("hash").apply(lambda g: list(zip(g.split, g.sid, g["round"]))).to_string())
    print("signal lengths:", df["len"].value_counts().to_dict(), "acc lengths:", df["acc_len"].value_counts().to_dict())


if __name__ == "__main__":
    main()
