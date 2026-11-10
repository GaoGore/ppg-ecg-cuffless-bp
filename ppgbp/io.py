"""Reading the dataset's .mat files.

Round ordering and target construction follow the data provider's loader
exactly, because they define which row ID ("subject_<id>_<round>")
belongs to which recording and label.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("PPGBP_DATA", PROJECT_DIR / "data"))

FS = 500          # PPG / ECG sampling rate (Hz)
ACC_FS = 100      # accelerometer sampling rate (Hz)
SIG_LEN = 30000   # samples per 1-min PPG/ECG segment
WAVELENGTHS = ("IR", "R", "G", "B")
PPG_CHANNELS = [f"PPG_{w}_{i}" for w in WAVELENGTHS for i in (1, 2, 3, 4)]
ACC_CHANNELS = ["Acc_x", "Acc_y", "Acc_z"]


def subject_paths(split: str, data_dir: Path = DATA_DIR) -> list[Path]:
    return sorted((Path(data_dir) / split).glob("*.mat"))


def seg_id(sid: str, rnd: int) -> str:
    return f"subject_{sid}_{rnd}"


def round_keys(data) -> list[str]:
    """Same rule as the data provider's loader: every field starting with 'round',
    sorted by the integer formed from all digits in its name."""
    rk = [f for f in data._fieldnames if f.lower().startswith("round")]
    return sorted(rk, key=lambda r: int("".join(ch for ch in r if ch.isdigit())))


def _text(x) -> str:
    if isinstance(x, np.ndarray):
        return "" if x.size == 0 else _text(x.ravel()[0])
    return str(x).strip()


def _num(x) -> float:
    try:
        return float(_text(x))
    except ValueError:
        return float("nan")


def parse_demo(demo) -> dict:
    fields = {f.lower(): f for f in demo._fieldnames}

    def get(name):
        return getattr(demo, fields[name]) if name in fields else ""

    return {
        "age": _num(get("age")),
        "height": _num(get("height")),
        "weight": _num(get("weight")),
        "male": float(_text(get("sex")).lower().startswith("m")),
        "med_his": float(_text(get("medicationhis")).lower().startswith("y")),
        "med_today": float(_text(get("medicationtoday")).lower().startswith("y")),
    }


def segment_labels(sbpdbp, n_rounds: int) -> np.ndarray:
    """(n_rounds, 2) array of [SBP, DBP]: mean of the cuff readings before and
    after each round (cast to float first: the raw arrays are uint8)."""
    bp = np.asarray(sbpdbp, dtype=float).reshape(-1, 2)
    out = np.full((n_rounds, 2), np.nan)
    for i in range(1, n_rounds + 1):
        if i < bp.shape[0]:
            out[i - 1] = (bp[i - 1] + bp[i]) / 2.0
    return out


def load_subject(path: Path, with_target: bool | None = None) -> dict:
    """Load one subject file.

    Returns sid, demo, labels ([SBP, DBP] per round, NaN without target) and
    rounds: list of dicts with ppg (16, L) float64, ecg (L,), acc (3, La) and key.
    """
    path = Path(path)
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    data = mat["data"]
    keys = round_keys(data)
    rounds = []
    for k in keys:
        rs = getattr(data, k)
        names = set(rs._fieldnames)
        ppg = np.stack([np.asarray(getattr(rs, c), dtype=np.float64).ravel() if c in names
                        else np.full(SIG_LEN, np.nan) for c in PPG_CHANNELS])
        ecg = np.asarray(getattr(rs, "ECG"), dtype=np.float64).ravel() if "ECG" in names else np.full(SIG_LEN, np.nan)
        acc = [np.asarray(getattr(rs, c), dtype=np.float64).ravel() if c in names else None for c in ACC_CHANNELS]
        la = min((a.size for a in acc if a is not None), default=0)
        acc = np.stack([a[:la] if a is not None else np.full(la, np.nan) for a in acc]) if la else np.zeros((3, 0))
        rounds.append({"key": k, "ppg": ppg, "ecg": ecg, "acc": acc})
    if with_target is None:
        with_target = "sbpdbp" in mat
    labels = segment_labels(mat["sbpdbp"], len(keys)) if with_target else np.full((len(keys), 2), np.nan)
    return {"sid": path.stem, "demo": parse_demo(mat["DemoInfo"]), "labels": labels, "rounds": rounds,
            "readings": np.asarray(mat["sbpdbp"], float).reshape(-1, 2) if with_target else None}


def load_light(path: Path) -> dict:
    """Demographics, labels and round keys without converting the signals."""
    path = Path(path)
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = round_keys(mat["data"])
    with_target = "sbpdbp" in mat
    labels = segment_labels(mat["sbpdbp"], len(keys)) if with_target else np.full((len(keys), 2), np.nan)
    return {"sid": path.stem, "demo": parse_demo(mat["DemoInfo"]), "labels": labels, "keys": keys}
