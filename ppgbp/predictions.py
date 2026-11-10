"""Write and validate a predictions CSV for the unlabeled hold-out rows (ID, Predicted_SBP, Predicted_DBP)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write(pred: np.ndarray, m_te: pd.DataFrame, path: Path) -> pd.DataFrame:
    """pred: (n_hold-out segments, 2) [SBP, DBP] aligned with m_te rows."""
    out = pd.DataFrame({"ID": m_te.seg_id.values, "Predicted_SBP": pred[:, 0], "Predicted_DBP": pred[:, 1]})
    check(out)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.2f")
    return out


def check(df: pd.DataFrame) -> None:
    assert list(df.columns) == ["ID", "Predicted_SBP", "Predicted_DBP"], df.columns
    assert df.ID.is_unique, "duplicate IDs"
    v = df[["Predicted_SBP", "Predicted_DBP"]].values
    assert np.isfinite(v).all(), "non-finite predictions"
    assert (v[:, 0] > v[:, 1]).all(), "SBP must exceed DBP"
    assert (v[:, 0] > 60).all() and (v[:, 0] < 230).all() and (v[:, 1] > 30).all() and (v[:, 1] < 150).all()
