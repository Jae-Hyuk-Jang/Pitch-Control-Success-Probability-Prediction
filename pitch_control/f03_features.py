"""Row-local F03 batter hist-gap features for inference.

Reads the frozen batter snapshot (official as-of state through 2024) and
derives eight features from the current row only. No cross-row operations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

K_BATTER = 300.0
F03_FEATURES = (
    "f03_batter_hist_success",
    "f03_batter_hist_middle",
    "f03_batter_hist_logn",
    "f03_batter_gap_success",
    "f03_batter_gap_middle",
    "f03_batter_gap_n",
    "f03_batter_rel",
    "f03_batter_known",
)


def load_f03_snapshot(model_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        Path(model_dir) / "f03_batter_snapshot_2025.csv", low_memory=False
    )
    frame["batter_id"] = frame["batter_id"].astype(str)
    snapshot = frame.set_index("batter_id")[["success", "middle", "n"]]
    for column in ("success", "middle", "n"):
        snapshot[column] = pd.to_numeric(snapshot[column], errors="raise")
    return snapshot


def add_f03_features(frame: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    keys = frame["batter_id"].astype("object").astype(str)
    joined = snapshot.reindex(keys)
    hist_success = joined["success"].to_numpy(float)
    hist_middle = joined["middle"].to_numpy(float)
    hist_n = joined["n"].to_numpy(float)
    row_success = pd.to_numeric(
        frame["asof_batter_success_rate"], errors="coerce"
    ).to_numpy(float)
    row_middle = pd.to_numeric(
        frame["asof_batter_middle_rate"], errors="coerce"
    ).to_numpy(float)
    row_n = (
        pd.to_numeric(frame["asof_batter_n"], errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
    )
    known = np.isfinite(hist_n)
    frame["f03_batter_hist_success"] = hist_success
    frame["f03_batter_hist_middle"] = hist_middle
    frame["f03_batter_hist_logn"] = np.log1p(hist_n)
    frame["f03_batter_gap_success"] = np.nan_to_num(
        row_success - hist_success, nan=0.0
    )
    frame["f03_batter_gap_middle"] = np.nan_to_num(
        row_middle - hist_middle, nan=0.0
    )
    frame["f03_batter_gap_n"] = np.nan_to_num(
        np.log1p(row_n) - np.log1p(hist_n), nan=0.0
    )
    frame["f03_batter_rel"] = row_n / (row_n + K_BATTER)
    frame["f03_batter_known"] = known.astype(float)
    return frame
