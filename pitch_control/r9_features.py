"""Row-local R9 same-hand reverse-trait features for inference.

Reads the frozen pitcher trait table (fitted from official train through
2024) and derives five features from the current row only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

R9_FEATURES = (
    "r9_same_hand_reverse_contrast",
    "r9_same_hand_reverse_reliability",
    "r9_same_hand_reverse_known",
    "r9_same_hand_reverse_oriented_current_signal",
    "r9_same_hand_reverse_oriented_current_signal_x_reliability",
)


def load_r9_artifact(model_dir: Path) -> dict[str, object]:
    model_dir = Path(model_dir)
    manifest = json.loads(
        (model_dir / "r9_trait_manifest.json").read_text(encoding="utf-8")
    )
    table = pd.read_csv(model_dir / "r9_trait_2025.csv", low_memory=False)
    table["pitcher_id"] = table["pitcher_id"].astype(str)
    table = table.set_index("pitcher_id")
    for column in ("contrast", "contrast_reliability"):
        table[column] = pd.to_numeric(table[column], errors="raise")
    table["known_contrast"] = table["known_contrast"].astype(bool)
    return {"table": table, "q": float(manifest["q"])}


def _normalize_hand(series: pd.Series) -> pd.Series:
    value = series.astype(str).str.strip().str.upper()
    normalized = value.replace(
        {"1": "L", "1.0": "L", "LEFT": "L", "2": "R", "2.0": "R", "RIGHT": "R"}
    )
    return normalized.where(series.notna(), "__NA__")


def same_hand_state(frame: pd.DataFrame) -> np.ndarray:
    pitcher_hand = _normalize_hand(frame["pitcher_hand"])
    batter_hand = _normalize_hand(frame["batter_hand"])
    valid = pitcher_hand.isin(["L", "R"]) & batter_hand.isin(["L", "R"])
    return (valid & pitcher_hand.eq(batter_hand)).to_numpy(np.int8)


def add_r9_features(frame: pd.DataFrame, artifact: dict[str, object]) -> pd.DataFrame:
    table: pd.DataFrame = artifact["table"]
    q = float(artifact["q"])
    state = same_hand_state(frame)
    pitcher = frame["pitcher_id"].astype(str)
    contrast = (
        pd.to_numeric(pitcher.map(table["contrast"]), errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
    )
    reliability = (
        pd.to_numeric(pitcher.map(table["contrast_reliability"]), errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
    )
    known = pitcher.map(table["known_contrast"]).fillna(False).to_numpy(bool)
    signal = -contrast * (state.astype(float) - q)
    frame["r9_same_hand_reverse_contrast"] = contrast.astype("float32")
    frame["r9_same_hand_reverse_reliability"] = reliability.astype("float32")
    frame["r9_same_hand_reverse_known"] = known.astype("float32")
    frame["r9_same_hand_reverse_oriented_current_signal"] = signal.astype("float32")
    frame["r9_same_hand_reverse_oriented_current_signal_x_reliability"] = (
        signal * reliability
    ).astype("float32")
    return frame
