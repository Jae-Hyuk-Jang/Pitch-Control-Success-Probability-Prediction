from __future__ import annotations

import numpy as np
import pandas as pd


PITCHER_ID = "pitcher_id"
AXIS_COLUMN = "platoon_matchup"


def add_ce_features(
    frame: pd.DataFrame,
    entity_artifact: pd.DataFrame,
    cell_artifact: pd.DataFrame,
    global_rate: float,
) -> pd.DataFrame:
    """Map frozen train-only pitcher/platoon rates to each current row."""
    output = frame.copy()
    if AXIS_COLUMN not in output.columns:
        raise KeyError(f"{AXIS_COLUMN} missing; call engineer_features first")
    entity = (
        entity_artifact.assign(
            **{PITCHER_ID: entity_artifact[PITCHER_ID].astype(str)}
        )
        .set_index(PITCHER_ID)["rate"]
        .astype(float)
    )
    cell = cell_artifact.copy()
    cell["_key"] = (
        cell[PITCHER_ID].astype(str) + "|" + cell[AXIS_COLUMN].astype(str)
    )
    if cell["_key"].duplicated().any():
        raise ValueError("duplicate key in frozen CE cell artifact")
    cell = cell.set_index("_key")

    pitcher = output[PITCHER_ID].astype(str)
    key = pitcher + "|" + output[AXIS_COLUMN].astype(str)
    output["ce_p_rate"] = (
        pitcher.map(entity)
        .astype(float)
        .fillna(float(global_rate))
        .astype("float32")
    )
    output["ce_platoon_rate"] = key.map(cell["rate"]).astype(float).astype("float32")
    output["ce_platoon_dev"] = key.map(cell["dev"]).astype(float).astype("float32")
    output["ce_platoon_logn"] = np.log1p(
        key.map(cell["size"]).astype(float).fillna(0.0).to_numpy()
    ).astype("float32")
    return output
