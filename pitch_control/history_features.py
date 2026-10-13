from __future__ import annotations

import numpy as np
import pandas as pd


SEASON = "season"
PITCHER_ID = "pitcher_id"
HISTORY_STATE_COLS = (
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def add_history_artifacts(
    frame: pd.DataFrame, artifacts: pd.DataFrame
) -> pd.DataFrame:
    """Join a frozen train-only 2025 pitcher artifact to each current row."""
    output = frame.copy()
    output["_r5_order"] = np.arange(len(output), dtype=np.int64)
    output[PITCHER_ID] = output[PITCHER_ID].astype(str)
    output[SEASON] = pd.to_numeric(output[SEASON], errors="coerce").astype("Int64")
    artifact = artifacts.copy()
    artifact[PITCHER_ID] = artifact[PITCHER_ID].astype(str)
    artifact[SEASON] = pd.to_numeric(artifact[SEASON], errors="coerce").astype(
        "Int64"
    )
    if artifact.duplicated([PITCHER_ID, SEASON]).any():
        raise ValueError("duplicate key in frozen history artifact")
    output = output.merge(
        artifact,
        on=[PITCHER_ID, SEASON],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    output = output.sort_values("_r5_order", kind="stable").reset_index(drop=True)
    for column in HISTORY_STATE_COLS:
        historical = f"hist_{column}__last"
        if column in output.columns and historical in output.columns:
            output[f"histgap_{column}"] = (
                _numeric(output[column]) - _numeric(output[historical])
            ).astype("float32")
    months = _numeric(
        output.get("hist_months", pd.Series(np.nan, index=output.index))
    )
    output["hist_depth_regime"] = np.select(
        [
            months.isna() | months.eq(0),
            months.lt(3),
            months.lt(6),
            months.lt(12),
        ],
        ["none", "lt3", "lt6", "lt12"],
        default="ge12",
    ).astype(str)
    return output.drop(columns=["_r5_order"], errors="ignore")
