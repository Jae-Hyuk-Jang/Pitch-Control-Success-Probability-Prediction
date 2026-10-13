from __future__ import annotations

import numpy as np
import pandas as pd


CONTROL_AXES = ("success", "reverse", "middle", "ball", "strike")
RATE_COLUMNS = {
    "success": "asof_pitcher_success_rate",
    "reverse": "asof_pitcher_reverse_rate",
    "middle": "asof_pitcher_middle_rate",
    "ball": "asof_pitcher_ball_rate",
    "strike": "asof_pitcher_strike_rate",
}
EB_PRIOR_STRENGTH = {
    "success": 122.0,
    "reverse": 83.0,
    "middle": 310.0,
    "ball": 304.0,
    "strike": 200.0,
}
CURRENT_PRIOR_STRENGTH = {
    "success": 80.0,
    "reverse": 80.0,
    "middle": 120.0,
    "ball": 120.0,
    "strike": 120.0,
}
PITCHER_ID = "pitcher_id"


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _string(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series("__NA__", index=frame.index, dtype=str)
    return (
        frame[column]
        .astype("object")
        .where(frame[column].notna(), "__NA__")
        .astype(str)
    )


def add_current_state_features(
    frame: pd.DataFrame,
    preseason_snapshot: pd.DataFrame,
    global_priors: dict[str, float],
) -> pd.DataFrame:
    """Attach train-frozen current-state fields using each row independently."""
    x = frame.copy()
    x["_r5_order"] = np.arange(len(x), dtype=np.int64)
    x[PITCHER_ID] = _string(x, PITCHER_ID)
    artifact = preseason_snapshot.copy()
    artifact[PITCHER_ID] = _string(artifact, PITCHER_ID)
    if artifact[PITCHER_ID].duplicated().any():
        raise ValueError("duplicate pitcher_id in frozen current-state artifact")
    x = x.merge(artifact, on=PITCHER_ID, how="left", sort=False, validate="many_to_one")
    x = x.sort_values("_r5_order", kind="stable").reset_index(drop=True)

    current_n_total = (
        _numeric(x, "asof_pitcher_n", 0)
        .fillna(0)
        .clip(lower=0)
        .to_numpy(float)
    )
    pre_n_total = (
        _numeric(x, "preseason_n", 0)
        .fillna(0)
        .clip(lower=0)
        .to_numpy(float)
    )
    current_observed_n = np.maximum(current_n_total - pre_n_total, 0.0)
    output: dict[str, np.ndarray] = {}
    for axis in CONTROL_AXES:
        prior = float(global_priors[axis])
        axis_pre_n = (
            _numeric(x, f"pre_{axis}_n", 0)
            .fillna(0)
            .clip(lower=0)
            .to_numpy(float)
        )
        axis_pre_rate = (
            _numeric(x, f"pre_{axis}_rate", prior)
            .fillna(prior)
            .clip(0, 1)
            .to_numpy(float)
        )
        strength = EB_PRIOR_STRENGTH[axis]
        ability_mean = (
            axis_pre_n * axis_pre_rate + strength * prior
        ) / (axis_pre_n + strength)

        current_rate = (
            _numeric(x, RATE_COLUMNS[axis], prior)
            .fillna(prior)
            .clip(0, 1)
            .to_numpy(float)
        )
        current_count = current_n_total * current_rate - axis_pre_n * axis_pre_rate
        current_count = np.clip(current_count, 0.0, current_observed_n)
        current_strength = CURRENT_PRIOR_STRENGTH[axis]
        current_mean = (
            current_count + current_strength * ability_mean
        ) / (current_observed_n + current_strength)
        current_reliability = current_observed_n / (
            current_observed_n + current_strength
        )
        current_delta = current_mean - ability_mean

        output[f"r1_current_{axis}_mean"] = current_mean.astype("float32")
        output[f"r1_current_{axis}_delta"] = current_delta.astype("float32")
        output[f"r1_current_{axis}_reliability"] = current_reliability.astype(
            "float32"
        )
        output[f"r1_current_{axis}_gated_delta"] = (
            current_delta * current_reliability
        ).astype("float32")

        if axis in {"success", "middle"}:
            for horizon, fixed_reliability in ((1, 0.35), (3, 0.65), (5, 0.85)):
                source = f"asof_pitcher_prev{horizon}_game_{axis}_rate"
                available = _numeric(x, source).notna().to_numpy()
                output[f"r1_recent_{axis}_prev{horizon}_available"] = (
                    available.astype("int8")
                )
                output[f"r1_recent_{axis}_prev{horizon}_reliability"] = (
                    available.astype(float) * fixed_reliability
                ).astype("float32")

    output["r1_current_log_n"] = np.log1p(current_observed_n).astype("float32")
    for column, values in output.items():
        x[column] = values
    return x.drop(columns=["_r5_order"], errors="ignore")
