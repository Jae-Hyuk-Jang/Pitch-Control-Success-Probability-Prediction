from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


SEEDS = tuple(range(20260821, 20260829))
ALPHA = 1.2
BETA = -0.3
CORRECTION_CAP = 0.10

CATBOOST_PARAMS = {
    "iterations": 160,
    "depth": 8,
    "learning_rate": 0.05,
    "l2_leaf_reg": 8.0,
    "loss_function": "RMSE",
    "verbose": 0,
    "allow_writing_files": False,
}


def centered_residual_target(
    label: np.ndarray,
    anchor_oof: np.ndarray,
    game_type: pd.Series,
) -> np.ndarray:
    residual = np.asarray(label, dtype=float) - np.asarray(anchor_oof, dtype=float)
    groups = pd.Series(game_type).fillna("__MISSING__").astype(str)
    level = pd.Series(residual).groupby(groups, sort=False).transform("mean")
    return residual - level.to_numpy(dtype=float)


def fit_corrector_family(
    features: pd.DataFrame,
    target: np.ndarray,
    categorical: list[str],
) -> list[CatBoostRegressor]:
    models = []
    for seed in SEEDS:
        model = CatBoostRegressor(**CATBOOST_PARAMS, random_seed=seed)
        model.fit(features, target, cat_features=categorical)
        models.append(model)
    return models


def predict_correction(
    models: list[CatBoostRegressor], features: pd.DataFrame
) -> np.ndarray:
    values = [
        np.clip(model.predict(features), -CORRECTION_CAP, CORRECTION_CAP)
        for model in models
    ]
    return np.mean(np.stack(values), axis=0)


def combine_rc3(
    a4_anchor: np.ndarray,
    base_corrector: np.ndarray,
    recent_h2_corrector: np.ndarray,
) -> np.ndarray:
    return np.clip(
        np.asarray(a4_anchor, dtype=float)
        + ALPHA * np.asarray(base_corrector, dtype=float)
        + BETA * np.asarray(recent_h2_corrector, dtype=float),
        0.0,
        1.0,
    )
