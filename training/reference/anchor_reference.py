from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


SEEDS = (20260807, 20260808, 20260809)
SEASON_WEIGHTS = {
    2019: 0.65,
    2020: 0.75,
    2021: 0.85,
    2022: 1.00,
    2023: 1.15,
    2024: 1.30,
}
OFFSET = -0.008815422191623918
R5_WEIGHT = 0.938127196
R8_WEIGHT = 0.061872804

CATBOOST_PARAMS = {
    "iterations": 96,
    "depth": 8,
    "learning_rate": 0.055,
    "l2_leaf_reg": 7.0,
    "loss_function": "MultiClass",
    "classes_count": 5,
    "verbose": 0,
    "allow_writing_files": False,
}


def make_sample_weight(season: pd.Series) -> np.ndarray:
    return season.map(SEASON_WEIGHTS).fillna(1.0).to_numpy(dtype=float)


def fit_family(
    features: pd.DataFrame,
    target_5class: np.ndarray,
    categorical: list[str],
    sample_weight: np.ndarray,
) -> list[CatBoostClassifier]:
    models = []
    for seed in SEEDS:
        model = CatBoostClassifier(**CATBOOST_PARAMS, random_seed=seed)
        model.fit(
            features,
            target_5class,
            cat_features=categorical,
            sample_weight=sample_weight,
        )
        models.append(model)
    return models


def predict_success(
    models: list[CatBoostClassifier], features: pd.DataFrame
) -> np.ndarray:
    probabilities = [model.predict_proba(features)[:, 0] for model in models]
    return np.mean(np.stack(probabilities), axis=0)


def blend_anchor(r5_raw: np.ndarray, r8_drop_ball_raw: np.ndarray) -> np.ndarray:
    r5 = np.clip(np.asarray(r5_raw, dtype=float) + OFFSET, 0.0, 1.0)
    r8 = np.clip(np.asarray(r8_drop_ball_raw, dtype=float) + OFFSET, 0.0, 1.0)
    return R5_WEIGHT * r5 + R8_WEIGHT * r8
