from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .v3_features import normalize_frame_for_catboost


PITCHER = "pitcher_id"
GAME_TYPE = "game_type"
DEFAULT_PITCHER = "__DEFAULT__"
GT_ROWS = "hist_current_gt_rows"
GT_MEAN = "hist_current_gt_residual_mean"


def _string_key(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), "__NA__").astype(str)


def load_r_cand_artifacts(model_dir: Path) -> dict:
    manifest = json.loads(
        (Path(model_dir) / "r_cand_manifest.json").read_text(encoding="utf-8")
    )
    table = pd.read_csv(
        Path(model_dir) / "r_cand_aggregate_2025.csv", low_memory=False
    )
    table["pitcher_id"] = table["pitcher_id"].astype(str)
    table["game_type"] = table["game_type"].astype(str)
    agg_columns = list(manifest["aggregate_columns"])
    default_rows = table.loc[table["pitcher_id"].eq(DEFAULT_PITCHER)]
    if len(default_rows) != 1:
        raise RuntimeError("aggregate table default row missing or duplicated")
    entries = table.loc[~table["pitcher_id"].eq(DEFAULT_PITCHER)]
    if entries.duplicated(subset=["pitcher_id", "game_type"]).any():
        raise RuntimeError("aggregate table has duplicate keys")
    models: list[CatBoostRegressor] = []
    for seed in manifest["seeds"]:
        model = CatBoostRegressor()
        model.load_model(
            str(Path(model_dir) / f"r_cand_correction_seed{int(seed)}.cbm")
        )
        models.append(model)
    variant_models: list[CatBoostRegressor] = []
    variant = manifest.get("variant")
    if variant is not None:
        for seed in variant["seeds"]:
            model = CatBoostRegressor()
            model.load_model(
                str(Path(model_dir) / f"{variant['prefix']}_seed{int(seed)}.cbm")
            )
            variant_models.append(model)
    return {
        "manifest": manifest,
        "table": entries.reset_index(drop=True),
        "agg_columns": agg_columns,
        "default_values": default_rows[agg_columns].iloc[0].to_numpy(float),
        "models": models,
        "variant_models": variant_models,
    }


def _join_aggregates(frame: pd.DataFrame, artifacts: dict) -> np.ndarray:
    agg_columns = artifacts["agg_columns"]
    table = artifacts["table"].rename(
        columns={"pitcher_id": "__pitcher_key", "game_type": "__gt_key"}
    )
    keys = pd.DataFrame(
        {
            "__pitcher_key": _string_key(frame[PITCHER]).to_numpy(),
            "__gt_key": _string_key(frame[GAME_TYPE]).to_numpy(),
        }
    )
    exact = keys.merge(table, on=["__pitcher_key", "__gt_key"], how="left")
    if len(exact) != len(frame):
        raise RuntimeError("aggregate join changed row count")
    values = exact[agg_columns].to_numpy(float)
    missing = ~np.isfinite(values[:, 0])
    if missing.any():
        pitcher_rows = table.loc[table["__gt_key"].eq("F")]
        fallback = (
            keys.loc[missing, ["__pitcher_key"]]
            .merge(pitcher_rows, on="__pitcher_key", how="left")
        )
        fallback_values = fallback[agg_columns].to_numpy(float)
        unknown = ~np.isfinite(fallback_values[:, 0])
        gt_rows_index = agg_columns.index(GT_ROWS)
        gt_mean_index = agg_columns.index(GT_MEAN)
        fallback_values[:, gt_rows_index] = 0.0
        fallback_values[:, gt_mean_index] = 0.0
        fallback_values[unknown] = artifacts["default_values"]
        values[missing] = fallback_values
    if not np.isfinite(values).all():
        raise RuntimeError("aggregate join produced non-finite values")
    return values.astype(np.float32)


def predict_r_cand_correction(
    frame: pd.DataFrame, artifacts: dict, threads: int = 6, family: str = "base"
) -> np.ndarray:
    if family == "base":
        models = artifacts["models"]
    elif family == "variant":
        models = artifacts["variant_models"]
        if not models:
            raise RuntimeError("variant family requested but not loaded")
    else:
        raise ValueError(f"unknown correction family: {family}")
    manifest = artifacts["manifest"]
    feature_order = list(manifest["feature_order"])
    categorical = list(manifest["categorical_features"])
    agg_columns = artifacts["agg_columns"]
    agg_set = set(agg_columns)
    base_features = [column for column in feature_order if column not in agg_set]
    missing = sorted(set(base_features) - set(frame.columns))
    if missing:
        raise ValueError(f"missing r_cand input columns: {missing}")
    working = frame[base_features].copy()
    aggregate_values = _join_aggregates(frame, artifacts)
    for index, column in enumerate(agg_columns):
        working[column] = aggregate_values[:, index]
    model_frame = normalize_frame_for_catboost(working, feature_order, categorical)
    cap = float(manifest["correction_cap"])
    predictions = []
    for model in models:
        values = np.asarray(
            model.predict(model_frame, thread_count=int(threads)), dtype=float
        )
        predictions.append(np.clip(values, -cap, cap))
    correction = np.mean(np.stack(predictions), axis=0)
    if correction.shape != (len(frame),) or not np.isfinite(correction).all():
        raise RuntimeError("invalid r_cand correction")
    return correction
