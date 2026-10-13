"""Row-independent inference graph of the final solution.

Pipeline: v3 features -> CE / current-state / F03 / R9 / latent-command features -> CatBoost R5 and R8_DROP_BALL
(base + history_current_gap families) -> R_CAND corrector -> asymmetric M1 (R rows) -> DL member (R rows)
-> fixed tail terms (R rows) -> component-aux correction.
Every prediction is a function of the row itself and frozen, train-derived artifacts in `model_dir`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .asym_m1_predictor import AsymmetricM1Predictor
from .ce_features import add_ce_features
from .component_aux_corr import apply_component_aux_corr
from .current_state_features import add_current_state_features
from .dl_convex_predictor import DLConvexPredictor
from .f03_features import add_f03_features, load_f03_snapshot
from .history_features import add_history_artifacts
from .latent_command_features import add_latent_command_features, load_latent_command_artifacts
from .r9_features import add_r9_features, load_r9_artifact
from .r_cand_features import load_r_cand_artifacts, predict_r_cand_correction
from .tail_terms import _tail_cold_mask, _tail_prior_rate, apply_tail_terms
from .v3_features import engineer_features, normalize_frame_for_catboost

ROW_ID = "row_id"
TARGET = "control_success"
MODEL_DIR = Path(os.environ.get("PITCH_CONTROL_MODEL_DIR", "model"))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))



class Predictor:
    def __init__(self, model_dir: Path = MODEL_DIR, threads: int = 6) -> None:
        self.model_dir = Path(model_dir)
        self.threads = int(threads)
        self.manifest = _read_json(self.model_dir / "feature_manifest.json")
        self.current_manifest = _read_json(
            self.model_dir / "current_state_manifest.json"
        )
        self.ce_manifest = _read_json(self.model_dir / "ce_manifest.json")
        self.calibration = _read_json(
            self.model_dir / "r5_calibration_manifest.json"
        )
        self.latent_command = load_latent_command_artifacts(self.model_dir)
        self.f03_snapshot = load_f03_snapshot(self.model_dir)
        self.r9_artifact = load_r9_artifact(self.model_dir)
        self.r_cand = load_r_cand_artifacts(self.model_dir)
        self.asym_m1 = AsymmetricM1Predictor(self.model_dir)
        self.tail_terms = _read_json(self.model_dir / "tail_terms_manifest.json")
        self.dl_convex = _read_json(self.model_dir / "dl_convex_manifest.json")
        self.dl = DLConvexPredictor(self.model_dir, threads=self.threads)

        weights = self.manifest["component_weights"]
        self.r5_weight = float(weights["R5"])
        self.r8_weight = float(weights["R8_DROP_BALL"])
        if abs(self.r5_weight - 0.938127196) > 1e-12 or abs(self.r8_weight - 0.061872804) > 1e-12:
            raise RuntimeError("component weight contract changed")
        self.offset = float(self.calibration["offset"])
        if abs(self.offset - float(self.manifest["r5_offset"])) > 1e-15:
            raise RuntimeError("calibration offset contract changed")

        self.history = pd.read_csv(
            self.model_dir / "history_artifact_2025.csv", low_memory=False
        )
        self.current_snapshot = pd.read_csv(
            self.model_dir / "current_state_preseason_2025.csv", low_memory=False
        )
        self.ce_entity = pd.read_csv(
            self.model_dir / "ce_pitcher_entity_2025.csv", low_memory=False
        )
        self.ce_cell = pd.read_csv(
            self.model_dir / "ce_platoon_cell_2025.csv", low_memory=False
        )

        self.r5_base_models = self._load_family("r5_v3_base")
        self.r5_history_models = self._load_family("r5_hist_current_gap")
        self.r8_base_models = self._load_family("r8_drop_ball_v3_base")
        self.r8_history_models = self._load_family(
            "r8_drop_ball_hist_current_gap"
        )

    def _load_family(self, prefix: str) -> list[CatBoostClassifier]:
        models = []
        for seed in self.manifest["seeds"]:
            model = CatBoostClassifier()
            model.load_model(
                str(self.model_dir / f"{prefix}_seed{int(seed)}.cbm")
            )
            models.append(model)
        return models

    def _predict_family(
        self,
        models: list[CatBoostClassifier],
        frame: pd.DataFrame,
        features: list[str],
        categorical: list[str],
    ) -> np.ndarray:
        model_frame = normalize_frame_for_catboost(
            frame, list(features), list(categorical)
        )
        success_index = int(self.manifest["success_class_index"])
        predictions = []
        for model in models:
            probability = np.asarray(
                model.predict_proba(model_frame, thread_count=self.threads),
                dtype=float,
            )
            if probability.ndim != 2 or probability.shape[1] <= success_index:
                raise RuntimeError(f"unexpected probability shape: {probability.shape}")
            predictions.append(probability[:, success_index])
        return np.mean(np.stack(predictions), axis=0)

    def _component_raw(
        self,
        base: pd.DataFrame,
        history: pd.DataFrame,
        base_models: list[CatBoostClassifier],
        history_models: list[CatBoostClassifier],
        base_features: list[str],
        base_categorical: list[str],
        history_features: list[str],
        history_categorical: list[str],
    ) -> np.ndarray:
        base_prediction = self._predict_family(
            base_models, base, base_features, base_categorical
        )
        history_prediction = self._predict_family(
            history_models, history, history_features, history_categorical
        )
        history_weight = float(self.manifest["history_weight"])
        return (
            (1.0 - history_weight) * base_prediction
            + history_weight * history_prediction
        )

    def predict_components(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        required = {ROW_ID, "season", "pitcher_id"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"missing test columns: {missing}")

        base = engineer_features(frame.copy())
        base = add_ce_features(
            base,
            self.ce_entity,
            self.ce_cell,
            float(self.ce_manifest["global_rate"]),
        )
        base = add_current_state_features(
            base,
            self.current_snapshot,
            dict(self.current_manifest["global_priors"]),
        )
        base = add_f03_features(base, self.f03_snapshot)
        base = add_r9_features(base, self.r9_artifact)
        augmented = add_latent_command_features(base, self.latent_command)
        augmented_history = add_history_artifacts(augmented, self.history)

        r5_raw = self._component_raw(
            augmented,
            augmented_history,
            self.r5_base_models,
            self.r5_history_models,
            self.manifest["r5_base_features"],
            self.manifest["r5_base_cat_features"],
            self.manifest["r5_history_features"],
            self.manifest["r5_history_cat_features"],
        )
        r8_raw = self._component_raw(
            augmented,
            augmented_history,
            self.r8_base_models,
            self.r8_history_models,
            self.manifest["r8_base_features"],
            self.manifest["r8_base_cat_features"],
            self.manifest["r8_history_features"],
            self.manifest["r8_history_cat_features"],
        )
        r5_calibrated = np.clip(r5_raw + self.offset, 0.0, 1.0)
        r8_calibrated = np.clip(r8_raw + self.offset, 0.0, 1.0)
        blended = (
            self.r5_weight * r5_calibrated + self.r8_weight * r8_calibrated
        )
        r_cand_correction = predict_r_cand_correction(
            frame, self.r_cand, self.threads
        )
        cap = float(self.r_cand["manifest"]["correction_cap"])
        if r_cand_correction.shape != (len(frame),):
            raise RuntimeError("r_cand correction length mismatch")
        if np.max(np.abs(r_cand_correction)) > cap + 1.0e-9:
            raise RuntimeError("r_cand correction outside cap")
        alpha = float(self.r_cand["manifest"]["alpha"])
        variant = self.r_cand["manifest"].get("variant")
        if variant is None:
            combined = alpha * r_cand_correction
        else:
            variant_correction = predict_r_cand_correction(
                frame, self.r_cand, self.threads, family="variant"
            )
            if variant_correction.shape != (len(frame),):
                raise RuntimeError("r_cand variant correction length mismatch")
            if np.max(np.abs(variant_correction)) > cap + 1.0e-9:
                raise RuntimeError("r_cand variant correction outside cap")
            beta = float(variant["beta"])
            combined = alpha * r_cand_correction + beta * variant_correction
        final = np.clip(blended + combined, 0.0, 1.0)
        game_type = (
            frame["game_type"].astype(str).str.strip().str.upper().to_numpy()
        )
        is_r = game_type == "R"
        if is_r.any():
            m1_probability = self.asym_m1.predict(
                frame.loc[is_r].reset_index(drop=True)
            )
            final[is_r] = np.clip(
                0.75 * final[is_r] + 0.25 * m1_probability, 0.0, 1.0
            )
            # DL_CONVEX: row-local deployable DL member on R rows (after M1 convex, before tail terms)
            dl_w = float(self.dl_convex["w"])
            dl_probability = self.dl.predict(frame.loc[is_r].reset_index(drop=True))
            if dl_probability.shape != (int(is_r.sum()),):
                raise RuntimeError("dl member length mismatch")
            final[is_r] = np.clip((1.0 - dl_w) * final[is_r] + dl_w * dl_probability, 0.0, 1.0)
            # TAIL_TERMS: fixed row-local logit terms, R rows only
            tail_cold = _tail_cold_mask(frame, self.asym_m1.tables)
            tail_prior = _tail_prior_rate(frame, self.model_dir) if any(t["type"] == "asof_minus_prior" for t in self.tail_terms["terms"]) else None
            final = apply_tail_terms(final, frame, is_r, self.tail_terms, tail_cold, tail_prior)

        final = apply_component_aux_corr(final, frame, self.model_dir)

        components = {
            "r5_raw": r5_raw,
            "r8_drop_ball_raw": r8_raw,
            "r5_calibrated": r5_calibrated,
            "r8_drop_ball_calibrated": r8_calibrated,
            "final": final,
        }
        for name, values in components.items():
            if values.shape != (len(frame),):
                raise RuntimeError(f"{name} prediction length mismatch")
            if not np.isfinite(values).all():
                raise RuntimeError(f"{name} contains non-finite predictions")
            if values.min(initial=0.0) < 0.0 or values.max(initial=1.0) > 1.0:
                raise RuntimeError(f"{name} outside [0, 1]")
        return components

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_components(frame)["final"]

