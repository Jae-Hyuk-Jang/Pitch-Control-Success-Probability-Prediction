"""Component-aux correction: small models are fitted *inside inference* from a frozen train-derived artifact.

The claim worth testing is that this stays row-independent: the fitted state depends only on the frozen artifact,
never on the rows being predicted.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pitch_control import component_aux_corr as aux

FACTOR_COLS = [
    "asof_success_center", "asof_n_log", "prior_success_center", "current_dev", "current_dev_x_rel",
    "prev1_gap", "prev3_gap", "mix_fb_vs_nonfb", "mix_entropy", "two_strike", "high_li", "hand_diff",
]
COMPONENTS = ["reverse", "middle", "ball", "strike"]


def write_artifact(path, scale: float = 0.05, rows: int = 2500, seed: int = 0):
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(rows, len(FACTOR_COLS))).astype(np.float32)
    arrays = {"factor_raw": factor, "z_resid": (0.3 * factor[:, 0] + rng.normal(size=rows)).astype(np.float32),
              "weight": np.ones(rows, dtype=np.float32)}
    for index, name in enumerate(COMPONENTS):
        logit = 0.8 * factor[:, index] + 0.2 * rng.normal(size=rows)
        arrays[f"y_{name}"] = (rng.random(rows) < 1 / (1 + np.exp(-logit))).astype(float)
    np.savez(path / "component_aux_train_2024.npz", **arrays)
    (path / "component_aux_corr_manifest.json").write_text(
        json.dumps({"components": COMPONENTS, "factor_cols": FACTOR_COLS, "scale": scale}), encoding="utf-8")


def test_without_the_frozen_artifact_the_probability_is_returned_unchanged(frame, tmp_path):
    p = np.linspace(0.2, 0.8, len(frame))
    out = aux.apply_component_aux_corr(p, frame, tmp_path)
    np.testing.assert_array_equal(out, p)


def test_logit_correction_is_bounded_and_sign_consistent(frame, tmp_path):
    write_artifact(tmp_path, scale=0.05)
    p = np.random.default_rng(4).uniform(0.3, 0.7, len(frame))
    out = aux.apply_component_aux_corr(p, frame, tmp_path)
    assert out.shape == p.shape and np.isfinite(out).all() and ((out >= 0) & (out <= 1)).all()
    shift = np.log(out / (1 - out)) - np.log(p / (1 - p))
    assert np.abs(shift).max() < 0.5                       # scale 0.05 x a standardised residual prediction
    assert np.abs(shift).max() > 0                         # the correction is not a no-op


def test_zero_scale_is_the_identity(frame, tmp_path):
    write_artifact(tmp_path, scale=0.0, seed=1)
    p = np.random.default_rng(5).uniform(0.2, 0.8, len(frame))
    np.testing.assert_allclose(aux.apply_component_aux_corr(p, frame, tmp_path), p, atol=1e-12)


def test_batch_composition_does_not_change_a_rows_prediction(frame, tmp_path):
    write_artifact(tmp_path, seed=2)
    p = np.random.default_rng(6).uniform(0.3, 0.7, len(frame))
    full = aux.apply_component_aux_corr(p, frame, tmp_path)
    idx = np.array([3, 17, 120, 222, 399])
    part = aux.apply_component_aux_corr(p[idx], frame.iloc[idx].reset_index(drop=True), tmp_path)
    np.testing.assert_allclose(part, full[idx], rtol=0, atol=1e-12)
    perm = np.random.default_rng(7).permutation(len(frame))
    shuffled = aux.apply_component_aux_corr(p[perm], frame.iloc[perm].reset_index(drop=True), tmp_path)
    restored = np.empty_like(full)
    restored[perm] = shuffled
    np.testing.assert_allclose(restored, full, rtol=0, atol=1e-12)


def test_factor_features_use_only_the_row_and_the_preseason_snapshot(frame, tmp_path):
    full = aux._add_factor_features(frame, tmp_path)
    alone = aux._add_factor_features(frame.iloc[[5]].reset_index(drop=True), tmp_path)
    for col in FACTOR_COLS:
        a, b = float(alone.loc[0, col]), float(full.loc[5, col])
        assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, abs=1e-12), col


def test_standardisation_statistics_come_from_the_training_block_only():
    train = np.array([[1.0, 10.0], [3.0, 10.0], [5.0, np.nan]], dtype=np.float32)
    test = np.array([[100.0, np.nan]], dtype=np.float32)
    tr, te = aux._standardize_train_test(train, test)
    assert np.allclose(tr.mean(axis=0), 0.0, atol=1e-6)
    # a test row never moves the statistics: shifting it leaves the training block untouched
    tr2, _ = aux._standardize_train_test(train, test + 1000.0)
    np.testing.assert_array_equal(tr, tr2)
    assert np.isfinite(te).all()                           # missing values fall back to the train median
