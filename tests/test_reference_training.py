"""Reference implementations of the anchor (5-class CatBoost) and the residual corrector, on synthetic data."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import anchor_reference as anchor
import rc3_corrector_reference as rc3


@pytest.fixture()
def tiny_catboost(monkeypatch):
    monkeypatch.setitem(anchor.CATBOOST_PARAMS, "iterations", 6)
    monkeypatch.setitem(anchor.CATBOOST_PARAMS, "depth", 3)
    monkeypatch.setattr(anchor, "SEEDS", (1, 2))
    monkeypatch.setitem(rc3.CATBOOST_PARAMS, "iterations", 6)
    monkeypatch.setitem(rc3.CATBOOST_PARAMS, "depth", 3)
    monkeypatch.setattr(rc3, "SEEDS", (1, 2, 3))


def test_season_weights_favour_recent_years_and_default_to_one():
    w = anchor.make_sample_weight(pd.Series([2019, 2022, 2024, 1999]))
    assert w.tolist() == [0.65, 1.00, 1.30, 1.00]
    assert list(anchor.SEASON_WEIGHTS.values()) == sorted(anchor.SEASON_WEIGHTS.values())


def test_five_class_anchor_returns_the_success_class_probability(tiny_catboost):
    rng = np.random.default_rng(0)
    n = 400
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.choice(["u", "v"], n)})
    y = rng.integers(0, 5, n)                       # 0 = success, 1..4 = failure types
    y[:5] = [0, 1, 2, 3, 4]
    weights = anchor.make_sample_weight(pd.Series(rng.choice([2022, 2023, 2024], n)))
    models = anchor.fit_family(x, y, ["c"], weights)
    assert len(models) == 2
    p = anchor.predict_success(models, x)
    assert p.shape == (n,) and ((p > 0) & (p < 1)).all()
    # class 0 column of the multiclass output, averaged over seeds
    manual = np.mean([m.predict_proba(x)[:, 0] for m in models], axis=0)
    np.testing.assert_allclose(p, manual)


def test_centered_residual_removes_the_level_per_game_type_only():
    label = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=float)
    anchor_oof = np.array([0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4])
    game_type = pd.Series(["R", "R", "R", "R", "F", "F", "F", "F"])
    target = rc3.centered_residual_target(label, anchor_oof, game_type)
    assert abs(target[:4].mean()) < 1e-12 and abs(target[4:].mean()) < 1e-12
    raw = label - anchor_oof
    np.testing.assert_allclose(target[:4] - target[:4].mean(), raw[:4] - raw[:4].mean())
    np.testing.assert_allclose(target[0] - target[1], raw[0] - raw[1])   # within-group differences survive


def test_correction_is_capped_per_seed_before_averaging():
    huge = SimpleNamespace(predict=lambda X: np.full(len(X), 5.0))
    negative = SimpleNamespace(predict=lambda X: np.full(len(X), -5.0))
    out = rc3.predict_correction([huge, negative, huge], pd.DataFrame({"a": [0, 1, 2]}))
    np.testing.assert_allclose(out, np.full(3, (rc3.CORRECTION_CAP - rc3.CORRECTION_CAP + rc3.CORRECTION_CAP) / 3))


def test_rc3_combination_formula_and_clipping():
    a4 = np.array([0.50, 0.99, 0.01])
    c_b = np.array([0.02, 0.05, -0.05])
    c_v = np.array([0.01, 0.00, 0.00])
    out = rc3.combine_rc3(a4, c_b, c_v)
    want = np.clip(a4 + rc3.ALPHA * c_b + rc3.BETA * c_v, 0, 1)
    np.testing.assert_allclose(out, want)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert rc3.ALPHA == 1.2 and rc3.BETA == -0.3          # final constants of the corrector


def test_corrector_family_trains_on_a_centered_target(tiny_catboost):
    rng = np.random.default_rng(1)
    n = 300
    x = pd.DataFrame({"a": rng.normal(size=n), "g": rng.choice(["R", "F"], n)})
    target = rc3.centered_residual_target(rng.integers(0, 2, n), np.full(n, 0.5), x["g"])
    models = rc3.fit_corrector_family(x, target, ["g"])
    assert len(models) == 3
    correction = rc3.predict_correction(models, x)
    assert correction.shape == (n,) and np.abs(correction).max() <= rc3.CORRECTION_CAP


def test_anchor_blend_uses_the_final_weights_after_the_offset():
    assert abs(anchor.R5_WEIGHT + anchor.R8_WEIGHT - 1.0) < 1e-12 and anchor.R5_WEIGHT > anchor.R8_WEIGHT
    same = np.full(5, 0.5)
    np.testing.assert_allclose(anchor.blend_anchor(same, same), 0.5 + anchor.OFFSET)
    r5, r8 = np.array([0.6]), np.array([0.4])
    want = anchor.R5_WEIGHT * (0.6 + anchor.OFFSET) + anchor.R8_WEIGHT * (0.4 + anchor.OFFSET)
    np.testing.assert_allclose(anchor.blend_anchor(r5, r8), want)
    assert anchor.blend_anchor(np.array([0.0]), np.array([1.0])).min() >= 0.0       # clipped before blending
