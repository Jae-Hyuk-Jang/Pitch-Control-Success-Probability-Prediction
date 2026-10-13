"""Row-local logit terms: logit(p') = logit(p) + scale * sum_j gamma_j * x_j."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pitch_control.tail_terms import _tail_cold_mask, _tail_prior_rate, _term_core, _term_value, apply_tail_terms

SPEC = {
    "scale": 1.0,
    "terms": [
        {"type": "linear", "column": "asof_pitcher_success_rate", "center": 0.52, "gamma": 0.2, "mask": "cold", "requires_asof_n": True},
        {"type": "linear", "column": "asof_pitcher_success_rate", "center": 0.52, "gamma": 0.5, "mask": "known", "requires_asof_n": True},
        {"type": "asof_minus_prior", "column": "asof_pitcher_success_rate", "gamma": 0.39, "mask": "known"},
        {"type": "count", "balls": 3, "strikes": 0, "gamma": 0.1427},
    ],
}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


@pytest.fixture()
def setup(frame):
    rng = np.random.default_rng(3)
    p = rng.uniform(0.3, 0.7, len(frame))
    is_r = (frame["game_type"] == "R").to_numpy()
    cold = rng.random(len(frame)) < 0.2
    prior = rng.uniform(0.45, 0.6, len(frame))
    return p, is_r, cold, prior


def expected_shift(frame, cold, prior):
    rate = frame["asof_pitcher_success_rate"].to_numpy(float)
    n = frame["asof_pitcher_n"].fillna(0).to_numpy(float)
    known_rate = np.isfinite(rate) & (n > 0)
    t_cold = np.where(known_rate, np.nan_to_num(rate) - 0.52, 0.0) * 0.2 * cold
    t_known = np.where(known_rate, np.nan_to_num(rate) - 0.52, 0.0) * 0.5 * ~cold
    dev_ok = np.isfinite(rate) & (n > 0) & np.isfinite(prior)
    t_dev = np.where(dev_ok, np.nan_to_num(rate) - np.nan_to_num(prior), 0.0) * 0.39 * ~cold
    t_count = ((frame["balls_before"] == 3) & (frame["strikes_before"] == 0)).to_numpy(float) * 0.1427
    return t_cold + t_known + t_dev + t_count


def test_matches_the_closed_form_on_r_rows_and_leaves_f_rows_alone(frame, setup):
    p, is_r, cold, prior = setup
    out = apply_tail_terms(p, frame, is_r, SPEC, cold, prior)
    want = sigmoid(logit(p) + expected_shift(frame, cold, prior))
    np.testing.assert_allclose(out[is_r], want[is_r], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(out[~is_r], p[~is_r])


def test_input_probabilities_are_not_modified_in_place(frame, setup):
    p, is_r, cold, prior = setup
    snapshot = p.copy()
    apply_tail_terms(p, frame, is_r, SPEC, cold, prior)
    np.testing.assert_array_equal(p, snapshot)


def test_f_intercept_is_a_constant_logit_shift_on_f_rows(frame, setup):
    p, is_r, cold, prior = setup
    spec = {**SPEC, "scale": 0.5, "f_intercept": -0.4}
    out = apply_tail_terms(p, frame, is_r, spec, cold, prior)
    want_f = sigmoid(logit(p) + 0.5 * -0.4)
    np.testing.assert_allclose(out[~is_r], want_f[~is_r], atol=1e-12)


def test_scale_zero_is_the_identity(frame, setup):
    p, is_r, cold, prior = setup
    out = apply_tail_terms(p, frame, is_r, {**SPEC, "scale": 0.0}, cold, prior)
    np.testing.assert_allclose(out, np.clip(p, 1e-6, 1 - 1e-6), atol=1e-12)


def test_every_row_is_a_function_of_itself_only(frame, setup):
    p, is_r, cold, prior = setup
    full = apply_tail_terms(p, frame, is_r, SPEC, cold, prior)
    rng = np.random.default_rng(9)
    perm = rng.permutation(len(frame))
    shuffled = apply_tail_terms(p[perm], frame.iloc[perm].reset_index(drop=True), is_r[perm], SPEC, cold[perm], prior[perm])
    restored = np.empty_like(full)
    restored[perm] = shuffled
    np.testing.assert_array_equal(full, restored)
    for i in (0, 11, 200):
        one = apply_tail_terms(p[[i]], frame.iloc[[i]].reset_index(drop=True), is_r[[i]], SPEC, cold[[i]], prior[[i]])
        assert one[0] == full[i]


def test_masks_and_sample_size_window(frame):
    cold = np.zeros(len(frame), dtype=bool)
    cold[::2] = True
    base = {"type": "linear", "column": "asof_pitcher_success_rate", "center": 0.5, "gamma": 1.0}
    only_cold = _term_value(frame, {**base, "mask": "cold"}, cold)
    only_known = _term_value(frame, {**base, "mask": "known"}, cold)
    assert (only_cold[~cold] == 0).all() and (only_known[cold] == 0).all()
    window = _term_value(frame, {**base, "n_min": 500, "n_max": 1500}, cold)
    n = frame["asof_pitcher_n"].fillna(0).to_numpy(float)
    assert (window[(n < 500) | (n >= 1500)] == 0).all()
    with pytest.raises(ValueError):
        _term_value(frame, {**base, "mask": "warm"}, cold)


def test_hinge_and_unknown_kinds(frame):
    hinge = _term_core(frame, {"type": "hinge", "column": "asof_pitcher_success_rate", "threshold": 0.55})
    rate = frame["asof_pitcher_success_rate"].to_numpy(float)
    n = frame["asof_pitcher_n"].fillna(0).to_numpy(float)
    want = np.where(np.isfinite(rate) & (n > 0), np.maximum(np.nan_to_num(rate) - 0.55, 0.0), 0.0)
    np.testing.assert_allclose(hinge, want)
    with pytest.raises(ValueError):
        _term_core(frame, {"type": "quadratic", "column": "asof_pitcher_success_rate"})


def test_positive_gamma_raises_probability_when_the_running_rate_is_above_center(frame):
    spec = {"scale": 1.0, "terms": [{"type": "linear", "column": "asof_pitcher_success_rate", "center": 0.5, "gamma": 1.0}]}
    p = np.full(len(frame), 0.5)
    out = apply_tail_terms(p, frame, np.ones(len(frame), dtype=bool), spec, np.zeros(len(frame), dtype=bool))
    rate = frame["asof_pitcher_success_rate"].to_numpy(float)
    n = frame["asof_pitcher_n"].fillna(0).to_numpy(float)
    live = np.isfinite(rate) & (n > 0)
    assert (out[live & (rate > 0.5)] > 0.5).all()
    assert (out[live & (rate < 0.5)] < 0.5).all()
    assert (out[~live] == 0.5).all()


def test_cold_mask_uses_the_frozen_history_table_only(frame):
    table = pd.DataFrame({"n": [10.0, 0.0]}, index=pd.Index(["1", "2"], name="pitcher_id"))
    tables = SimpleNamespace(pitcher_all=table)
    probe = frame.iloc[:6].copy()
    probe["pitcher_id"] = [1, 2, 3, 1, 2, 3]
    cold = _tail_cold_mask(probe, tables)
    assert cold.tolist() == [False, True, True, False, True, True]   # unknown id and n == 0 are both cold


def test_prior_rate_lookup_ignores_unseen_pitchers_and_empty_priors(frame, tmp_path):
    pd.DataFrame(
        {"pitcher_id": [1, 2, 3], "preseason_n": [120, 0, 40], "pre_success_rate": [0.55, 0.9, 0.48]}
    ).to_csv(tmp_path / "current_state_preseason_2025.csv", index=False)
    probe = frame.iloc[:5].copy()
    probe["pitcher_id"] = [1, 2, 3, 4, 1]
    prior = _tail_prior_rate(probe, tmp_path)
    np.testing.assert_allclose(prior[[0, 2, 4]], [0.55, 0.48, 0.55])
    assert np.isnan(prior[1]) and np.isnan(prior[3])
