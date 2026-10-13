"""The audit must pass row-local predictors and catch every kind of cross-row dependence."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitch_control.tail_terms import apply_tail_terms
from pitch_control.v3_features import engineer_features
from validation.row_independence_audit import audit


def row_local(frame: pd.DataFrame) -> np.ndarray:
    feats = engineer_features(frame)
    z = (-0.3 + 0.8 * feats["three_ball"] - 0.5 * feats["two_strike"]
         + 0.9 * (feats["asof_pitcher_success_rate"].fillna(0.52) - 0.52) + 0.02 * feats["strikes_minus_balls"])
    return (1 / (1 + np.exp(-z))).to_numpy(float)


def with_tail_terms(frame: pd.DataFrame) -> np.ndarray:
    p = row_local(frame)
    spec = {"scale": 1.0, "terms": [{"type": "linear", "column": "asof_pitcher_success_rate", "center": 0.52, "gamma": 0.5}]}
    return apply_tail_terms(p, frame, np.ones(len(frame), dtype=bool), spec, np.zeros(len(frame), dtype=bool))


def test_row_local_predictors_pass_with_zero_difference(frame):
    for predict in (row_local, with_tail_terms):
        report = audit(predict, frame, n_probe=24, chunk_sizes=(1, 17, 400), tol=0.0)
        assert report["passed"], report
        assert report["max_abs_diff"] == 0.0


def test_batch_standardisation_is_caught(frame):
    def leaky(f):  # z-scores a column over the batch: the classic transductive feature
        col = f["asof_pitcher_success_rate"].fillna(0.52)
        return (1 / (1 + np.exp(-(col - col.mean()) / (col.std() + 1e-9)))).to_numpy(float)

    report = audit(leaky, frame, n_probe=16, chunk_sizes=(37,))
    assert not report["passed"]
    assert report["single_row"] > 1e-3 and report["unrelated_rows_scrambled"] > 1e-3


def test_order_dependence_is_caught_by_the_permutation_check(frame):
    def rolling(f):  # expanding mean over the order of appearance
        return f["balls_before"].expanding().mean().fillna(0).to_numpy(float) / 4.0

    report = audit(rolling, frame, n_probe=16, chunk_sizes=(400,))
    assert not report["passed"] and report["permutation"] > 1e-3


def test_group_frequency_features_are_caught_by_the_duplicate_check(frame):
    def frequency(f):  # value counts over the batch
        counts = f.groupby("pitcher_id")["row_id"].transform("size")
        return (counts / (counts.max() + 1)).to_numpy(float)

    report = audit(frequency, frame, n_probe=16, chunk_sizes=(400,))
    assert not report["passed"] and report["duplicates"] > 0


def test_hidden_state_between_calls_is_caught_by_the_repeat_check(frame):
    calls = {"n": 0}

    def stateful(f):
        calls["n"] += 1
        return row_local(f) * (1.0 - 1e-3 * (calls["n"] > 1))

    report = audit(stateful, frame, n_probe=8, chunk_sizes=(400,))
    assert not report["passed"] and report["repeat"] > 0


def test_shape_and_size_validation(frame):
    with pytest.raises(ValueError):
        audit(lambda f: np.zeros(len(f) + 1), frame)
    with pytest.raises(ValueError):
        audit(row_local, frame.iloc[:2])
