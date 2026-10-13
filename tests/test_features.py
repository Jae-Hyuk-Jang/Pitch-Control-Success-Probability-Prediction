"""Row-local feature construction (v3 engineering) on a synthetic frame."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pitch_control import config
from pitch_control.v3_features import (
    build_feature_spec,
    engineer_features,
    make_model_frame,
    normalize_frame_for_catboost,
)

ENGINEERED = [
    "count_state", "base_out_state", "inning_phase", "platoon_matchup", "pressure_state", "score_state",
    "runners_state", "cold_start_state", "recent_state", "game_type_x_count",
    "balls_strikes_sum", "strikes_minus_balls", "three_ball", "two_strike", "full_count", "even_count",
    "count_pressure", "close_game", "late_inning", "extra_inning", "late_close_game", "li_x_abs_score_diff",
    "runner_in_scoring_position", "bases_loaded", "win_expectancy_sum", "home_minus_away_win_expectancy",
    "win_expectancy_extremity", "log1p_asof_pitcher_n", "asof_pitcher_n_is_zero", "pitcher_recent1_minus_long_success",
    "pitcher_recent5_minus_long_middle", "recent_history_missing", "pitcher_minus_batter_success", "pitchmix_entropy",
    "fastball_minus_breaking",
]


def test_engineered_columns_and_input_is_not_mutated(frame):
    before = frame.copy(deep=True)
    out = engineer_features(frame)
    pd.testing.assert_frame_equal(frame, before)
    assert len(out) == len(frame)
    missing = [c for c in ENGINEERED if c not in out.columns]
    assert not missing, missing


def test_count_and_phase_encodings(frame):
    out = engineer_features(frame)
    assert (out["count_state"] == frame["balls_before"].astype(str) + "-" + frame["strikes_before"].astype(str)).all()
    phase = out["inning_phase"]
    assert set(phase[frame["inning"] <= 3]) == {"early"}
    assert set(phase[(frame["inning"] >= 4) & (frame["inning"] <= 6)]) == {"mid"}
    assert set(phase[frame["inning"] >= 10]) == {"extra"}
    assert set(out["pressure_state"]) <= {"very_high", "high", "low", "normal"}
    assert (out["full_count"] == ((frame["balls_before"] == 3) & (frame["strikes_before"] == 2)).astype(int)).all()


def test_cold_start_state_follows_sample_size(frame):
    out = engineer_features(frame)
    cold = frame["asof_pitcher_n"].fillna(0) == 0
    assert set(out.loc[cold, "cold_start_state"]) == {"pitcher_new"}
    assert "pitcher_new" not in set(out.loc[~cold, "cold_start_state"])
    assert np.isfinite(out["log1p_asof_pitcher_n"]).all()


def test_each_row_gets_the_same_features_alone_or_in_a_batch(frame):
    """The feature graph is row-local: no group statistics, no ordering, no vocabulary built from the batch."""
    full = engineer_features(frame)
    for i in (0, 7, 133, len(frame) - 1):
        alone = engineer_features(frame.iloc[[i]])
        pd.testing.assert_frame_equal(alone.reset_index(drop=True), full.iloc[[i]].reset_index(drop=True), check_dtype=False)


def test_shuffled_batch_gives_the_same_features(frame):
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(frame))
    a = engineer_features(frame).iloc[perm].reset_index(drop=True)
    b = engineer_features(frame.iloc[perm].reset_index(drop=True))
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_normalize_frame_for_catboost_contract(frame):
    out = engineer_features(frame)
    features = ["count_state", "asof_pitcher_success_rate", "does_not_exist", "pitcher_id"]
    cats = ["count_state", "pitcher_id"]
    model_frame = normalize_frame_for_catboost(out, features, cats)
    assert list(model_frame.columns) == features
    assert model_frame["does_not_exist"].isna().all()
    assert str(model_frame["asof_pitcher_success_rate"].dtype) == "float32"
    assert model_frame["count_state"].map(type).eq(str).all()
    assert "__NA__" not in set(model_frame["count_state"])


def test_feature_spec_separates_official_and_engineered_modes(frame):
    train_like = frame.assign(control_success=1)
    official = build_feature_spec(train_like, "official")
    engineered = build_feature_spec(train_like, "engineered")
    assert "control_success" not in official.features and "row_id" not in official.features
    assert set(official.features) < set(engineered.features)
    assert set(config.ENGINEERED_CATEGORICAL) <= set(engineered.cat_features)
    assert not set(config.ENGINEERED_CATEGORICAL) & set(official.cat_features)
    model_frame = make_model_frame(train_like, engineered)
    assert list(model_frame.columns) == engineered.features
    assert len(model_frame) == len(frame)
