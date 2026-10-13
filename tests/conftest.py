"""Shared fixtures: a synthetic frame with the official input schema.

The competition data is not part of this repository. Everything below is random numbers that only mimic the *schema*
(column names, value ranges, cold-start missingness), so the tests exercise code paths, not model quality.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "training" / "reference"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _base_state(r1, r2, r3):
    return np.char.add(np.char.add(np.where(r1 == 1, "1", "_"), np.where(r2 == 1, "2", "_")), np.where(r3 == 1, "3", "_"))


def make_frame(n: int = 400, seed: int = 0, season: int = 2025, with_label: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    balls, strikes, outs = rng.integers(0, 4, n), rng.integers(0, 3, n), rng.integers(0, 3, n)
    r1, r2, r3 = (rng.integers(0, 2, n) for _ in range(3))
    inning = rng.integers(1, 12, n)
    home_we = np.clip(rng.normal(50, 20, n), 1, 99)
    pitcher_n = rng.integers(0, 3000, n).astype(float)
    pitcher_n[rng.random(n) < 0.10] = 0.0                      # cold start
    batter_n = rng.integers(0, 2500, n).astype(float)
    batter_n[rng.random(n) < 0.05] = 0.0

    def rate(center, spread, count):
        values = np.clip(rng.normal(center, spread, n), 0.0, 1.0)
        return np.where(count > 0, values, np.nan)

    mix = rng.dirichlet([4, 3, 2], size=n)
    frame = pd.DataFrame(
        {
            "row_id": [f"r{seed}_{i}" for i in range(n)],
            "season": season,
            "game_month": rng.integers(4, 11, n),
            "game_dayofweek": rng.integers(0, 7, n),
            "inning": inning,
            "top_bottom": rng.choice(["T", "B"], n),
            "game_type": rng.choice(["R", "F"], n, p=[0.88, 0.12]),
            "balls_before": balls,
            "strikes_before": strikes,
            "outs_before": outs,
            "run_top_before": rng.integers(0, 8, n),
            "run_bot_before": rng.integers(0, 8, n),
            "score_diff_pitcher_team": rng.integers(-6, 7, n),
            "runner_on_1b": r1,
            "runner_on_2b": r2,
            "runner_on_3b": r3,
            "num_runners_on": r1 + r2 + r3,
            "base_state": _base_state(r1, r2, r3),
            "home_win_expectancy": home_we,
            "away_win_expectancy": 100.0 - home_we,
            "li": np.clip(rng.lognormal(0.0, 0.5, n), 0.1, 6.0),
            "pitcher_id": rng.integers(1, 41, n),
            "batter_id": rng.integers(1, 121, n),
            "pitcher_hand": rng.choice(["R", "L"], n, p=[0.7, 0.3]),
            "batter_hand": rng.choice(["R", "L"], n, p=[0.6, 0.4]),
            "pitcher_team_id": rng.integers(1, 11, n),
            "batter_team_id": rng.integers(1, 11, n),
            "asof_pitcher_n": pitcher_n,
            "asof_pitcher_success_rate": rate(0.52, 0.06, pitcher_n),
            "asof_pitcher_reverse_rate": rate(0.22, 0.05, pitcher_n),
            "asof_pitcher_middle_rate": rate(0.16, 0.04, pitcher_n),
            "asof_pitcher_ball_rate": rate(0.30, 0.05, pitcher_n),
            "asof_pitcher_strike_rate": rate(0.70, 0.05, pitcher_n),
            "asof_pitcher_prev1_game_success_rate": rate(0.52, 0.12, pitcher_n),
            "asof_pitcher_prev3_game_success_rate": rate(0.52, 0.09, pitcher_n),
            "asof_pitcher_prev5_game_success_rate": rate(0.52, 0.07, pitcher_n),
            "asof_pitcher_prev1_game_middle_rate": rate(0.16, 0.08, pitcher_n),
            "asof_pitcher_prev3_game_middle_rate": rate(0.16, 0.06, pitcher_n),
            "asof_pitcher_prev5_game_middle_rate": rate(0.16, 0.05, pitcher_n),
            "asof_batter_n": batter_n,
            "asof_batter_success_rate": rate(0.52, 0.05, batter_n),
            "asof_batter_middle_rate": rate(0.16, 0.04, batter_n),
            "asof_pitcher_pitchmix_n": pitcher_n,
            "asof_pitcher_fastball_rate": np.where(pitcher_n > 0, mix[:, 0], np.nan),
            "asof_pitcher_breaking_rate": np.where(pitcher_n > 0, mix[:, 1], np.nan),
            "asof_pitcher_offspeed_rate": np.where(pitcher_n > 0, mix[:, 2], np.nan),
        }
    )
    frame["run_total_before"] = frame["run_top_before"] + frame["run_bot_before"]
    frame["score_diff_home"] = -frame["score_diff_pitcher_team"]
    if with_label:
        frame["control_success"] = (rng.random(n) < 0.5).astype(int)
    return frame


@pytest.fixture(scope="session")
def frame() -> pd.DataFrame:
    return make_frame(400, seed=1)


@pytest.fixture()
def small_frame() -> pd.DataFrame:
    return make_frame(60, seed=2)
