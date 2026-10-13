from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PITCHER_ID = "pitcher_id"
GAME_TYPE = "game_type"
STATE_PRIOR = 100.0

STATE_NUMERIC = (
    "hist_n_all_log",
    "hist_n_r_log",
    "hist_n_f_log",
    "hist_r_share",
    "hist_f_share",
    "hist_active_seasons",
    "hist_years_since_seen",
    "hist_known",
    "prev_n_all_log",
    "prev_n_r_log",
    "prev_n_f_log",
    "prev_known",
    "state_long_all",
    "state_long_all_rel",
    "state_long_current_league",
    "state_long_current_league_rel",
    "state_prev_all",
    "state_prev_all_rel",
    "state_prev_current_league",
    "state_prev_current_league_rel",
    "state_recent_minus_long",
    "state_current_league_recent_minus_long",
    "state_r_minus_f",
    "preseason_n_log",
    "preseason_rate",
    "preseason_known",
    "current_season_n_log",
    "current_season_rate",
    "current_season_reliability",
    "current_season_known",
    "current_minus_preseason_rate",
    "current_minus_long_trait",
)


@dataclass(frozen=True)
class HistoryTables:
    season: int
    pitcher_all: pd.DataFrame
    pitcher_previous: pd.DataFrame
    pitcher_all_league: pd.DataFrame
    pitcher_previous_league: pd.DataFrame
    league_all: pd.DataFrame
    latest_snapshot: pd.DataFrame


def key_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column].astype("object")
    return values.where(values.notna(), "__MISSING__").astype(str)


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def add_row_local_features(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    balls = numeric_series(frame, "balls_before").fillna(-1).astype(int).astype(str)
    strikes = numeric_series(frame, "strikes_before").fillna(-1).astype(int).astype(str)
    frame["count_state"] = balls + "_" + strikes
    frame["hand_matchup"] = key_series(frame, "pitcher_hand") + "_" + key_series(
        frame, "batter_hand"
    )
    inning = numeric_series(frame, "inning")
    frame["inning_bucket"] = pd.cut(
        inning,
        bins=[-np.inf, 3.0, 6.0, 8.0, np.inf],
        labels=["EARLY", "MIDDLE", "LATE", "EXTRA"],
    ).astype("object")
    frame["inning_bucket"] = frame["inning_bucket"].where(
        frame["inning_bucket"].notna(), "__MISSING__"
    ).astype(str)
    frame["is_risp"] = (
        numeric_series(frame, "runner_on_2b").fillna(0).gt(0)
        | numeric_series(frame, "runner_on_3b").fillna(0).gt(0)
    ).astype("float32")
    frame["is_late_close"] = (
        inning.ge(7)
        & numeric_series(frame, "score_diff_pitcher_team").abs().le(2)
    ).astype("float32")
    frame["score_diff_abs"] = numeric_series(
        frame, "score_diff_pitcher_team"
    ).abs().astype("float32")
    frame["pitcher_batter_rate_product"] = (
        numeric_series(frame, "asof_pitcher_success_rate")
        * numeric_series(frame, "asof_batter_success_rate")
    ).astype("float32")
    return frame


def _read_indexed(path: Path, keys: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        return frame.set_index(keys)
    for key in keys:
        frame[key] = frame[key].astype(str)
    return frame.set_index(keys)


def load_history_tables(model_dir: Path) -> HistoryTables:
    model_dir = Path(model_dir)
    return HistoryTables(
        season=2025,
        pitcher_all=_read_indexed(model_dir / "asym_m1_state_pitcher_all.csv", [PITCHER_ID]),
        pitcher_previous=_read_indexed(
            model_dir / "asym_m1_state_pitcher_previous.csv", [PITCHER_ID]
        ),
        pitcher_all_league=_read_indexed(
            model_dir / "asym_m1_state_pitcher_all_league.csv", [PITCHER_ID, GAME_TYPE]
        ),
        pitcher_previous_league=_read_indexed(
            model_dir / "asym_m1_state_pitcher_previous_league.csv",
            [PITCHER_ID, GAME_TYPE],
        ),
        league_all=_read_indexed(model_dir / "asym_m1_state_league_all.csv", [GAME_TYPE]),
        latest_snapshot=_read_indexed(
            model_dir / "asym_m1_state_latest_snapshot.csv", [PITCHER_ID]
        ),
    )


def _map_index(table: pd.DataFrame, column: str, pitcher: pd.Series) -> np.ndarray:
    if table.empty:
        return np.zeros(len(pitcher), dtype=float)
    return pd.to_numeric(pitcher.map(table[column]), errors="coerce").fillna(0.0).to_numpy(float)


def _map_pair(
    table: pd.DataFrame, column: str, pitcher: pd.Series, league: pd.Series
) -> np.ndarray:
    if table.empty:
        return np.zeros(len(pitcher), dtype=float)
    keys = pd.MultiIndex.from_arrays([pitcher.to_numpy(str), league.to_numpy(str)])
    return pd.to_numeric(table[column].reindex(keys), errors="coerce").fillna(0.0).to_numpy(float)


def _shrunk(residual_sum: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return residual_sum / (n + STATE_PRIOR), n / (n + STATE_PRIOR)


def attach_state(frame: pd.DataFrame, tables: HistoryTables) -> pd.DataFrame:
    result = frame.copy()
    pitcher = key_series(result, PITCHER_ID)
    league = key_series(result, GAME_TYPE)
    rows = len(result)

    all_n = _map_index(tables.pitcher_all, "n", pitcher)
    all_sum = _map_index(tables.pitcher_all, "residual_sum", pitcher)
    active_seasons = _map_index(tables.pitcher_all, "active_seasons", pitcher)
    last_season = _map_index(tables.pitcher_all, "last_season", pitcher)
    prev_n = _map_index(tables.pitcher_previous, "n", pitcher)
    prev_sum = _map_index(tables.pitcher_previous, "residual_sum", pitcher)
    current_n = _map_pair(tables.pitcher_all_league, "n", pitcher, league)
    current_sum = _map_pair(tables.pitcher_all_league, "residual_sum", pitcher, league)
    prev_current_n = _map_pair(tables.pitcher_previous_league, "n", pitcher, league)
    prev_current_sum = _map_pair(
        tables.pitcher_previous_league, "residual_sum", pitcher, league
    )
    r_key = pd.Series(np.repeat("R", rows), index=result.index, dtype=str)
    f_key = pd.Series(np.repeat("F", rows), index=result.index, dtype=str)
    r_n = _map_pair(tables.pitcher_all_league, "n", pitcher, r_key)
    f_n = _map_pair(tables.pitcher_all_league, "n", pitcher, f_key)
    r_sum = _map_pair(tables.pitcher_all_league, "residual_sum", pitcher, r_key)
    f_sum = _map_pair(tables.pitcher_all_league, "residual_sum", pitcher, f_key)
    prev_r_n = _map_pair(tables.pitcher_previous_league, "n", pitcher, r_key)
    prev_f_n = _map_pair(tables.pitcher_previous_league, "n", pitcher, f_key)

    prior_state = np.full(rows, "NEW", dtype=object)
    historical = all_n > 0
    prior_state[historical & (prev_n <= 0)] = "DORMANT"
    prior_state[(prev_r_n > 0) & (prev_f_n <= 0)] = "R_ONLY"
    prior_state[(prev_f_n > 0) & (prev_r_n <= 0)] = "F_ONLY"
    prior_state[(prev_r_n > 0) & (prev_f_n > 0)] = "BOTH"
    result["prior_league_state"] = prior_state.astype(str)
    result["league_transition"] = result["prior_league_state"].astype(str) + "_TO_" + league

    denominator = np.maximum(r_n + f_n, 1.0)
    result["hist_n_all_log"] = np.log1p(all_n)
    result["hist_n_r_log"] = np.log1p(r_n)
    result["hist_n_f_log"] = np.log1p(f_n)
    result["hist_r_share"] = r_n / denominator
    result["hist_f_share"] = f_n / denominator
    result["hist_active_seasons"] = active_seasons
    result["hist_years_since_seen"] = np.where(
        historical, np.maximum(0.0, tables.season - last_season), 0.0
    )
    result["hist_known"] = historical.astype(float)
    result["prev_n_all_log"] = np.log1p(prev_n)
    result["prev_n_r_log"] = np.log1p(prev_r_n)
    result["prev_n_f_log"] = np.log1p(prev_f_n)
    result["prev_known"] = (prev_n > 0).astype(float)

    long_all, long_all_rel = _shrunk(all_sum, all_n)
    long_current, long_current_rel = _shrunk(current_sum, current_n)
    prev_all, prev_all_rel = _shrunk(prev_sum, prev_n)
    prev_current, prev_current_rel = _shrunk(prev_current_sum, prev_current_n)
    r_state, _ = _shrunk(r_sum, r_n)
    f_state, _ = _shrunk(f_sum, f_n)
    result["state_long_all"] = long_all
    result["state_long_all_rel"] = long_all_rel
    result["state_long_current_league"] = long_current
    result["state_long_current_league_rel"] = long_current_rel
    result["state_prev_all"] = prev_all
    result["state_prev_all_rel"] = prev_all_rel
    result["state_prev_current_league"] = prev_current
    result["state_prev_current_league_rel"] = prev_current_rel
    result["state_recent_minus_long"] = np.where(prev_n > 0, prev_all - long_all, 0.0)
    result["state_current_league_recent_minus_long"] = np.where(
        prev_current_n > 0, prev_current - long_current, 0.0
    )
    result["state_r_minus_f"] = np.where(
        (r_n > 0) & (f_n > 0), r_state - f_state, 0.0
    )

    snapshot_n = _map_index(tables.latest_snapshot, "snapshot_n", pitcher)
    snapshot_success = _map_index(tables.latest_snapshot, "snapshot_success", pitcher)
    snapshot_rate = _map_index(tables.latest_snapshot, "snapshot_rate", pitcher)
    snapshot_known = snapshot_n > 0
    asof_n = numeric_series(result, "asof_pitcher_n").fillna(0.0).to_numpy(float)
    asof_rate = numeric_series(result, "asof_pitcher_success_rate").fillna(0.0).to_numpy(float)
    season_n = asof_n - snapshot_n
    season_success = asof_n * asof_rate - snapshot_success
    plausible = (
        snapshot_known
        & np.isfinite(season_n)
        & np.isfinite(season_success)
        & (season_n >= 1.0)
        & (season_success >= -0.5)
        & (season_success <= season_n + 0.5)
    )
    season_rate = np.zeros(rows, dtype=float)
    season_rate[plausible] = np.clip(
        season_success[plausible] / season_n[plausible], 0.0, 1.0
    )
    season_n_safe = np.where(plausible, np.maximum(season_n, 0.0), 0.0)
    result["preseason_n_log"] = np.log1p(snapshot_n)
    result["preseason_rate"] = np.where(snapshot_known, snapshot_rate, 0.0)
    result["preseason_known"] = snapshot_known.astype(float)
    result["current_season_n_log"] = np.log1p(season_n_safe)
    result["current_season_rate"] = season_rate
    result["current_season_reliability"] = season_n_safe / (season_n_safe + STATE_PRIOR)
    result["current_season_known"] = plausible.astype(float)
    result["current_minus_preseason_rate"] = np.where(
        plausible, season_rate - snapshot_rate, 0.0
    )
    league_mean = (
        np.zeros(rows, dtype=float)
        if tables.league_all.empty
        else pd.to_numeric(league.map(tables.league_all["rate"]), errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
    )
    historical_expected = np.clip(league_mean + long_current, 0.0, 1.0)
    result["current_minus_long_trait"] = np.where(
        plausible, season_rate - historical_expected, 0.0
    )

    for column in STATE_NUMERIC:
        result[column] = pd.to_numeric(result[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0).astype("float32")
    return result


def build_deployment_features(raw: pd.DataFrame, tables: HistoryTables) -> pd.DataFrame:
    return attach_state(add_row_local_features(raw), tables)
