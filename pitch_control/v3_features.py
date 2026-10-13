
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import BASE_CATEGORICAL, ENGINEERED_CATEGORICAL, RATE_FEATURES, ROW_ID, TARGET


@dataclass
class FeatureSpec:
    mode: str
    features: list[str]
    cat_features: list[str]
    num_features: list[str]


def _to_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df:
        return pd.Series(default, index=df.index, dtype="float32")
    return pd.to_numeric(df[col], errors="coerce")


def _safe_div(a: pd.Series, b: pd.Series, default: float = 0.0) -> pd.Series:
    out = a.astype("float64") / b.replace(0, np.nan).astype("float64")
    return out.replace([np.inf, -np.inf], np.nan).fillna(default).astype("float32")


def bucket_numeric(series: pd.Series, edges: list[float], labels: list[str], missing: str = "missing") -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    out = pd.cut(s, bins=edges, labels=labels, include_lowest=True).astype("object")
    return out.fillna(missing).astype(str)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    balls = _to_num(out, "balls_before")
    strikes = _to_num(out, "strikes_before")
    outs = _to_num(out, "outs_before")
    inning = _to_num(out, "inning")
    score_diff = _to_num(out, "score_diff_pitcher_team")
    li = _to_num(out, "li")

    out["count_state"] = balls.fillna(-1).astype("int16").astype(str) + "-" + strikes.fillna(-1).astype("int16").astype(str)
    out["base_out_state"] = out.get("base_state", pd.Series("__NA__", index=out.index)).astype(str) + "_o" + outs.fillna(-1).astype("int16").astype(str)

    out["inning_phase"] = np.select(
        [inning <= 3, (inning >= 4) & (inning <= 6), (inning >= 7) & (inning <= 9), inning >= 10],
        ["early", "mid", "late", "extra"],
        default="unknown",
    ).astype(str)

    ph = out.get("pitcher_hand", pd.Series("__NA__", index=out.index)).astype(str)
    bh = out.get("batter_hand", pd.Series("__NA__", index=out.index)).astype(str)
    out["platoon_matchup"] = ph + "_vs_" + bh

    out["pressure_state"] = np.select(
        [li >= 2.0, li >= 1.2, li <= 0.7],
        ["very_high", "high", "low"],
        default="normal",
    ).astype(str)

    out["score_state"] = np.select(
        [score_diff >= 4, score_diff >= 1, score_diff == 0, score_diff <= -4],
        ["ahead_big", "ahead", "tie", "behind_big"],
        default="behind",
    ).astype(str)

    runners = _to_num(out, "num_runners_on")
    out["runners_state"] = runners.fillna(-1).astype("int16").astype(str)

    pitcher_n = _to_num(out, "asof_pitcher_n")
    batter_n = _to_num(out, "asof_batter_n")
    recent_cols = [c for c in out.columns if c.startswith("asof_pitcher_prev")]
    recent_missing = out[recent_cols].isna().all(axis=1) if recent_cols else pd.Series(False, index=out.index)
    out["cold_start_state"] = np.select(
        [pitcher_n.fillna(0).eq(0), batter_n.fillna(0).eq(0), (pitcher_n.fillna(0) <= 20) | (batter_n.fillna(0) <= 20)],
        ["pitcher_new", "batter_new", "low_sample"],
        default="known",
    ).astype(str)
    out["recent_state"] = np.where(recent_missing, "recent_missing", "recent_present").astype(str)
    out["game_type_x_count"] = out.get("game_type", pd.Series("__NA__", index=out.index)).astype(str) + "_" + out["count_state"].astype(str)

    # Numeric domain features
    out["balls_strikes_sum"] = (balls + strikes).astype("float32")
    out["strikes_minus_balls"] = (strikes - balls).astype("float32")
    out["three_ball"] = balls.eq(3).astype("int8")
    out["two_strike"] = strikes.eq(2).astype("int8")
    out["full_count"] = (balls.eq(3) & strikes.eq(2)).astype("int8")
    out["even_count"] = balls.eq(strikes).astype("int8")
    out["count_pressure"] = (out["three_ball"].astype("float32") + out["two_strike"].astype("float32")).astype("float32")

    out["abs_score_diff_pitcher_team"] = score_diff.abs().astype("float32")
    out["close_game"] = score_diff.abs().le(2).astype("int8")
    out["late_inning"] = inning.ge(7).astype("int8")
    out["extra_inning"] = inning.ge(10).astype("int8")
    out["late_close_game"] = (out["late_inning"].eq(1) & out["close_game"].eq(1)).astype("int8")
    out["li_x_abs_score_diff"] = (li.fillna(0) * score_diff.abs().fillna(0)).astype("float32")

    if {"runner_on_2b", "runner_on_3b"}.issubset(out.columns):
        out["runner_in_scoring_position"] = (
            _to_num(out, "runner_on_2b").fillna(0).gt(0) | _to_num(out, "runner_on_3b").fillna(0).gt(0)
        ).astype("int8")
    else:
        out["runner_in_scoring_position"] = 0
    out["bases_loaded"] = runners.eq(3).astype("int8")

    if {"home_win_expectancy", "away_win_expectancy"}.issubset(out.columns):
        home = _to_num(out, "home_win_expectancy")
        away = _to_num(out, "away_win_expectancy")
        out["win_expectancy_sum"] = (home + away).astype("float32")
        out["home_minus_away_win_expectancy"] = (home - away).astype("float32")
        out["win_expectancy_extremity"] = (home.sub(50).abs()).astype("float32")
    else:
        out["win_expectancy_sum"] = np.nan
        out["home_minus_away_win_expectancy"] = np.nan
        out["win_expectancy_extremity"] = np.nan

    # Sample-size, missing, and recent-vs-long features
    for col in ["asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"]:
        if col in out:
            vals = _to_num(out, col)
            out[f"log1p_{col}"] = np.log1p(vals.clip(lower=0)).astype("float32")
            out[f"{col}_is_zero"] = vals.fillna(0).eq(0).astype("int8")

    if "asof_pitcher_n" in out and "asof_batter_n" in out:
        out["log_pitcher_n_minus_log_batter_n"] = (
            np.log1p(pitcher_n.clip(lower=0).fillna(0)) - np.log1p(batter_n.clip(lower=0).fillna(0))
        ).astype("float32")

    long_success = _to_num(out, "asof_pitcher_success_rate")
    for w in [1, 3, 5]:
        c = f"asof_pitcher_prev{w}_game_success_rate"
        if c in out:
            out[f"pitcher_recent{w}_minus_long_success"] = (_to_num(out, c) - long_success).astype("float32")
        m = f"asof_pitcher_prev{w}_game_middle_rate"
        if m in out and "asof_pitcher_middle_rate" in out:
            out[f"pitcher_recent{w}_minus_long_middle"] = (_to_num(out, m) - _to_num(out, "asof_pitcher_middle_rate")).astype("float32")

    out["recent_history_missing"] = recent_missing.astype("int8")

    if {"asof_pitcher_success_rate", "asof_batter_success_rate"}.issubset(out.columns):
        ps = _to_num(out, "asof_pitcher_success_rate")
        bs = _to_num(out, "asof_batter_success_rate")
        out["pitcher_batter_success_mean"] = ((ps + bs) / 2.0).astype("float32")
        out["pitcher_minus_batter_success"] = (ps - bs).astype("float32")

    if {"asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"}.issubset(out.columns):
        f = _to_num(out, "asof_pitcher_fastball_rate")
        b = _to_num(out, "asof_pitcher_breaking_rate")
        o = _to_num(out, "asof_pitcher_offspeed_rate")
        out["pitchmix_entropy"] = (-(f*np.log(np.clip(f, 1e-6, 1)) + b*np.log(np.clip(b, 1e-6, 1)) + o*np.log(np.clip(o, 1e-6, 1)))).astype("float32")
        out["fastball_minus_breaking"] = (f - b).astype("float32")
        out["offspeed_minus_breaking"] = (o - b).astype("float32")

    return out


def normalize_frame_for_catboost(df: pd.DataFrame, features: list[str], cat_features: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in features:
        if col not in out:
            out[col] = np.nan
    for col in cat_features:
        out[col] = out[col].astype("object").where(out[col].notna(), "__NA__").astype(str)
    num_features = [c for c in features if c not in set(cat_features)]
    for col in num_features:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    return out[features]


def build_feature_spec(df: pd.DataFrame, mode: str) -> FeatureSpec:
    if mode not in {"official", "engineered"}:
        raise ValueError(f"unknown feature mode: {mode}")

    frame = engineer_features(df) if mode == "engineered" else df.copy()
    drop = {ROW_ID, TARGET}
    features = [c for c in frame.columns if c not in drop]

    cat_features = [c for c in BASE_CATEGORICAL if c in features]
    if mode == "engineered":
        cat_features += [c for c in ENGINEERED_CATEGORICAL if c in features]

    # Keep categorical features unique in feature order.
    cat_features = [c for c in features if c in set(cat_features)]
    num_features = [c for c in features if c not in set(cat_features)]
    return FeatureSpec(mode=mode, features=features, cat_features=cat_features, num_features=num_features)


def make_model_frame(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    frame = engineer_features(df) if spec.mode == "engineered" else df.copy()
    return normalize_frame_for_catboost(frame, spec.features, spec.cat_features)
