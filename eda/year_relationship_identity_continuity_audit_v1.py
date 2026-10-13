from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.stats import rankdata


KST = ZoneInfo("Asia/Seoul")
YEARS = tuple(range(2019, 2025))
ENTITY_SPECS = {
    "pitcher": {
        "id": "pitcher_id",
        "team": "pitcher_team_id",
        "asof_n": "asof_pitcher_n",
        "asof_rate": "asof_pitcher_success_rate",
    },
    "batter": {
        "id": "batter_id",
        "team": "batter_team_id",
        "asof_n": "asof_batter_n",
        "asof_rate": "asof_batter_success_rate",
    },
}
WINDOWS = {
    "all_prior": None,
    "last_1_season": 1,
    "last_2_seasons": 2,
    "last_3_seasons": 3,
}
USECOLS = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "num_runners_on",
    "base_state",
    "home_win_expectancy",
    "li",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "control_success",
]
DTYPES = {
    "season": "int16",
    "game_month": "int8",
    "game_dayofweek": "int8",
    "inning": "int8",
    "top_bottom": "category",
    "game_type": "category",
    "balls_before": "int8",
    "strikes_before": "int8",
    "outs_before": "int8",
    "score_diff_home": "int16",
    "score_diff_pitcher_team": "int16",
    "num_runners_on": "int8",
    "base_state": "category",
    "home_win_expectancy": "float32",
    "li": "float32",
    "pitcher_id": "int32",
    "batter_id": "int32",
    "pitcher_hand": "int8",
    "batter_hand": "int8",
    "pitcher_team_id": "int16",
    "batter_team_id": "int16",
    "asof_pitcher_n": "int32",
    "asof_pitcher_success_rate": "float32",
    "asof_batter_n": "int32",
    "asof_batter_success_rate": "float32",
    "control_success": "int8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--data-description", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def kst_now() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST (UTC+09:00)")


def log(message: str) -> None:
    print(f"[{kst_now()}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8")


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.square(p.astype(float) - y.astype(float))))


def weighted_corr(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(weight) & (weight > 0)
    if mask.sum() < 3:
        return math.nan
    x = x[mask].astype(float)
    y = y[mask].astype(float)
    weight = weight[mask].astype(float)
    weight = weight / weight.sum()
    x_centered = x - np.sum(weight * x)
    y_centered = y - np.sum(weight * y)
    denominator = math.sqrt(
        float(np.sum(weight * np.square(x_centered)))
        * float(np.sum(weight * np.square(y_centered)))
    )
    if denominator == 0:
        return math.nan
    return float(np.sum(weight * x_centered * y_centered) / denominator)


def weighted_slope(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(weight) & (weight > 0)
    if mask.sum() < 3:
        return math.nan
    x = x[mask].astype(float)
    y = y[mask].astype(float)
    weight = weight[mask].astype(float)
    weight = weight / weight.sum()
    x_centered = x - np.sum(weight * x)
    denominator = float(np.sum(weight * np.square(x_centered)))
    if denominator == 0:
        return math.nan
    return float(np.sum(weight * x_centered * (y - np.sum(weight * y))) / denominator)


def fit_beta_prior(counts: pd.DataFrame) -> dict[str, float | bool | str]:
    n = counts["history_n"].to_numpy(float)
    k = counts["history_successes"].to_numpy(float)
    valid = (n > 0) & np.isfinite(n) & np.isfinite(k)
    n = n[valid]
    k = k[valid]
    if len(n) == 0:
        raise ValueError("Cannot fit a beta prior without history rows")
    mean = float(k.sum() / n.sum())
    mean = float(np.clip(mean, 1e-5, 1 - 1e-5))
    initial_strength = 50.0

    def objective(theta: np.ndarray) -> float:
        alpha, beta = np.exp(theta)
        value = np.sum(betaln(k + alpha, n - k + beta) - betaln(alpha, beta))
        return -float(value)

    result = minimize(
        objective,
        x0=np.log([mean * initial_strength, (1 - mean) * initial_strength]),
        method="L-BFGS-B",
        bounds=[(math.log(1e-3), math.log(1e7))] * 2,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    alpha, beta = np.exp(result.x)
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "prior_mean": float(alpha / (alpha + beta)),
        "prior_strength": float(alpha + beta),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }


def add_context_columns(frame: pd.DataFrame) -> None:
    frame["count_state"] = (
        frame["balls_before"].astype(str) + "-" + frame["strikes_before"].astype(str)
    )
    frame["inning_bucket"] = pd.cut(
        frame["inning"],
        bins=[0, 3, 6, 9, np.inf],
        labels=["01_03", "04_06", "07_09", "10_plus"],
        right=True,
    ).astype("category")
    frame["score_bucket"] = frame["score_diff_pitcher_team"].clip(-4, 4).astype(str)
    frame["li_bucket"] = pd.cut(
        frame["li"],
        bins=[-np.inf, 0.5, 1.0, 2.0, 4.0, np.inf],
        labels=["le_0.5", "0.5_1", "1_2", "2_4", "gt_4"],
        right=True,
    ).astype("category")
    frame["joint_context"] = (
        frame["game_type"].astype(str)
        + "|"
        + frame["pitcher_hand"].astype(str)
        + "|"
        + frame["batter_hand"].astype(str)
        + "|"
        + frame["count_state"].astype(str)
        + "|"
        + frame["inning_bucket"].astype(str)
        + "|"
        + frame["base_state"].astype(str)
    )


def year_context_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    dimensions = [
        "__overall__",
        "game_type",
        "game_month",
        "game_dayofweek",
        "pitcher_hand",
        "batter_hand",
        "pitcher_team_id",
        "batter_team_id",
        "count_state",
        "inning_bucket",
        "num_runners_on",
        "base_state",
        "score_bucket",
        "li_bucket",
        "joint_context",
    ]
    rows: list[dict[str, object]] = []
    for dimension in dimensions:
        if dimension == "__overall__":
            grouped = frame.groupby("season", observed=True)
            summary = grouped.agg(
                rows=("control_success", "size"),
                target_successes=("control_success", "sum"),
                target_mean=("control_success", "mean"),
                pitcher_ids=("pitcher_id", "nunique"),
                batter_ids=("batter_id", "nunique"),
                pitcher_asof_n_median=("asof_pitcher_n", "median"),
                batter_asof_n_median=("asof_batter_n", "median"),
            ).reset_index()
            for record in summary.to_dict("records"):
                record.update({"dimension": "overall", "group_value": "ALL"})
                rows.append(record)
            continue
        summary = (
            frame.groupby(["season", dimension], observed=True, dropna=False)
            .agg(
                rows=("control_success", "size"),
                target_successes=("control_success", "sum"),
                target_mean=("control_success", "mean"),
                pitcher_ids=("pitcher_id", "nunique"),
                batter_ids=("batter_id", "nunique"),
            )
            .reset_index()
            .rename(columns={dimension: "group_value"})
        )
        summary["dimension"] = dimension
        summary["pitcher_asof_n_median"] = np.nan
        summary["batter_asof_n_median"] = np.nan
        rows.extend(summary.to_dict("records"))
    result = pd.DataFrame(rows)
    result["row_share_within_year_dimension"] = result["rows"] / result.groupby(
        ["season", "dimension"], observed=True
    )["rows"].transform("sum")
    return result[
        [
            "season",
            "dimension",
            "group_value",
            "rows",
            "row_share_within_year_dimension",
            "target_successes",
            "target_mean",
            "pitcher_ids",
            "batter_ids",
            "pitcher_asof_n_median",
            "batter_asof_n_median",
        ]
    ].sort_values(["season", "dimension", "group_value"], kind="stable")


def year_pair_relationship_matrix(context: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    overall = (
        context.loc[context["dimension"] == "overall"]
        .set_index("season")["target_mean"]
        .astype(float)
    )
    dimensions = sorted(
        value for value in context["dimension"].unique() if value != "overall"
    )
    for from_index, from_year in enumerate(YEARS[:-1]):
        for to_year in YEARS[from_index + 1 :]:
            for dimension in dimensions:
                selected = context.loc[context["dimension"] == dimension].copy()
                selected["group_value"] = selected["group_value"].astype(str)
                left = selected.loc[selected["season"] == from_year].set_index(
                    "group_value"
                )
                right = selected.loc[selected["season"] == to_year].set_index(
                    "group_value"
                )
                categories = left.index.union(right.index)
                p = left["row_share_within_year_dimension"].reindex(
                    categories, fill_value=0.0
                ).to_numpy(float)
                q = right["row_share_within_year_dimension"].reindex(
                    categories, fill_value=0.0
                ).to_numpy(float)
                midpoint = 0.5 * (p + q)
                p_positive = p > 0
                q_positive = q > 0
                js_divergence = 0.5 * float(
                    np.sum(p[p_positive] * np.log(p[p_positive] / midpoint[p_positive]))
                    + np.sum(q[q_positive] * np.log(q[q_positive] / midpoint[q_positive]))
                )
                shared = left.index.intersection(right.index)
                left_shared = left.loc[shared]
                right_shared = right.loc[shared]
                average_shared_weight = 0.5 * (
                    left_shared["row_share_within_year_dimension"].to_numpy(float)
                    + right_shared["row_share_within_year_dimension"].to_numpy(float)
                )
                conditional_difference = (
                    right_shared["target_mean"].to_numpy(float)
                    - left_shared["target_mean"].to_numpy(float)
                )
                rows.append(
                    {
                        "from_year": from_year,
                        "to_year": to_year,
                        "dimension": dimension,
                        "overall_target_delta": float(overall[to_year] - overall[from_year]),
                        "absolute_overall_target_delta": abs(
                            float(overall[to_year] - overall[from_year])
                        ),
                        "total_variation_distance": 0.5 * float(np.abs(p - q).sum()),
                        "jensen_shannon_divergence_nats": js_divergence,
                        "from_shared_row_coverage": float(
                            left_shared["row_share_within_year_dimension"].sum()
                        ),
                        "to_shared_row_coverage": float(
                            right_shared["row_share_within_year_dimension"].sum()
                        ),
                        "shared_weighted_absolute_conditional_target_delta": float(
                            np.sum(average_shared_weight * np.abs(conditional_difference))
                            / average_shared_weight.sum()
                        ),
                        "shared_weighted_signed_conditional_target_delta": float(
                            np.sum(average_shared_weight * conditional_difference)
                            / average_shared_weight.sum()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_entity_season(frame: pd.DataFrame, entity_type: str) -> pd.DataFrame:
    spec = ENTITY_SPECS[entity_type]
    return (
        frame.groupby(["season", spec["id"]], observed=True, sort=False)
        .agg(
            n=("control_success", "size"),
            successes=("control_success", "sum"),
            team_first=(spec["team"], "first"),
            team_last=(spec["team"], "last"),
            teams=(spec["team"], "nunique"),
            game_types=("game_type", "nunique"),
        )
        .reset_index()
    )


def entity_year_pair_persistence(
    frame: pd.DataFrame,
    entity_seasons: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    game_rates = (
        frame.groupby(["season", "game_type"], observed=True)["control_success"]
        .mean()
        .rename("game_type_rate")
    )
    game_index = pd.MultiIndex.from_arrays(
        [frame["season"], frame["game_type"]], names=["season", "game_type"]
    )
    expected = game_rates.reindex(game_index).to_numpy(float)
    for entity_type, season_table in entity_seasons.items():
        entity_col = ENTITY_SPECS[entity_type]["id"]
        raw = season_table.copy()
        raw["raw_rate"] = raw["successes"] / raw["n"]
        residual_frame = pd.DataFrame(
            {
                "season": frame["season"].to_numpy(),
                entity_col: frame[entity_col].to_numpy(),
                "residual": frame["control_success"].to_numpy(float) - expected,
            }
        )
        residual = (
            residual_frame.groupby(["season", entity_col], observed=True)
            .agg(n=("residual", "size"), residual_rate=("residual", "mean"))
            .reset_index()
        )
        for from_index, from_year in enumerate(YEARS[:-1]):
            for to_year in YEARS[from_index + 1 :]:
                left_raw = raw.loc[raw["season"] == from_year].set_index(entity_col)
                right_raw = raw.loc[raw["season"] == to_year].set_index(entity_col)
                joined_raw = left_raw[["n", "raw_rate"]].join(
                    right_raw[["n", "raw_rate"]],
                    how="inner",
                    lsuffix="_from",
                    rsuffix="_to",
                )
                left_residual = residual.loc[
                    residual["season"] == from_year
                ].set_index(entity_col)
                right_residual = residual.loc[
                    residual["season"] == to_year
                ].set_index(entity_col)
                joined_residual = left_residual[["residual_rate"]].join(
                    right_residual[["residual_rate"]],
                    how="inner",
                    lsuffix="_from",
                    rsuffix="_to",
                ).loc[joined_raw.index]
                weight = joined_raw["n_to"].to_numpy(float)
                raw_from = joined_raw["raw_rate_from"].to_numpy(float)
                raw_to = joined_raw["raw_rate_to"].to_numpy(float)
                residual_from = joined_residual["residual_rate_from"].to_numpy(float)
                residual_to = joined_residual["residual_rate_to"].to_numpy(float)
                rows.append(
                    {
                        "entity_type": entity_type,
                        "from_year": from_year,
                        "to_year": to_year,
                        "year_gap": to_year - from_year,
                        "from_entities": len(left_raw),
                        "to_entities": len(right_raw),
                        "shared_entities": len(joined_raw),
                        "to_entity_coverage": len(joined_raw) / len(right_raw),
                        "to_row_coverage": float(
                            joined_raw["n_to"].sum() / right_raw["n"].sum()
                        ),
                        "raw_rate_pearson_to_row_weighted": weighted_corr(
                            raw_from, raw_to, weight
                        ),
                        "raw_rate_spearman_to_row_weighted": weighted_corr(
                            rankdata(raw_from), rankdata(raw_to), weight
                        ),
                        "raw_rate_slope_to_row_weighted": weighted_slope(
                            raw_from, raw_to, weight
                        ),
                        "game_type_adjusted_residual_pearson_to_row_weighted": weighted_corr(
                            residual_from, residual_to, weight
                        ),
                        "game_type_adjusted_residual_spearman_to_row_weighted": weighted_corr(
                            rankdata(residual_from), rankdata(residual_to), weight
                        ),
                        "game_type_adjusted_residual_slope_to_row_weighted": weighted_slope(
                            residual_from, residual_to, weight
                        ),
                    }
                )
    return pd.DataFrame(rows)


def entity_continuity(
    frame: pd.DataFrame,
    entity_seasons: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    continuity_rows: list[dict[str, object]] = []
    span_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    for entity_type, season_table in entity_seasons.items():
        entity_col = ENTITY_SPECS[entity_type]["id"]
        sets = {
            year: set(season_table.loc[season_table["season"] == year, entity_col])
            for year in YEARS
        }
        counts = season_table.groupby(entity_col, observed=True)["season"].nunique()
        span = counts.value_counts().sort_index()
        for season_count, entities in span.items():
            span_rows.append(
                {
                    "entity_type": entity_type,
                    "seasons_observed": int(season_count),
                    "entities": int(entities),
                }
            )

        for to_year in YEARS[1:]:
            target_set = sets[to_year]
            current = frame.loc[frame["season"] == to_year, [entity_col]]
            prior_union = set().union(*(sets[year] for year in YEARS if year < to_year))
            previous = sets[to_year - 1]
            for source_scope, source_set, from_year in (
                ("single_previous_year", previous, str(to_year - 1)),
                ("all_prior_years", prior_union, f"2019-{to_year - 1}"),
            ):
                intersection = source_set & target_set
                union = source_set | target_set
                continuity_rows.append(
                    {
                        "entity_type": entity_type,
                        "source_scope": source_scope,
                        "from_year": from_year,
                        "to_year": to_year,
                        "source_entities": len(source_set),
                        "target_entities": len(target_set),
                        "intersection_entities": len(intersection),
                        "jaccard": len(intersection) / len(union),
                        "target_entity_coverage": len(intersection) / len(target_set),
                        "target_row_coverage": float(current[entity_col].isin(source_set).mean()),
                    }
                )

            status = np.select(
                [
                    current[entity_col].isin(previous),
                    current[entity_col].isin(prior_union),
                ],
                ["continuing", "returning"],
                default="new",
            )
            status_counts = pd.Series(status).value_counts()
            for value in ("continuing", "returning", "new"):
                rows = int(status_counts.get(value, 0))
                status_rows.append(
                    {
                        "entity_type": entity_type,
                        "eval_year": to_year,
                        "status": value,
                        "rows": rows,
                        "row_share": rows / len(current),
                    }
                )
    return (
        pd.DataFrame(continuity_rows),
        pd.DataFrame(span_rows),
        pd.DataFrame(status_rows),
    )


def asof_cross_season_continuity(
    frame: pd.DataFrame,
    entity_seasons: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for entity_type, season_table in entity_seasons.items():
        spec = ENTITY_SPECS[entity_type]
        entity_col = spec["id"]
        for eval_year in YEARS[1:]:
            prior = season_table.loc[season_table["season"] < eval_year]
            prior_agg = prior.groupby(entity_col, observed=True).agg(
                expected_n=("n", "sum"),
                expected_successes=("successes", "sum"),
            )
            prior_agg["expected_rate"] = (
                prior_agg["expected_successes"] / prior_agg["expected_n"]
            )
            first_rows = (
                frame.loc[
                    frame["season"] == eval_year,
                    [entity_col, spec["asof_n"], spec["asof_rate"]],
                ]
                .drop_duplicates(entity_col, keep="first")
                .set_index(entity_col)
                .join(prior_agg, how="left")
            )
            first_rows["expected_n"] = first_rows["expected_n"].fillna(0)
            prior_known = first_rows["expected_n"] > 0
            n_difference = (
                first_rows[spec["asof_n"]].astype(float) - first_rows["expected_n"]
            )
            rate_difference = (
                first_rows[spec["asof_rate"]].astype(float)
                - first_rows["expected_rate"].astype(float)
            ).abs()
            finite_rate = prior_known & np.isfinite(rate_difference)
            rows.append(
                {
                    "entity_type": entity_type,
                    "eval_year": eval_year,
                    "eval_entities": len(first_rows),
                    "prior_known_entities": int(prior_known.sum()),
                    "prior_known_entity_share": float(prior_known.mean()),
                    "asof_n_exact_all_fraction": float((n_difference == 0).mean()),
                    "asof_n_exact_known_fraction": float(
                        (n_difference[prior_known] == 0).mean()
                    ),
                    "asof_n_abs_diff_max": float(n_difference.abs().max()),
                    "asof_rate_comparable_entities": int(finite_rate.sum()),
                    "asof_rate_abs_diff_median": float(
                        rate_difference[finite_rate].median()
                    ),
                    "asof_rate_abs_diff_p99": float(
                        rate_difference[finite_rate].quantile(0.99)
                    ),
                    "asof_rate_abs_diff_max": float(rate_difference[finite_rate].max()),
                }
            )
    return pd.DataFrame(rows)


def history_aggregate(
    season_table: pd.DataFrame,
    entity_col: str,
    eval_year: int,
    window: int | None,
) -> pd.DataFrame:
    lower = min(YEARS) if window is None else max(min(YEARS), eval_year - window)
    selected = season_table.loc[
        (season_table["season"] >= lower) & (season_table["season"] < eval_year)
    ]
    return selected.groupby(entity_col, observed=True).agg(
        history_n=("n", "sum"),
        history_successes=("successes", "sum"),
    )


def entity_status_and_team(
    current: pd.DataFrame,
    season_table: pd.DataFrame,
    entity_type: str,
    eval_year: int,
) -> tuple[np.ndarray, np.ndarray]:
    spec = ENTITY_SPECS[entity_type]
    entity_col = spec["id"]
    prior = season_table.loc[season_table["season"] < eval_year]
    previous_ids = set(
        season_table.loc[season_table["season"] == eval_year - 1, entity_col]
    )
    all_prior_ids = set(prior[entity_col])
    status = np.select(
        [
            current[entity_col].isin(previous_ids),
            current[entity_col].isin(all_prior_ids),
        ],
        ["continuing", "returning"],
        default="new",
    )
    last_prior_team = (
        prior.sort_values([entity_col, "season"], kind="stable")
        .drop_duplicates(entity_col, keep="last")
        .set_index(entity_col)["team_last"]
    )
    mapped_team = current[entity_col].map(last_prior_team)
    team_status = np.select(
        [mapped_team.isna(), mapped_team == current[spec["team"]]],
        ["new_or_unknown", "same_team"],
        default="moved_team",
    )
    return status, team_status


def entity_performance_persistence(
    frame: pd.DataFrame,
    entity_seasons: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    subgroup_rows: list[dict[str, object]] = []
    prior_rows: list[dict[str, object]] = []
    for entity_type, season_table in entity_seasons.items():
        spec = ENTITY_SPECS[entity_type]
        entity_col = spec["id"]
        for eval_year in YEARS[1:]:
            current = frame.loc[frame["season"] == eval_year].copy()
            y = current["control_success"].to_numpy(float)
            status, team_status = entity_status_and_team(
                current, season_table, entity_type, eval_year
            )
            current["entity_status"] = status
            current["team_status"] = team_status
            eval_group = current.groupby(entity_col, observed=True).agg(
                eval_n=("control_success", "size"),
                eval_successes=("control_success", "sum"),
            )
            eval_group["eval_rate"] = eval_group["eval_successes"] / eval_group["eval_n"]

            all_prior = history_aggregate(
                season_table, entity_col, eval_year, window=None
            )
            all_prior_global_mean = float(
                all_prior["history_successes"].sum() / all_prior["history_n"].sum()
            )
            all_prior_global_prediction = np.full(
                len(current), all_prior_global_mean, dtype=float
            )
            all_prior_global_brier = brier(y, all_prior_global_prediction)

            for window_name, window in WINDOWS.items():
                history = history_aggregate(
                    season_table, entity_col, eval_year, window=window
                )
                prior = fit_beta_prior(history)
                history = history.copy()
                history["eb_rate"] = (
                    history["history_successes"] + prior["alpha"]
                ) / (history["history_n"] + prior["alpha"] + prior["beta"])
                prediction = current[entity_col].map(history["eb_rate"])
                known = prediction.notna().to_numpy()
                prediction = prediction.fillna(prior["prior_mean"]).to_numpy(float)
                candidate_brier = brier(y, prediction)
                window_prior_prediction = np.full(
                    len(current), prior["prior_mean"], dtype=float
                )
                window_prior_brier = brier(y, window_prior_prediction)

                joined = history[["history_n", "history_successes", "eb_rate"]].join(
                    eval_group, how="inner"
                )
                x = joined["eb_rate"].to_numpy(float)
                observed_rate = joined["eval_rate"].to_numpy(float)
                eval_weight = joined["eval_n"].to_numpy(float)
                metric_rows.append(
                    {
                        "entity_type": entity_type,
                        "eval_year": eval_year,
                        "history_window": window_name,
                        "history_start_year": int(
                            min(YEARS) if window is None else max(min(YEARS), eval_year - window)
                        ),
                        "history_end_year": eval_year - 1,
                        "history_entities": len(history),
                        "eval_entities": len(eval_group),
                        "known_eval_entities": len(joined),
                        "known_entity_coverage": len(joined) / len(eval_group),
                        "known_row_coverage": float(known.mean()),
                        "prior_mean": prior["prior_mean"],
                        "prior_strength": prior["prior_strength"],
                        "optimizer_success": prior["optimizer_success"],
                        "all_prior_global_mean": all_prior_global_mean,
                        "all_prior_global_brier": all_prior_global_brier,
                        "window_prior_brier": window_prior_brier,
                        "history_eb_brier": candidate_brier,
                        "temporal_prior_delta_vs_all_prior_global": (
                            window_prior_brier - all_prior_global_brier
                        ),
                        "entity_delta_vs_window_prior": (
                            candidate_brier - window_prior_brier
                        ),
                        "total_delta_vs_all_prior_global": (
                            candidate_brier - all_prior_global_brier
                        ),
                        "known_only_brier": brier(y[known], prediction[known]),
                        "prediction_mean": float(prediction.mean()),
                        "target_mean": float(y.mean()),
                        "absolute_bias": abs(float(prediction.mean() - y.mean())),
                        "entity_pearson_unweighted": float(
                            np.corrcoef(x, observed_rate)[0, 1]
                        )
                        if len(joined) >= 3 and np.std(x) > 0 and np.std(observed_rate) > 0
                        else math.nan,
                        "entity_pearson_row_weighted": weighted_corr(
                            x, observed_rate, eval_weight
                        ),
                        "entity_spearman_unweighted": float(
                            np.corrcoef(rankdata(x), rankdata(observed_rate))[0, 1]
                        )
                        if len(joined) >= 3
                        else math.nan,
                        "entity_spearman_row_weighted": weighted_corr(
                            rankdata(x), rankdata(observed_rate), eval_weight
                        ),
                        "entity_calibration_slope_row_weighted": weighted_slope(
                            x, observed_rate, eval_weight
                        ),
                    }
                )
                prior_rows.append(
                    {
                        "entity_type": entity_type,
                        "eval_year": eval_year,
                        "history_window": window_name,
                        **prior,
                    }
                )

                current["prediction"] = prediction
                current["known_history"] = np.where(known, "known", "unknown")
                for dimension in (
                    "entity_status",
                    "team_status",
                    "known_history",
                    "game_type",
                ):
                    for value, group in current.groupby(dimension, observed=True):
                        group_y = group["control_success"].to_numpy(float)
                        group_p = group["prediction"].to_numpy(float)
                        subgroup_rows.append(
                            {
                                "entity_type": entity_type,
                                "eval_year": eval_year,
                                "history_window": window_name,
                                "dimension": dimension,
                                "group_value": str(value),
                                "rows": len(group),
                                "row_share": len(group) / len(current),
                                "target_mean": float(group_y.mean()),
                                "prediction_mean": float(group_p.mean()),
                                "brier": brier(group_y, group_p),
                                "absolute_bias": abs(float(group_p.mean() - group_y.mean())),
                            }
                        )

            asof_prediction = (
                current[spec["asof_rate"]]
                .fillna(all_prior_global_mean)
                .to_numpy(float)
            )
            asof_known = current[spec["asof_rate"]].notna().to_numpy()
            asof_brier = brier(y, asof_prediction)
            metric_rows.append(
                {
                    "entity_type": entity_type,
                    "eval_year": eval_year,
                    "history_window": "official_row_local_asof",
                    "history_start_year": min(YEARS),
                    "history_end_year": eval_year,
                    "history_entities": math.nan,
                    "eval_entities": len(eval_group),
                    "known_eval_entities": int(current.loc[asof_known, entity_col].nunique()),
                    "known_entity_coverage": float(
                        current.loc[asof_known, entity_col].nunique() / len(eval_group)
                    ),
                    "known_row_coverage": float(asof_known.mean()),
                    "prior_mean": all_prior_global_mean,
                    "prior_strength": math.nan,
                    "optimizer_success": True,
                    "all_prior_global_mean": all_prior_global_mean,
                    "all_prior_global_brier": all_prior_global_brier,
                    "window_prior_brier": math.nan,
                    "history_eb_brier": asof_brier,
                    "temporal_prior_delta_vs_all_prior_global": math.nan,
                    "entity_delta_vs_window_prior": math.nan,
                    "total_delta_vs_all_prior_global": (
                        asof_brier - all_prior_global_brier
                    ),
                    "known_only_brier": brier(y[asof_known], asof_prediction[asof_known]),
                    "prediction_mean": float(asof_prediction.mean()),
                    "target_mean": float(y.mean()),
                    "absolute_bias": abs(float(asof_prediction.mean() - y.mean())),
                    "entity_pearson_unweighted": math.nan,
                    "entity_pearson_row_weighted": math.nan,
                    "entity_spearman_unweighted": math.nan,
                    "entity_spearman_row_weighted": math.nan,
                    "entity_calibration_slope_row_weighted": math.nan,
                }
            )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(subgroup_rows),
        pd.DataFrame(prior_rows),
    )


def entity_game_type_conditional_persistence(
    frame: pd.DataFrame,
    entity_seasons: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for entity_type, season_table in entity_seasons.items():
        entity_col = ENTITY_SPECS[entity_type]["id"]
        entity_game_season = (
            frame.groupby(
                ["season", entity_col, "game_type"],
                observed=True,
                sort=False,
            )
            .agg(n=("control_success", "size"), successes=("control_success", "sum"))
            .reset_index()
        )
        for eval_year in YEARS[1:]:
            current = frame.loc[
                frame["season"] == eval_year,
                [entity_col, "game_type", "control_success"],
            ].copy()
            current["game_type_key"] = current["game_type"].astype(str)
            y = current["control_success"].to_numpy(float)
            all_prior = history_aggregate(
                season_table, entity_col, eval_year, window=None
            )
            all_prior_global_mean = float(
                all_prior["history_successes"].sum() / all_prior["history_n"].sum()
            )
            all_prior_global_brier = brier(
                y, np.full(len(current), all_prior_global_mean, dtype=float)
            )
            eval_game_rates = current.groupby("game_type_key", observed=True)[
                "control_success"
            ].mean()
            eval_residual = (
                current["control_success"].astype(float)
                - current["game_type_key"].map(eval_game_rates).astype(float)
            )
            eval_entity = (
                pd.DataFrame(
                    {
                        entity_col: current[entity_col].to_numpy(),
                        "residual": eval_residual.to_numpy(float),
                    }
                )
                .groupby(entity_col, observed=True)
                .agg(eval_n=("residual", "size"), eval_residual=("residual", "mean"))
            )

            for window_name, window in WINDOWS.items():
                lower = (
                    min(YEARS)
                    if window is None
                    else max(min(YEARS), eval_year - window)
                )
                history_game = entity_game_season.loc[
                    (entity_game_season["season"] >= lower)
                    & (entity_game_season["season"] < eval_year)
                ].copy()
                history_game["game_type_key"] = history_game["game_type"].astype(str)
                game_totals = history_game.groupby("game_type_key", observed=True).agg(
                    n=("n", "sum"), successes=("successes", "sum")
                )
                game_totals["rate"] = game_totals["successes"] / game_totals["n"]
                history_game["expected_successes"] = (
                    history_game["n"]
                    * history_game["game_type_key"].map(game_totals["rate"]).astype(float)
                )
                history_game["residual_sum"] = (
                    history_game["successes"] - history_game["expected_successes"]
                )
                entity_residual = history_game.groupby(entity_col, observed=True).agg(
                    history_n=("n", "sum"), residual_sum=("residual_sum", "sum")
                )
                raw_history = history_aggregate(
                    season_table, entity_col, eval_year, window=window
                )
                prior = fit_beta_prior(raw_history)
                entity_residual["raw_residual"] = (
                    entity_residual["residual_sum"] / entity_residual["history_n"]
                )
                entity_residual["shrunk_residual"] = (
                    entity_residual["residual_sum"]
                    / (entity_residual["history_n"] + prior["prior_strength"])
                )

                game_baseline = (
                    current["game_type_key"]
                    .map(game_totals["rate"])
                    .fillna(all_prior_global_mean)
                    .to_numpy(float)
                )
                mapped_residual = current[entity_col].map(
                    entity_residual["shrunk_residual"]
                )
                known = mapped_residual.notna().to_numpy()
                mapped_residual = mapped_residual.fillna(0.0).to_numpy(float)
                conditional_prediction = np.clip(
                    game_baseline + mapped_residual, 1e-6, 1 - 1e-6
                )
                baseline_brier = brier(y, game_baseline)
                conditional_brier = brier(y, conditional_prediction)

                joined = entity_residual.join(eval_entity, how="inner")
                history_residual_values = joined["raw_residual"].to_numpy(float)
                eval_residual_values = joined["eval_residual"].to_numpy(float)
                eval_weights = joined["eval_n"].to_numpy(float)
                rows.append(
                    {
                        "entity_type": entity_type,
                        "eval_year": eval_year,
                        "history_window": window_name,
                        "history_start_year": lower,
                        "history_end_year": eval_year - 1,
                        "known_entity_coverage": len(joined) / len(eval_entity),
                        "known_row_coverage": float(known.mean()),
                        "shrinkage_strength": prior["prior_strength"],
                        "game_type_baseline_brier": baseline_brier,
                        "game_type_plus_entity_brier": conditional_brier,
                        "entity_delta_vs_game_type_baseline": (
                            conditional_brier - baseline_brier
                        ),
                        "game_type_baseline_delta_vs_all_prior_global": (
                            baseline_brier - all_prior_global_brier
                        ),
                        "total_delta_vs_all_prior_global": (
                            conditional_brier - all_prior_global_brier
                        ),
                        "residual_pearson_unweighted": float(
                            np.corrcoef(
                                history_residual_values, eval_residual_values
                            )[0, 1]
                        )
                        if len(joined) >= 3
                        and np.std(history_residual_values) > 0
                        and np.std(eval_residual_values) > 0
                        else math.nan,
                        "residual_pearson_row_weighted": weighted_corr(
                            history_residual_values,
                            eval_residual_values,
                            eval_weights,
                        ),
                        "residual_spearman_row_weighted": weighted_corr(
                            rankdata(history_residual_values),
                            rankdata(eval_residual_values),
                            eval_weights,
                        ),
                        "residual_slope_row_weighted": weighted_slope(
                            history_residual_values,
                            eval_residual_values,
                            eval_weights,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def shared_support_kitagawa(
    frame: pd.DataFrame,
    previous_year: int,
    current_year: int,
    dimension: str,
) -> dict[str, object]:
    subset = frame.loc[
        frame["season"].isin([previous_year, current_year]),
        ["season", dimension, "control_success"],
    ]
    grouped = (
        subset.groupby(["season", dimension], observed=True, dropna=False)
        ["control_success"]
        .agg(["size", "mean"])
        .reset_index()
    )
    previous = grouped.loc[grouped["season"] == previous_year].set_index(dimension)
    current = grouped.loc[grouped["season"] == current_year].set_index(dimension)
    shared = previous.index.intersection(current.index)
    previous_shared = previous.loc[shared]
    current_shared = current.loc[shared]
    previous_coverage = float(previous_shared["size"].sum() / previous["size"].sum())
    current_coverage = float(current_shared["size"].sum() / current["size"].sum())
    previous_weight = previous_shared["size"] / previous_shared["size"].sum()
    current_weight = current_shared["size"] / current_shared["size"].sum()
    previous_rate = previous_shared["mean"]
    current_rate = current_shared["mean"]
    within = float(
        np.sum(0.5 * (previous_weight + current_weight) * (current_rate - previous_rate))
    )
    composition = float(
        np.sum(0.5 * (previous_rate + current_rate) * (current_weight - previous_weight))
    )
    previous_overall = float(previous["mean"].mul(previous["size"]).sum() / previous["size"].sum())
    current_overall = float(current["mean"].mul(current["size"]).sum() / current["size"].sum())
    previous_shared_mean = float(np.sum(previous_weight * previous_rate))
    current_shared_mean = float(np.sum(current_weight * current_rate))
    shared_delta = current_shared_mean - previous_shared_mean
    overall_delta = current_overall - previous_overall
    return {
        "previous_year": previous_year,
        "current_year": current_year,
        "dimension": dimension,
        "previous_categories": len(previous),
        "current_categories": len(current),
        "shared_categories": len(shared),
        "previous_shared_row_coverage": previous_coverage,
        "current_shared_row_coverage": current_coverage,
        "previous_overall_target_mean": previous_overall,
        "current_overall_target_mean": current_overall,
        "overall_delta": overall_delta,
        "previous_shared_target_mean": previous_shared_mean,
        "current_shared_target_mean": current_shared_mean,
        "shared_support_delta": shared_delta,
        "within_group_component": within,
        "composition_component": composition,
        "decomposition_numerical_error": shared_delta - within - composition,
        "unshared_support_gap": overall_delta - shared_delta,
    }


def drift_decomposition(
    frame: pd.DataFrame,
    entity_seasons: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dimensions = [
        "pitcher_id",
        "batter_id",
        "game_type",
        "pitcher_team_id",
        "batter_team_id",
        "pitcher_hand",
        "batter_hand",
        "count_state",
        "inning_bucket",
        "num_runners_on",
        "base_state",
        "score_bucket",
        "li_bucket",
        "joint_context",
    ]
    decomposition_rows: list[dict[str, object]] = []
    roster_rows: list[dict[str, object]] = []
    for current_year in YEARS[1:]:
        previous_year = current_year - 1
        for dimension in dimensions:
            decomposition_rows.append(
                shared_support_kitagawa(
                    frame, previous_year, current_year, dimension
                )
            )

        for entity_type, season_table in entity_seasons.items():
            spec = ENTITY_SPECS[entity_type]
            entity_col = spec["id"]
            previous_ids = set(
                season_table.loc[season_table["season"] == previous_year, entity_col]
            )
            all_earlier_ids = set(
                season_table.loc[season_table["season"] < previous_year, entity_col]
            )
            current_ids = set(
                season_table.loc[season_table["season"] == current_year, entity_col]
            )
            previous = frame.loc[frame["season"] == previous_year]
            current = frame.loc[frame["season"] == current_year]
            previous_status = np.where(
                previous[entity_col].isin(current_ids), "retained_next_year", "exiting"
            )
            current_status = np.select(
                [
                    current[entity_col].isin(previous_ids),
                    current[entity_col].isin(all_earlier_ids),
                ],
                ["continuing", "returning"],
                default="new",
            )
            for side, data, status in (
                ("previous", previous, previous_status),
                ("current", current, current_status),
            ):
                temporary = pd.DataFrame(
                    {
                        "status": status,
                        "target": data["control_success"].to_numpy(float),
                    }
                )
                for value, group in temporary.groupby("status", observed=True):
                    roster_rows.append(
                        {
                            "entity_type": entity_type,
                            "previous_year": previous_year,
                            "current_year": current_year,
                            "side": side,
                            "status": value,
                            "rows": len(group),
                            "row_share": len(group) / len(data),
                            "target_mean": float(group["target"].mean()),
                        }
                    )
    return pd.DataFrame(decomposition_rows), pd.DataFrame(roster_rows)


def semantics_markdown(train: Path, data_description: Path) -> str:
    return f"""# Semantics And Provenance

- Generated: `{kst_now()}`
- Official train input: `{train}`
- Official local description: `{data_description}`
- Hidden 2025 test read: `false`
- Distributed five-row test read: `false`
- External player/team mapping: `false`
- Trackman read in Phase A-D: `false`

## Frozen semantics

- One row represents one pitch.
- `control_success` is the organizer-defined binary outcome for that pitch.
- The task is row-level probability prediction, not player season rate or game-win prediction.
- `home_win_expectancy` is an input context field, not the target.
- `pitcher_id`, `batter_id`, and team IDs are anonymous identifiers.
- `game_type=R` is regular-season first-team play and `F` the second-team league, as clarified by the organizers; no real-league identities are used.
- Main IDs and Trackman IDs are separate namespaces without an official direct join key.

## Provenance boundary

Only the official `train.csv` is read for Phase A-D measurements. Domain knowledge may
later motivate interpretation, but real-player identities and external statistics cannot
enter features, lookups, weights, or model artifacts. Evaluation rows remain independent.
"""


def visual_atlas(
    output: Path,
    context: pd.DataFrame,
    year_pairs: pd.DataFrame,
    continuity: pd.DataFrame,
    entity_year_pairs: pd.DataFrame,
    asof_continuity: pd.DataFrame,
    persistence: pd.DataFrame,
    conditional_persistence: pd.DataFrame,
    decomposition: pd.DataFrame,
    roster: pd.DataFrame,
) -> None:
    colors = {
        "black": "#202124",
        "gray": "#687076",
        "teal": "#00796B",
        "red": "#C43D3D",
        "gold": "#B7791F",
        "blue": "#2563A6",
        "green": "#4D7C0F",
    }
    figures: list[tuple[str, str, go.Figure]] = []

    overall = context.loc[context["dimension"] == "overall"].sort_values("season")
    game_type = context.loc[context["dimension"] == "game_type"].copy()
    game_type["group_value"] = game_type["group_value"].astype(str)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=overall["season"],
            y=overall["target_mean"],
            mode="lines+markers",
            name="Overall",
            line={"color": colors["black"], "width": 3},
        )
    )
    for value, color in (("F", colors["red"]), ("R", colors["teal"])):
        selected = game_type.loc[game_type["group_value"] == value].sort_values("season")
        figure.add_trace(
            go.Scatter(
                x=selected["season"],
                y=selected["target_mean"],
                mode="lines+markers",
                name=f"game_type={value}",
                line={"color": color, "width": 2},
                customdata=np.stack([selected["rows"]], axis=-1),
                hovertemplate="year=%{x}<br>target mean=%{y:.6f}<br>rows=%{customdata[0]:,}<extra></extra>",
            )
        )
    figure.update_layout(yaxis_title="control_success mean", xaxis_title="Season")
    figures.append(
        (
            "1. Target drift by season",
            "The F/R lines are data codes only; their real-league meaning remains unknown.",
            figure,
        )
    )

    def symmetric_pair_matrix(
        source: pd.DataFrame, metric: str, diagonal: float
    ) -> np.ndarray:
        matrix = np.full((len(YEARS), len(YEARS)), np.nan, dtype=float)
        np.fill_diagonal(matrix, diagonal)
        positions = {year: index for index, year in enumerate(YEARS)}
        for record in source.to_dict("records"):
            left = positions[int(record["from_year"])]
            right = positions[int(record["to_year"])]
            matrix[left, right] = float(record[metric])
            matrix[right, left] = float(record[metric])
        return matrix

    target_pair_source = year_pairs.drop_duplicates(["from_year", "to_year"])
    target_matrix = symmetric_pair_matrix(
        target_pair_source, "absolute_overall_target_delta", 0.0
    )
    context_matrix = symmetric_pair_matrix(
        year_pairs.loc[year_pairs["dimension"] == "joint_context"],
        "jensen_shannon_divergence_nats",
        0.0,
    )
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Absolute target-mean difference",
            "Joint-context Jensen-Shannon divergence",
            "Pitcher F/R-adjusted residual correlation",
            "Batter F/R-adjusted residual correlation",
        ),
        vertical_spacing=0.16,
    )
    sequential_scale = [
        [0.0, "#f7fafc"],
        [0.55, "#d9a441"],
        [1.0, "#c43d3d"],
    ]
    for column, matrix, title in (
        (1, target_matrix, "Target difference"),
        (2, context_matrix, "Context JSD"),
    ):
        figure.add_trace(
            go.Heatmap(
                z=matrix,
                x=list(YEARS),
                y=list(YEARS),
                colorscale=sequential_scale,
                colorbar={"title": title, "len": 0.38, "y": 0.79, "x": 0.46 + 0.52 * (column - 1)},
                text=np.round(matrix, 4),
                texttemplate="%{text}",
                hovertemplate="from=%{y}<br>to=%{x}<br>value=%{z:.6f}<extra></extra>",
            ),
            row=1,
            col=column,
        )
    for column, entity_type in ((1, "pitcher"), (2, "batter")):
        pair_source = entity_year_pairs.loc[
            entity_year_pairs["entity_type"] == entity_type
        ]
        matrix = np.full((len(YEARS), len(YEARS)), np.nan, dtype=float)
        np.fill_diagonal(matrix, 1.0)
        positions = {year: index for index, year in enumerate(YEARS)}
        for record in pair_source.to_dict("records"):
            left = positions[int(record["from_year"])]
            right = positions[int(record["to_year"])]
            matrix[left, right] = float(
                record["game_type_adjusted_residual_pearson_to_row_weighted"]
            )
        figure.add_trace(
            go.Heatmap(
                z=matrix,
                x=list(YEARS),
                y=list(YEARS),
                zmin=-1,
                zmax=1,
                colorscale=[
                    [0.0, colors["red"]],
                    [0.5, "#f7fafc"],
                    [1.0, colors["blue"]],
                ],
                colorbar={
                    "title": "Correlation",
                    "len": 0.38,
                    "y": 0.22,
                    "x": 0.46 + 0.52 * (column - 1),
                },
                text=np.round(matrix, 3),
                texttemplate="%{text}",
                hovertemplate="from=%{y}<br>to=%{x}<br>correlation=%{z:.4f}<extra></extra>",
            ),
            row=2,
            col=column,
        )
    figure.update_yaxes(autorange="reversed")
    figures.append(
        (
            "2. Pairwise year relationship map",
            "Upper-triangle entity cells are strict earlier-to-later comparisons. Context similarity and outcome similarity are separate quantities.",
            figure,
        )
    )

    selected = continuity.loc[continuity["source_scope"] == "all_prior_years"]
    figure = go.Figure()
    line_specs = [
        ("pitcher", "target_entity_coverage", "Pitcher ID coverage", colors["blue"], "solid"),
        ("pitcher", "target_row_coverage", "Pitcher row coverage", colors["blue"], "dash"),
        ("batter", "target_entity_coverage", "Batter ID coverage", colors["gold"], "solid"),
        ("batter", "target_row_coverage", "Batter row coverage", colors["gold"], "dash"),
    ]
    for entity_type, metric, name, color, dash in line_specs:
        part = selected.loc[selected["entity_type"] == entity_type].sort_values("to_year")
        figure.add_trace(
            go.Scatter(
                x=part["to_year"],
                y=part[metric],
                mode="lines+markers",
                name=name,
                line={"color": color, "dash": dash, "width": 2},
            )
        )
    figure.update_layout(yaxis_title="Coverage", xaxis_title="Evaluation season")
    figure.update_yaxes(range=[0, 1.02], tickformat=".0%")
    figures.append(
        (
            "3. Prior-known anonymous ID coverage",
            "Entity coverage counts IDs equally; row coverage weights frequent participants.",
            figure,
        )
    )

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Pitcher: total effect",
            "Batter: total effect",
            "Pitcher: entity-only effect",
            "Batter: entity-only effect",
        ),
        shared_yaxes=True,
        vertical_spacing=0.16,
    )
    window_colors = {
        "all_prior": colors["black"],
        "last_1_season": colors["red"],
        "last_2_seasons": colors["teal"],
        "last_3_seasons": colors["gold"],
        "official_row_local_asof": colors["blue"],
    }
    for column, entity_type in enumerate(("pitcher", "batter"), start=1):
        for window in window_colors:
            part = persistence.loc[
                (persistence["entity_type"] == entity_type)
                & (persistence["history_window"] == window)
            ].sort_values("eval_year")
            figure.add_trace(
                go.Scatter(
                    x=part["eval_year"],
                    y=part["total_delta_vs_all_prior_global"] * 10_000,
                    mode="lines+markers",
                    name=window,
                    legendgroup=window,
                    showlegend=column == 1,
                    line={"color": window_colors[window], "width": 2},
                    hovertemplate="year=%{x}<br>Delta Brier x1e4=%{y:.3f}<extra></extra>",
                ),
                row=1,
                col=column,
            )
            if window != "official_row_local_asof":
                figure.add_trace(
                    go.Scatter(
                        x=part["eval_year"],
                        y=part["entity_delta_vs_window_prior"] * 10_000,
                        mode="lines+markers",
                        name=window,
                        legendgroup=window,
                        showlegend=False,
                        line={"color": window_colors[window], "width": 2},
                        hovertemplate=(
                            "year=%{x}<br>entity-only Delta Brier x1e4=%{y:.3f}"
                            "<extra></extra>"
                        ),
                    ),
                    row=2,
                    col=column,
                )
        figure.add_hline(y=0, line={"color": colors["gray"], "dash": "dot"}, row=1, col=column)
        figure.add_hline(y=0, line={"color": colors["gray"], "dash": "dot"}, row=2, col=column)
    figure.update_yaxes(title_text="Total Delta Brier x 1e4", row=1, col=1)
    figure.update_yaxes(title_text="Entity-only Delta Brier x 1e4", row=2, col=1)
    figure.update_xaxes(title_text="Evaluation season")
    figures.append(
        (
            "4. Does frozen prior entity history predict the next season?",
            "Top: recency plus entity effects versus the all-prior global mean. Bottom: entity deviations versus the same-window constant prior. Negative is better.",
            figure,
        )
    )

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("asof count exact agreement", "asof rate absolute difference p99"),
    )
    for entity_type, color in (("pitcher", colors["blue"]), ("batter", colors["gold"])):
        part = asof_continuity.loc[
            asof_continuity["entity_type"] == entity_type
        ].sort_values("eval_year")
        figure.add_trace(
            go.Scatter(
                x=part["eval_year"],
                y=part["asof_n_exact_known_fraction"],
                mode="lines+markers",
                name=entity_type,
                legendgroup=entity_type,
                line={"color": color, "width": 2},
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=part["eval_year"],
                y=part["asof_rate_abs_diff_p99"],
                mode="lines+markers",
                name=entity_type,
                legendgroup=entity_type,
                showlegend=False,
                line={"color": color, "width": 2},
            ),
            row=1,
            col=2,
        )
    figure.update_yaxes(range=[0, 1.02], tickformat=".0%", row=1, col=1)
    figure.update_yaxes(title_text="Absolute difference", row=1, col=2)
    figure.update_xaxes(title_text="Evaluation season")
    figures.append(
        (
            "5. Official asof continuity across season boundaries",
            "Exact count agreement and near-zero rate differences test whether repeated anonymous IDs carry cumulative official history.",
            figure,
        )
    )

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Raw-rate Pearson",
            "Raw-rate calibration slope",
            "game_type-adjusted residual Pearson",
            "Entity Delta Brier beyond game_type",
        ),
        vertical_spacing=0.16,
    )
    for entity_type, color in (("pitcher", colors["blue"]), ("batter", colors["gold"])):
        part = persistence.loc[
            (persistence["entity_type"] == entity_type)
            & (persistence["history_window"] == "all_prior")
        ].sort_values("eval_year")
        figure.add_trace(
            go.Scatter(
                x=part["eval_year"],
                y=part["entity_pearson_row_weighted"],
                mode="lines+markers",
                name=entity_type,
                legendgroup=entity_type,
                line={"color": color, "width": 2},
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=part["eval_year"],
                y=part["entity_calibration_slope_row_weighted"],
                mode="lines+markers",
                name=entity_type,
                legendgroup=entity_type,
                showlegend=False,
                line={"color": color, "width": 2},
            ),
            row=1,
            col=2,
        )
        for window, dash in (("all_prior", "solid"), ("last_1_season", "dash")):
            conditional_part = conditional_persistence.loc[
                (conditional_persistence["entity_type"] == entity_type)
                & (conditional_persistence["history_window"] == window)
            ].sort_values("eval_year")
            trace_name = f"{entity_type} {window}"
            figure.add_trace(
                go.Scatter(
                    x=conditional_part["eval_year"],
                    y=conditional_part["residual_pearson_row_weighted"],
                    mode="lines+markers",
                    name=trace_name,
                    legendgroup=trace_name,
                    showlegend=True,
                    line={"color": color, "dash": dash, "width": 2},
                ),
                row=2,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=conditional_part["eval_year"],
                    y=(
                        conditional_part["entity_delta_vs_game_type_baseline"]
                        * 10_000
                    ),
                    mode="lines+markers",
                    name=trace_name,
                    legendgroup=trace_name,
                    showlegend=False,
                    line={"color": color, "dash": dash, "width": 2},
                ),
                row=2,
                col=2,
            )
    figure.add_hline(y=0, line={"color": colors["gray"], "dash": "dot"}, row=1, col=1)
    figure.add_hline(y=1, line={"color": colors["gray"], "dash": "dot"}, row=1, col=2)
    figure.add_hline(y=0, line={"color": colors["gray"], "dash": "dot"}, row=2, col=1)
    figure.add_hline(y=0, line={"color": colors["gray"], "dash": "dot"}, row=2, col=2)
    figure.update_yaxes(title_text="Delta Brier x 1e4", row=2, col=2)
    figure.update_xaxes(title_text="Evaluation season")
    figures.append(
        (
            "6. Raw and game_type-adjusted entity persistence",
            "The lower row separates player residuals from the F/R game type. A single fold cannot establish a permanent mechanism.",
            figure,
        )
    )

    drift_dimensions = ["pitcher_id", "batter_id", "game_type", "joint_context"]
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=tuple(drift_dimensions),
        shared_yaxes=True,
    )
    for index, dimension in enumerate(drift_dimensions):
        row = index // 2 + 1
        column = index % 2 + 1
        part = decomposition.loc[decomposition["dimension"] == dimension].copy()
        part["transition"] = (
            part["previous_year"].astype(str) + "->" + part["current_year"].astype(str)
        )
        for metric, name, color in (
            ("within_group_component", "Within-group", colors["blue"]),
            ("composition_component", "Composition", colors["gold"]),
        ):
            figure.add_trace(
                go.Bar(
                    x=part["transition"],
                    y=part[metric] * 10_000,
                    name=name,
                    legendgroup=name,
                    showlegend=index == 0,
                    marker_color=color,
                    customdata=np.stack(
                        [
                            part["previous_shared_row_coverage"],
                            part["current_shared_row_coverage"],
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "%{x}<br>component x1e4=%{y:.3f}"
                        "<br>previous shared coverage=%{customdata[0]:.2%}"
                        "<br>current shared coverage=%{customdata[1]:.2%}<extra></extra>"
                    ),
                ),
                row=row,
                col=column,
            )
        figure.add_trace(
            go.Scatter(
                x=part["transition"],
                y=part["overall_delta"] * 10_000,
                mode="lines+markers",
                name="Overall delta",
                legendgroup="Overall delta",
                showlegend=index == 0,
                line={"color": colors["black"], "width": 2},
            ),
            row=row,
            col=column,
        )
        figure.add_hline(
            y=0,
            line={"color": colors["gray"], "dash": "dot"},
            row=row,
            col=column,
        )
    figure.update_layout(barmode="relative")
    figure.update_yaxes(title_text="Target-rate change x 1e4", row=1, col=1)
    figure.update_yaxes(title_text="Target-rate change x 1e4", row=2, col=1)
    figures.append(
        (
            "7. Shared-support drift decomposition",
            "Bars decompose change on categories observed in both years. The black line is the full overall change; gaps can reflect entrants/exits or unshared contexts.",
            figure,
        )
    )

    current_roster = roster.loc[roster["side"] == "current"].copy()
    current_roster["transition"] = (
        current_roster["previous_year"].astype(str)
        + "->"
        + current_roster["current_year"].astype(str)
    )
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Pitcher row composition", "Batter row composition"),
        shared_yaxes=True,
    )
    status_colors = {
        "continuing": colors["teal"],
        "returning": colors["gold"],
        "new": colors["red"],
    }
    for column, entity_type in enumerate(("pitcher", "batter"), start=1):
        entity_part = current_roster.loc[current_roster["entity_type"] == entity_type]
        for status in ("continuing", "returning", "new"):
            part = entity_part.loc[entity_part["status"] == status]
            figure.add_trace(
                go.Bar(
                    x=part["transition"],
                    y=part["row_share"],
                    name=status,
                    legendgroup=status,
                    showlegend=column == 1,
                    marker_color=status_colors[status],
                    customdata=np.stack([part["target_mean"], part["rows"]], axis=-1),
                    hovertemplate=(
                        "%{x}<br>row share=%{y:.2%}"
                        "<br>target mean=%{customdata[0]:.6f}"
                        "<br>rows=%{customdata[1]:,}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
    figure.update_layout(barmode="stack")
    figure.update_yaxes(range=[0, 1], tickformat=".0%")
    figures.append(
        (
            "8. Continuing, returning, and new participant row shares",
            "This distinguishes ID-count churn from the share of actual pitches or plate appearances.",
            figure,
        )
    )

    for title, _, figure in figures:
        height = 760 if title.startswith(("2.", "4.", "6.", "7.")) else 520
        figure.update_layout(
            template="plotly_white",
            height=height,
            margin={"l": 70, "r": 30, "t": 70, "b": 70},
            font={"family": "Arial, sans-serif", "size": 13, "color": colors["black"]},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
            hovermode="x unified",
        )

    plotly_js = get_plotlyjs()
    sections = []
    for title, note, figure in figures:
        fragment = pio.to_html(
            figure,
            full_html=False,
            include_plotlyjs=False,
            config={"displaylogo": False, "responsive": True},
        )
        sections.append(
            f"<section><h2>{title}</h2><p>{note}</p>{fragment}</section>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Year Relationship And Identity Continuity Audit V1</title>
<style>
body {{ margin: 0; background: #ffffff; color: #202124; font-family: Arial, sans-serif; }}
header, section {{ max-width: 1280px; margin: 0 auto; padding: 28px 32px; }}
header {{ border-bottom: 3px solid #202124; }}
h1 {{ margin: 0 0 10px; font-size: 28px; letter-spacing: 0; }}
h2 {{ margin: 0 0 6px; font-size: 20px; letter-spacing: 0; }}
p {{ margin: 0 0 16px; color: #5f6368; line-height: 1.5; }}
section {{ border-bottom: 1px solid #d8dde3; }}
.meta {{ color: #5f6368; font-size: 14px; }}
@media (max-width: 700px) {{ header, section {{ padding: 20px 12px; }} h1 {{ font-size: 23px; }} }}
</style>
<script>{plotly_js}</script>
</head>
<body>
<header>
<h1>Year Relationship &amp; Identity Continuity Audit V1</h1>
<p class="meta">Generated {kst_now()} | Train-only Phase A-D | Hidden test not read</p>
<p>Interactive views support macro interpretation. Hover for exact values and click legend entries to isolate series. Numeric CSV files remain the source of record.</p>
</header>
{''.join(sections)}
</body>
</html>
"""
    (output / "VISUAL_ATLAS.html").write_text(html, encoding="utf-8")
    index_lines = [
        "# Visualization Index",
        "",
        f"- Generated: `{kst_now()}`",
        "- Artifact: `VISUAL_ATLAS.html`",
        "- Format: self-contained interactive HTML; no internet required",
        "- Source of record: numeric CSV artifacts in the same directory",
        "",
    ]
    for title, note, _ in figures:
        index_lines.extend([f"## {title}", "", note, ""])
    index_lines.extend(
        [
            "## Interpretation guardrails",
            "",
            "- F/R is the organizers' game-type code (regular first-team vs second-team); no real-league mapping is used.",
            "- Shared-support decomposition is descriptive, not causal.",
            "- A stable ID does not automatically imply stable performance.",
            "- No chart proves the hidden 2025 relationship.",
            "",
        ]
    )
    (output / "VISUALIZATION_INDEX.md").write_text(
        "\n".join(index_lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    start = time.time()
    if not args.train.is_file():
        raise FileNotFoundError(args.train)
    if not args.data_description.is_file():
        raise FileNotFoundError(args.data_description)
    args.output.mkdir(parents=True, exist_ok=False)
    log("Phase A-D audit started")

    input_hashes = {
        "train_sha256": sha256_file(args.train),
        "data_description_sha256": sha256_file(args.data_description),
        "source_sha256": sha256_file(Path(__file__)),
    }
    log("Input hashes complete; reading official train only")
    frame = pd.read_csv(
        args.train,
        usecols=USECOLS,
        dtype=DTYPES,
        encoding="utf-8-sig",
        low_memory=False,
    )
    if len(frame) != 1_475_092:
        raise ValueError(f"Unexpected train row count: {len(frame)}")
    if tuple(sorted(frame["season"].unique().tolist())) != YEARS:
        raise ValueError("Unexpected season set")
    if sorted(frame["control_success"].unique().tolist()) != [0, 1]:
        raise ValueError("Target is not binary")
    add_context_columns(frame)
    log(f"Train loaded: {len(frame):,} rows, {len(frame.columns)} audit columns")

    (args.output / "SEMANTICS_AND_PROVENANCE.md").write_text(
        semantics_markdown(args.train, args.data_description), encoding="utf-8"
    )
    context = year_context_matrix(frame)
    year_pairs = year_pair_relationship_matrix(context)
    write_csv(context, args.output / "YEAR_TARGET_AND_CONTEXT_MATRIX.csv")
    write_csv(year_pairs, args.output / "YEAR_PAIR_RELATIONSHIP_MATRIX.csv")
    log("Phase A and year/context summaries complete")

    entity_seasons = {
        entity_type: build_entity_season(frame, entity_type)
        for entity_type in ENTITY_SPECS
    }
    entity_year_pairs = entity_year_pair_persistence(frame, entity_seasons)
    continuity, spans, statuses = entity_continuity(frame, entity_seasons)
    asof_continuity = asof_cross_season_continuity(frame, entity_seasons)
    write_csv(continuity, args.output / "ENTITY_CONTINUITY_MATRIX.csv")
    write_csv(
        entity_year_pairs,
        args.output / "ENTITY_YEAR_PAIR_PERSISTENCE.csv",
    )
    write_csv(spans, args.output / "ENTITY_SPAN_DISTRIBUTION.csv")
    write_csv(statuses, args.output / "ENTITY_STATUS_ROW_SHARES.csv")
    write_csv(asof_continuity, args.output / "ASOF_CROSS_SEASON_CONTINUITY.csv")
    log("Phase B entity and as-of continuity complete")

    persistence, persistence_subgroups, beta_priors = entity_performance_persistence(
        frame, entity_seasons
    )
    conditional_persistence = entity_game_type_conditional_persistence(
        frame, entity_seasons
    )
    write_csv(persistence, args.output / "ENTITY_PERFORMANCE_PERSISTENCE.csv")
    write_csv(
        conditional_persistence,
        args.output / "ENTITY_GAME_TYPE_CONDITIONAL_PERSISTENCE.csv",
    )
    write_csv(
        persistence_subgroups,
        args.output / "ENTITY_PERSISTENCE_SUBGROUPS.csv",
    )
    write_csv(beta_priors, args.output / "ENTITY_BETA_PRIORS.csv")
    log("Phase C performance persistence complete")

    decomposition, roster = drift_decomposition(frame, entity_seasons)
    write_csv(decomposition, args.output / "DRIFT_DECOMPOSITION.csv")
    write_csv(roster, args.output / "ROSTER_TRANSITION_TARGETS.csv")
    log("Phase D descriptive drift decomposition complete")

    visual_atlas(
        args.output,
        context,
        year_pairs,
        continuity,
        entity_year_pairs,
        asof_continuity,
        persistence,
        conditional_persistence,
        decomposition,
        roster,
    )
    log("Interactive visual atlas complete")

    generated_files = sorted(
        path.name for path in args.output.iterdir() if path.is_file()
    )
    output_hashes = {
        name: sha256_file(args.output / name) for name in generated_files
    }
    elapsed = time.time() - start
    manifest = {
        "audit_id": "year_relationship_identity_continuity_audit_v1",
        "scope": "Phase A-D only",
        "started_at_kst": datetime.fromtimestamp(start, KST).strftime(
            "%Y-%m-%d %H:%M:%S KST (UTC+09:00)"
        ),
        "completed_at_kst": kst_now(),
        "elapsed_seconds": elapsed,
        "python": sys.version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "train_rows": len(frame),
        "train_seasons": list(YEARS),
        "hidden_test_read": False,
        "distributed_test_read": False,
        "trackman_read": False,
        "external_data_read": False,
        "model_fit": False,
        "submission_generated": False,
        "input_hashes": input_hashes,
        "output_hashes_before_manifest": output_hashes,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    write_json(args.output / "RUN_MANIFEST.json", manifest)
    log(f"Audit complete in {elapsed:.2f} seconds; outputs={len(generated_files) + 1}")


if __name__ == "__main__":
    main()
