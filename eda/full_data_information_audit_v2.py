from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


RATE_LABELS = {
    "success": "asof_pitcher_success_rate",
    "reverse": "asof_pitcher_reverse_rate",
    "middle": "asof_pitcher_middle_rate",
    "ball": "asof_pitcher_ball_rate",
    "strike": "asof_pitcher_strike_rate",
    "fastball": "asof_pitcher_fastball_rate",
    "breaking": "asof_pitcher_breaking_rate",
    "offspeed": "asof_pitcher_offspeed_rate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def schema_table(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        series = frame[column]
        nonnull = series.dropna()
        numeric = pd.api.types.is_numeric_dtype(series)
        examples = nonnull.astype(str).drop_duplicates().head(5).tolist()
        rows.append(
            {
                "source": source,
                "column": column,
                "dtype": str(series.dtype),
                "rows": int(len(series)),
                "missing": int(series.isna().sum()),
                "missing_rate": float(series.isna().mean()),
                "unique_nonnull": int(series.nunique(dropna=True)),
                "minimum": float(nonnull.min()) if numeric and len(nonnull) else None,
                "maximum": float(nonnull.max()) if numeric and len(nonnull) else None,
                "examples": " | ".join(examples),
            }
        )
    return pd.DataFrame(rows)


def parse_ids(series: pd.Series, prefix: str) -> np.ndarray:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    values = []
    for value in series.astype(str):
        match = pattern.match(value)
        values.append(int(match.group(1)) if match else -1)
    return np.asarray(values, dtype=np.int64)


# Training file only: differencing against the next row is valid for refining training targets and
# must never be applied to evaluation rows.
def reconstruct_binary(
    frame: pd.DataFrame,
    current_n: np.ndarray,
    next_n: np.ndarray,
    next_same_pitcher: np.ndarray,
    next_same_season: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    result_rows = []
    rounded_labels: dict[str, np.ndarray] = {}
    for label, column in RATE_LABELS.items():
        current_rate = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        next_rate = frame.groupby("pitcher_id", sort=False)[column].shift(-1)
        next_rate = pd.to_numeric(next_rate, errors="coerce").to_numpy(float)
        current_sum = np.where(current_n == 0, 0.0, current_n * current_rate)
        next_sum = next_n * next_rate
        delta = next_sum - current_sum
        rounded = np.rint(delta)
        residual = np.abs(delta - rounded)
        binary = np.isin(rounded, (0.0, 1.0))
        finite = np.isfinite(delta)
        valid_pitcher = next_same_pitcher & finite
        valid_season = next_same_season & finite
        rounded_labels[label] = np.where(valid_season & binary, rounded, np.nan)
        for scope, mask in (
            ("same_pitcher_next", valid_pitcher),
            ("same_season_next", valid_season),
        ):
            scope_residual = residual[mask]
            result_rows.append(
                {
                    "label": label,
                    "rate_column": column,
                    "scope": scope,
                    "rows": int(mask.sum()),
                    "binary_rows": int((mask & binary).sum()),
                    "binary_fraction": float(binary[mask].mean()) if mask.any() else None,
                    "residual_p50": float(np.quantile(scope_residual, 0.50)) if mask.any() else None,
                    "residual_p95": float(np.quantile(scope_residual, 0.95)) if mask.any() else None,
                    "residual_p99": float(np.quantile(scope_residual, 0.99)) if mask.any() else None,
                    "residual_max": float(scope_residual.max()) if mask.any() else None,
                    "within_0_02_fraction": float((scope_residual <= 0.02).mean()) if mask.any() else None,
                }
            )
    return pd.DataFrame(result_rows), rounded_labels


def label_relationships(
    frame: pd.DataFrame,
    labels: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict]:
    label_frame = pd.DataFrame(labels)
    valid = label_frame.notna().all(axis=1)
    observed = label_frame.loc[valid].astype(int)
    target = frame.loc[valid, "control_success"].to_numpy(int)

    success_agreement = float(np.mean(observed["success"].to_numpy() == target))
    pitch_sum = observed[["fastball", "breaking", "offspeed"]].sum(axis=1)
    observed["other_pitch"] = (1 - pitch_sum).clip(lower=0)
    observed["pitch_group_sum"] = pitch_sum + observed["other_pitch"]

    failure_combo = (
        "s=" + observed["success"].astype(str)
        + "|r=" + observed["reverse"].astype(str)
        + "|m=" + observed["middle"].astype(str)
        + "|b=" + observed["ball"].astype(str)
        + "|k=" + observed["strike"].astype(str)
    )
    failure_counts = (
        failure_combo.value_counts(dropna=False)
        .rename_axis("combination")
        .reset_index(name="rows")
    )
    failure_counts["fraction"] = failure_counts["rows"] / len(observed)

    pitch_group_counts = pd.DataFrame(
        {
            "pitch_group": ["fastball", "breaking", "offspeed", "other"],
            "rows": [
                int(observed["fastball"].sum()),
                int(observed["breaking"].sum()),
                int(observed["offspeed"].sum()),
                int(observed["other_pitch"].sum()),
            ],
        }
    )
    pitch_group_counts["fraction"] = pitch_group_counts["rows"] / len(observed)

    audit = {
        "rows_with_all_reconstructed_labels": int(valid.sum()),
        "coverage": float(valid.mean()),
        "success_target_agreement": success_agreement,
        "success_with_reverse": int(
            ((observed["success"] == 1) & (observed["reverse"] == 1)).sum()
        ),
        "success_with_middle": int(
            ((observed["success"] == 1) & (observed["middle"] == 1)).sum()
        ),
        "failure_without_reverse_or_middle": int(
            (
                (observed["success"] == 0)
                & (observed["reverse"] == 0)
                & (observed["middle"] == 0)
            ).sum()
        ),
        "pitch_group_exactly_one_fraction": float(
            (observed["pitch_group_sum"] == 1).mean()
        ),
        "pitch_group_multihot_rows": int((pitch_sum > 1).sum()),
    }
    return pd.concat(
        [
            failure_counts.assign(table="failure_combinations"),
            pitch_group_counts.rename(columns={"pitch_group": "combination"}).assign(
                table="pitch_group"
            ),
        ],
        ignore_index=True,
    ), audit


def main_sequence_audit(frame: pd.DataFrame) -> dict:
    ids = parse_ids(frame["row_id"], "TRAIN")
    season = pd.to_numeric(frame["season"], errors="raise").to_numpy(int)
    inning = pd.to_numeric(frame["inning"], errors="raise").to_numpy(int)
    run_top = pd.to_numeric(frame["run_top_before"], errors="raise").to_numpy(int)
    run_bot = pd.to_numeric(frame["run_bot_before"], errors="raise").to_numpy(int)
    team_a = np.minimum(
        pd.to_numeric(frame["pitcher_team_id"]).to_numpy(int),
        pd.to_numeric(frame["batter_team_id"]).to_numpy(int),
    )
    team_b = np.maximum(
        pd.to_numeric(frame["pitcher_team_id"]).to_numpy(int),
        pd.to_numeric(frame["batter_team_id"]).to_numpy(int),
    )
    signature = pd.DataFrame(
        {
            "season": season,
            "month": frame["game_month"].to_numpy(),
            "dayofweek": frame["game_dayofweek"].to_numpy(),
            "game_type": frame["game_type"].astype(str).to_numpy(),
            "team_a": team_a,
            "team_b": team_b,
        }
    )
    signature_change = np.r_[
        True,
        np.any(signature.iloc[1:].to_numpy() != signature.iloc[:-1].to_numpy(), axis=1),
    ]
    state_reset = np.r_[
        False,
        (inning[1:] < inning[:-1])
        | (run_top[1:] < run_top[:-1])
        | (run_bot[1:] < run_bot[:-1]),
    ]
    boundary = signature_change | state_reset
    segment = np.cumsum(boundary)
    segment_frame = signature.copy()
    segment_frame["segment"] = segment
    segments_per_signature = (
        segment_frame.groupby(list(signature.columns), observed=True)["segment"].nunique()
    )

    same_pitcher_batter = (
        frame["pitcher_id"].to_numpy()[1:] == frame["pitcher_id"].to_numpy()[:-1]
    ) & (
        frame["batter_id"].to_numpy()[1:] == frame["batter_id"].to_numpy()[:-1]
    )
    same_segment = segment[1:] == segment[:-1]
    balls = frame["balls_before"].to_numpy(int)
    strikes = frame["strikes_before"].to_numpy(int)
    plausible_count_step = (
        ((balls[1:] == balls[:-1] + 1) & (strikes[1:] == strikes[:-1]))
        | ((strikes[1:] == strikes[:-1] + 1) & (balls[1:] == balls[:-1]))
        | ((balls[1:] == balls[:-1]) & (strikes[1:] == strikes[:-1]))
    )
    candidate_same_pa = same_segment & same_pitcher_batter

    return {
        "row_id_pattern_valid_fraction": float((ids >= 0).mean()),
        "row_id_strictly_consecutive": bool(np.array_equal(ids, np.arange(1, len(ids) + 1))),
        "season_monotonic": bool(np.all(season[1:] >= season[:-1])),
        "season_backward_transitions": int((season[1:] < season[:-1]).sum()),
        "adjacent_same_signature_fraction": float((~signature_change[1:]).mean()),
        "heuristic_game_segments": int(segment.max()),
        "unique_coarse_game_signatures": int(signature.drop_duplicates().shape[0]),
        "signatures_split_into_multiple_segments": int((segments_per_signature > 1).sum()),
        "max_segments_for_one_signature": int(segments_per_signature.max()),
        "adjacent_candidate_same_pa_rows": int(candidate_same_pa.sum()),
        "candidate_same_pa_plausible_count_fraction": float(
            plausible_count_step[candidate_same_pa].mean()
        ) if candidate_same_pa.any() else None,
        "warning": (
            "Segments are heuristic because main data has no game_date, game_id, "
            "pitch number, plate-appearance id, or pitch-of-PA column."
        ),
    }


def contract_checks(frame: pd.DataFrame) -> dict:
    checks = {
        "row_id_unique": bool(frame["row_id"].is_unique),
        "target_binary": sorted(frame["control_success"].dropna().unique().tolist()),
        "run_total_equals_top_plus_bottom_fraction": float(
            (
                frame["run_total_before"]
                == frame["run_top_before"] + frame["run_bot_before"]
            ).mean()
        ),
        "num_runners_equals_bases_fraction": float(
            (
                frame["num_runners_on"]
                == frame[["runner_on_1b", "runner_on_2b", "runner_on_3b"]].sum(axis=1)
            ).mean()
        ),
        "pitchmix_n_equals_pitcher_n_fraction": float(
            (frame["asof_pitcher_pitchmix_n"] == frame["asof_pitcher_n"]).mean()
        ),
        "win_expectancy_sum_min": float(
            (frame["home_win_expectancy"] + frame["away_win_expectancy"]).min()
        ),
        "win_expectancy_sum_max": float(
            (frame["home_win_expectancy"] + frame["away_win_expectancy"]).max()
        ),
    }
    return checks


def trackman_audit(frame: pd.DataFrame, main_summary: dict) -> tuple[dict, pd.DataFrame]:
    game_pitch_duplicates = frame.duplicated(["trackman_game_id", "pitch_no"]).sum()
    dates = pd.to_datetime(frame["game_date"], format="mixed", errors="coerce")
    groups = frame.groupby("trackman_game_id", observed=True)["pitch_no"]
    game_pitch = groups.agg(["size", "min", "max", "nunique"]).reset_index()
    game_pitch["complete_from_min_to_max"] = (
        game_pitch["nunique"] == game_pitch["max"] - game_pitch["min"] + 1
    )

    trackman_pitchers = set(frame["pitcher_trackman_id"].dropna().astype(str).unique())
    trackman_batters = set(frame["batter_trackman_id"].dropna().astype(str).unique())
    main_pitchers = set(main_summary["pitcher_ids"])
    main_batters = set(main_summary["batter_ids"])

    season_rows = (
        frame.groupby("season", observed=True)
        .agg(
            rows=("trackman_id", "size"),
            games=("trackman_game_id", "nunique"),
            pitchers=("pitcher_trackman_id", "nunique"),
            batters=("batter_trackman_id", "nunique"),
        )
        .reset_index()
    )
    main_season = pd.DataFrame(main_summary["season_rows"])
    season_rows = season_rows.merge(main_season, on="season", how="outer")
    season_rows["trackman_minus_main_rows"] = season_rows["rows"] - season_rows["main_rows"]
    season_rows["main_to_trackman_row_ratio"] = season_rows["main_rows"] / season_rows["rows"]

    audit = {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "trackman_id_unique": bool(frame["trackman_id"].is_unique),
        "trackman_id_strictly_consecutive": bool(
            np.array_equal(
                frame["trackman_id"].to_numpy(int),
                np.arange(1, len(frame) + 1),
            )
        ),
        "file_order_pitch_no_monotonic_within_game": bool(
            frame.groupby("trackman_game_id", observed=True)["pitch_no"]
            .apply(lambda x: x.is_monotonic_increasing)
            .all()
        ),
        "game_pitch_key_duplicates": int(game_pitch_duplicates),
        "games": int(frame["trackman_game_id"].nunique()),
        "games_complete_between_observed_min_max_fraction": float(
            game_pitch["complete_from_min_to_max"].mean()
        ),
        "date_parse_success_fraction": float(dates.notna().mean()),
        "date_min": str(dates.min().date()),
        "date_max": str(dates.max().date()),
        "invalid_baseball_state_rows": {
            "inning_below_1": int((pd.to_numeric(frame["inning"]) < 1).sum()),
            "balls_outside_0_to_3": int(
                (~pd.to_numeric(frame["balls_before"]).between(0, 3)).sum()
            ),
            "strikes_outside_0_to_2": int(
                (~pd.to_numeric(frame["strikes_before"]).between(0, 2)).sum()
            ),
            "outs_outside_0_to_2": int(
                (~pd.to_numeric(frame["outs_before"]).between(0, 2)).sum()
            ),
        },
        "pitcher_trackman_ids": len(trackman_pitchers),
        "batter_trackman_ids": len(trackman_batters),
        "exact_string_overlap_main_trackman_pitcher_ids": len(main_pitchers & trackman_pitchers),
        "exact_string_overlap_main_trackman_batter_ids": len(main_batters & trackman_batters),
        "pitcher_teams": int(frame["pitcher_team"].nunique()),
        "batter_teams": int(frame["batter_team"].nunique()),
        "tagged_pitch_types": int(frame["tagged_pitch_type"].nunique()),
        "auto_pitch_types": int(frame["auto_pitch_type"].nunique()),
        "pitch_type_groups": sorted(frame["pitch_type_group"].dropna().unique().tolist()),
        "direct_join_contract": (
            "No official key maps main pitcher_id/batter_id/team_id to Trackman IDs or names."
        ),
    }
    return audit, season_rows


def run(root: Path, output: Path) -> None:
    started = time.time()
    output.mkdir(parents=True, exist_ok=True)
    data = root / "data"
    train_path = data / "train.csv"
    test_path = data / "test.csv"
    trackman_path = data / "trackman_history.csv"

    print("[1/4] loading complete train.csv", flush=True)
    train = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    train_schema = schema_table(train, "train")
    train_schema.to_csv(output / "train_schema.csv", index=False)

    test = pd.read_csv(test_path, encoding="utf-8-sig", low_memory=False)
    test_schema = schema_table(test, "test_schema_sample_only")
    test_schema.to_csv(output / "test_sample_schema.csv", index=False)

    current_n = pd.to_numeric(train["asof_pitcher_n"], errors="raise").to_numpy(float)
    next_n = (
        train.groupby("pitcher_id", sort=False)["asof_pitcher_n"]
        .shift(-1)
        .to_numpy(float)
    )
    next_season = (
        train.groupby("pitcher_id", sort=False)["season"].shift(-1).to_numpy(float)
    )
    current_season = pd.to_numeric(train["season"]).to_numpy(float)
    next_same_pitcher = np.isfinite(next_n) & np.isclose(next_n, current_n + 1.0)
    next_same_season = next_same_pitcher & (next_season == current_season)

    print("[2/4] reconstructing train-only auxiliary labels", flush=True)
    reconstruction, labels = reconstruct_binary(
        train,
        current_n=current_n,
        next_n=next_n,
        next_same_pitcher=next_same_pitcher,
        next_same_season=next_same_season,
    )
    reconstruction.to_csv(output / "reconstruction_quality.csv", index=False)
    relationships, relationship_audit = label_relationships(train, labels)
    relationships.to_csv(output / "reconstructed_label_relationships.csv", index=False)

    sequence_audit = main_sequence_audit(train)
    checks = contract_checks(train)
    season_rows = (
        train.groupby("season", observed=True)
        .agg(
            main_rows=("row_id", "size"),
            main_pitchers=("pitcher_id", "nunique"),
            main_batters=("batter_id", "nunique"),
            target_mean=("control_success", "mean"),
        )
        .reset_index()
    )
    main_summary = {
        "rows": int(len(train)),
        "columns": int(train.shape[1]),
        "memory_bytes_deep": int(train.memory_usage(index=True, deep=True).sum()),
        "pitchers": int(train["pitcher_id"].nunique()),
        "batters": int(train["batter_id"].nunique()),
        "teams": int(
            len(
                set(train["pitcher_team_id"].unique())
                | set(train["batter_team_id"].unique())
            )
        ),
        "seasons": sorted(train["season"].unique().tolist()),
        "same_pitcher_next_rows": int(next_same_pitcher.sum()),
        "same_season_next_rows": int(next_same_season.sum()),
        "rows_without_same_season_next": int((~next_same_season).sum()),
        "pitcher_ids": train["pitcher_id"].astype(str).unique().tolist(),
        "batter_ids": train["batter_id"].astype(str).unique().tolist(),
        "season_rows": season_rows.to_dict(orient="records"),
    }
    public_main_summary = {k: v for k, v in main_summary.items() if not k.endswith("_ids")}
    write_json(
        output / "main_audit.json",
        {
            "summary": public_main_summary,
            "contract_checks": checks,
            "sequence_audit": sequence_audit,
            "reconstructed_labels": relationship_audit,
            "test_sample_only": {
                "rows": int(len(test)),
                "columns": int(test.shape[1]),
                "season_values": sorted(test["season"].unique().tolist()),
                "distribution_analysis_permitted": False,
            },
        },
    )
    del train, test, labels

    print("[3/4] loading complete trackman_history.csv", flush=True)
    trackman = pd.read_csv(trackman_path, encoding="utf-8-sig", low_memory=False)
    trackman_schema = schema_table(trackman, "trackman_history")
    trackman_schema.to_csv(output / "trackman_schema.csv", index=False)
    trackman_result, source_comparison = trackman_audit(trackman, main_summary)
    source_comparison.to_csv(output / "main_trackman_season_comparison.csv", index=False)
    write_json(output / "trackman_audit.json", trackman_result)

    inventory = {
        "train_csv_bytes": train_path.stat().st_size,
        "test_schema_sample_csv_bytes": test_path.stat().st_size,
        "sample_submission_csv_bytes": (data / "sample_submission.csv").stat().st_size,
        "trackman_history_csv_bytes": trackman_path.stat().st_size,
    }
    write_json(
        output / "run_summary.json",
        {
            "inventory": inventory,
            "elapsed_seconds": time.time() - started,
            "train_rows_read": public_main_summary["rows"],
            "trackman_rows_read": trackman_result["rows"],
            "hidden_test_read": False,
            "test_schema_sample_rows_read": 5,
        },
    )
    print("[4/4] audit complete", flush=True)


def main() -> None:
    args = parse_args()
    run(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
