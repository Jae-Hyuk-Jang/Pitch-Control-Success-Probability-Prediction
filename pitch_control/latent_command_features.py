from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PITCH_TYPES = ("fastball", "breaking", "offspeed")
PITCH_RATE_COLUMNS = {
    "fastball": "asof_pitcher_fastball_rate",
    "breaking": "asof_pitcher_breaking_rate",
    "offspeed": "asof_pitcher_offspeed_rate",
}
AXES = ("success", "ball")
MIX_SMOOTHING = 200.0
CONTEXT_SMOOTHING = 384.0
LIKELIHOOD_MIN = 0.25
LIKELIHOOD_MAX = 4.0


def feature_names() -> tuple[str, ...]:
    names = [f"r7b_q_{pitch}" for pitch in PITCH_TYPES]
    names.extend(f"r7b_q_minus_base_{pitch}" for pitch in PITCH_TYPES)
    names.extend(
        [
            "r7b_q_entropy",
            "r7b_base_entropy",
            "r7b_q_to_base_kl",
            "r7b_policy_l1_shift",
        ]
    )
    for axis in AXES:
        names.extend(
            [
                f"r7b_{axis}_expected_mean",
                f"r7b_{axis}_weighted_dev",
                f"r7b_{axis}_weighted_abs_dev",
                f"r7b_{axis}_weighted_variance",
                f"r7b_{axis}_expected_reliability",
                f"r7b_{axis}_known_coverage",
                f"r7b_{axis}_best_minus_expected",
                f"r7b_{axis}_expected_minus_worst",
            ]
        )
    names.extend(
        [
            "r7b_expected_success_minus_ball",
            "r7b_weighted_dev_success_minus_ball",
        ]
    )
    return tuple(names)


R7B_FEATURES = feature_names()


@dataclass(frozen=True)
class PolicyArtifact:
    global_probability: np.ndarray
    hand: pd.DataFrame
    count: pd.DataFrame
    context: pd.DataFrame


@dataclass(frozen=True)
class CommandArtifact:
    axis: str
    global_mean: float
    cells: dict[int, pd.DataFrame]


@dataclass(frozen=True)
class LatentCommandArtifacts:
    policy: PolicyArtifact
    command: dict[str, CommandArtifact]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _string(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), "__NA__").astype(str)


def _normalize_hand(series: pd.Series) -> pd.Series:
    value = _string(series).str.strip().str.upper()
    return value.map(
        {
            "1": "L",
            "1.0": "L",
            "L": "L",
            "LEFT": "L",
            "2": "R",
            "2.0": "R",
            "R": "R",
            "RIGHT": "R",
        }
    ).fillna("__NA__")


def _policy_keys(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    output["pitcher_hand_key"] = _normalize_hand(frame["pitcher_hand"])
    output["batter_hand_key"] = _normalize_hand(frame["batter_hand"])
    balls = _numeric(frame["balls_before"]).fillna(-1).astype(int)
    strikes = _numeric(frame["strikes_before"]).fillna(-1).astype(int)
    output["count_key"] = balls.astype(str) + "-" + strikes.astype(str)
    inning = _numeric(frame["inning"]).fillna(-1)
    output["inning_phase_key"] = np.select(
        [inning <= 3, inning <= 6, inning <= 9],
        ["early", "middle", "late"],
        default="extra",
    )
    return output


def load_latent_command_artifacts(model_dir: Path) -> LatentCommandArtifacts:
    model_dir = Path(model_dir)
    manifest = json.loads(
        (model_dir / "latent_command_manifest.json").read_text(encoding="utf-8")
    )
    if tuple(manifest["features"]) != R7B_FEATURES:
        raise RuntimeError("R7-B feature manifest mismatch")

    hand = pd.read_csv(model_dir / "r7b_policy_hand_2025.csv", low_memory=False)
    count = pd.read_csv(model_dir / "r7b_policy_count_2025.csv", low_memory=False)
    context = pd.read_csv(model_dir / "r7b_policy_context_2025.csv", low_memory=False)
    for table in (hand, count, context):
        for column in ("pitcher_hand_key", "batter_hand_key", "count_key", "inning_phase_key"):
            if column in table:
                table[column] = _string(table[column])

    command: dict[str, CommandArtifact] = {}
    for axis in AXES:
        cell = pd.read_csv(
            model_dir / f"r7b_{axis}_command_cell_2025.csv", low_memory=False
        )
        cell["pitcher_id"] = _string(cell["pitcher_id"])
        cell["pitch_group"] = _numeric(cell["pitch_group"]).astype(int)
        cells = {
            index: cell.loc[
                cell["pitch_group"].eq(index),
                ["pitcher_id", "cell_dev", "cell_reliability"],
            ].set_index("pitcher_id")
            for index in range(len(PITCH_TYPES))
        }
        command[axis] = CommandArtifact(
            axis=axis,
            global_mean=float(manifest["global_command"][axis]),
            cells=cells,
        )
    policy = PolicyArtifact(
        global_probability=np.asarray(manifest["global_policy"], dtype=float),
        hand=hand,
        count=count,
        context=context,
    )
    return LatentCommandArtifacts(policy=policy, command=command)


def _predict_policy(
    frame: pd.DataFrame, artifact: PolicyArtifact
) -> tuple[np.ndarray, np.ndarray]:
    keys = _policy_keys(frame).reset_index(drop=True)
    keys["_order"] = np.arange(len(keys), dtype=np.int64)
    hand_keys = ["pitcher_hand_key"]
    count_keys = ["pitcher_hand_key", "batter_hand_key", "count_key"]
    context_keys = [*count_keys, "inning_phase_key"]
    output = keys.merge(
        artifact.hand[
            hand_keys
            + ["hand_n", *[f"hand_{pitch}_prob" for pitch in PITCH_TYPES]]
        ],
        on=hand_keys,
        how="left",
        validate="many_to_one",
    )
    output = output.merge(
        artifact.count[
            count_keys
            + ["count_n", *[f"count_{pitch}_prob" for pitch in PITCH_TYPES]]
        ],
        on=count_keys,
        how="left",
        validate="many_to_one",
    )
    output = output.merge(
        artifact.context[
            context_keys
            + ["context_n", *[f"context_{pitch}_prob" for pitch in PITCH_TYPES]]
        ],
        on=context_keys,
        how="left",
        validate="many_to_one",
    )
    output = output.sort_values("_order", kind="stable").reset_index(drop=True)

    hand_probability = np.column_stack(
        [
            _numeric(output[f"hand_{pitch}_prob"])
            .fillna(artifact.global_probability[index])
            .to_numpy(float)
            for index, pitch in enumerate(PITCH_TYPES)
        ]
    )
    hand_probability /= hand_probability.sum(axis=1, keepdims=True)
    context_probability = np.column_stack(
        [
            _numeric(output[f"context_{pitch}_prob"])
            .fillna(_numeric(output[f"count_{pitch}_prob"]))
            .fillna(pd.Series(hand_probability[:, index]))
            .to_numpy(float)
            for index, pitch in enumerate(PITCH_TYPES)
        ]
    )
    context_probability /= context_probability.sum(axis=1, keepdims=True)

    raw_mix = np.column_stack(
        [
            _numeric(frame[PITCH_RATE_COLUMNS[pitch]]).to_numpy(float)
            for pitch in PITCH_TYPES
        ]
    )
    raw_mix = np.where(np.isfinite(raw_mix), raw_mix, 0.0)
    raw_sum = raw_mix.sum(axis=1, keepdims=True)
    normalized_mix = np.divide(
        raw_mix,
        raw_sum,
        out=hand_probability.copy(),
        where=raw_sum > 1e-8,
    )
    mix_n = _numeric(frame["asof_pitcher_pitchmix_n"]).fillna(0.0).to_numpy(float)
    mix_reliability = mix_n / (mix_n + MIX_SMOOTHING)
    base = (
        mix_reliability[:, None] * normalized_mix
        + (1.0 - mix_reliability[:, None]) * hand_probability
    )
    context_n = _numeric(output["context_n"]).fillna(0.0).to_numpy(float)
    context_reliability = context_n / (context_n + CONTEXT_SMOOTHING)
    likelihood_ratio = np.clip(
        context_probability / np.clip(hand_probability, 1e-8, None),
        LIKELIHOOD_MIN,
        LIKELIHOOD_MAX,
    )
    candidate = base * np.power(likelihood_ratio, context_reliability[:, None])
    candidate /= candidate.sum(axis=1, keepdims=True)
    return base, candidate


def add_latent_command_features(
    frame: pd.DataFrame, artifacts: LatentCommandArtifacts
) -> pd.DataFrame:
    required = {
        "pitcher_id",
        "pitcher_hand",
        "batter_hand",
        "balls_before",
        "strikes_before",
        "inning",
        "asof_pitcher_pitchmix_n",
        "r1_current_success_mean",
        "r1_current_ball_mean",
        *PITCH_RATE_COLUMNS.values(),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"R7-B row columns missing: {missing}")
    output = frame.copy()
    pitcher_id = _string(output["pitcher_id"])
    base, q = _predict_policy(output, artifacts.policy)
    for index, pitch in enumerate(PITCH_TYPES):
        output[f"r7b_q_{pitch}"] = q[:, index].astype("float32")
        output[f"r7b_q_minus_base_{pitch}"] = (
            q[:, index] - base[:, index]
        ).astype("float32")
    clipped_q = np.clip(q, 1e-12, 1.0)
    clipped_base = np.clip(base, 1e-12, 1.0)
    output["r7b_q_entropy"] = (-np.sum(q * np.log(clipped_q), axis=1)).astype(
        "float32"
    )
    output["r7b_base_entropy"] = (
        -np.sum(base * np.log(clipped_base), axis=1)
    ).astype("float32")
    output["r7b_q_to_base_kl"] = np.sum(
        q * np.log(clipped_q / clipped_base), axis=1
    ).astype("float32")
    output["r7b_policy_l1_shift"] = np.sum(np.abs(q - base), axis=1).astype(
        "float32"
    )

    expected_means: dict[str, np.ndarray] = {}
    weighted_devs: dict[str, np.ndarray] = {}
    for axis in AXES:
        artifact = artifacts.command[axis]
        dev_columns = []
        reliability_columns = []
        known_columns = []
        for pitch_index in range(len(PITCH_TYPES)):
            mapping = artifact.cells[pitch_index]
            dev_series = pitcher_id.map(mapping["cell_dev"])
            reliability_series = pitcher_id.map(mapping["cell_reliability"])
            known_columns.append(dev_series.notna().to_numpy(float))
            dev_columns.append(dev_series.fillna(0.0).to_numpy(float))
            reliability_columns.append(
                reliability_series.fillna(0.0).to_numpy(float)
            )
        deviations = np.column_stack(dev_columns)
        reliabilities = np.column_stack(reliability_columns)
        known_matrix = np.column_stack(known_columns)
        current = _numeric(output[f"r1_current_{axis}_mean"]).to_numpy(float)
        if not np.isfinite(current).all():
            raise RuntimeError(f"non-finite current prior for {axis}")
        means = np.clip(current[:, None] + deviations, 0.0, 1.0)
        expected = np.sum(q * means, axis=1)
        weighted_dev = np.sum(q * deviations, axis=1)
        expected_means[axis] = expected
        weighted_devs[axis] = weighted_dev
        output[f"r7b_{axis}_expected_mean"] = expected.astype("float32")
        output[f"r7b_{axis}_weighted_dev"] = weighted_dev.astype("float32")
        output[f"r7b_{axis}_weighted_abs_dev"] = np.sum(
            q * np.abs(deviations), axis=1
        ).astype("float32")
        output[f"r7b_{axis}_weighted_variance"] = np.sum(
            q * np.square(deviations - weighted_dev[:, None]), axis=1
        ).astype("float32")
        output[f"r7b_{axis}_expected_reliability"] = np.sum(
            q * reliabilities, axis=1
        ).astype("float32")
        output[f"r7b_{axis}_known_coverage"] = np.sum(
            q * known_matrix, axis=1
        ).astype("float32")
        output[f"r7b_{axis}_best_minus_expected"] = (
            np.max(means, axis=1) - expected
        ).astype("float32")
        output[f"r7b_{axis}_expected_minus_worst"] = (
            expected - np.min(means, axis=1)
        ).astype("float32")

    output["r7b_expected_success_minus_ball"] = (
        expected_means["success"] - expected_means["ball"]
    ).astype("float32")
    output["r7b_weighted_dev_success_minus_ball"] = (
        weighted_devs["success"] - weighted_devs["ball"]
    ).astype("float32")
    values = output[list(R7B_FEATURES)].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite R7-B feature")
    return output
