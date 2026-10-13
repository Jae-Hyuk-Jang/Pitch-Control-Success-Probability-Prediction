import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("LG_AIMERS_ROOT", "."))
DATA = ROOT / "data"
RUNS = ROOT / "runs"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = RUNS / f"visual_eda_atlas_v1_{STAMP}"
FIG = OUT / "figures"
TAB = OUT / "tables"

TARGET = "control_success"
COUNT_ORDER = ["0-0", "0-1", "0-2", "1-0", "1-1", "1-2", "2-0", "2-1", "2-2", "3-0", "3-1", "3-2"]
PITCH_ORDER = ["fastball", "breaking", "offspeed", "other"]
PITCH_COLORS = {
    "fastball": "#2a78d6",
    "breaking": "#eb6834",
    "offspeed": "#1baf7a",
    "other": "#777777",
}
ACCENT = "#111111"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#fbfbf8",
            "axes.facecolor": "#fbfbf8",
            "savefig.facecolor": "#fbfbf8",
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#4a4a45",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#111111",
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ensure_out() -> None:
    FIG.mkdir(parents=True, exist_ok=False)
    TAB.mkdir(parents=True, exist_ok=False)


def savefig(name: str) -> None:
    path = FIG / name
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_table(name: str, df: pd.DataFrame) -> None:
    df.to_csv(TAB / name, index=False)


def write_json(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def make_count_state(df: pd.DataFrame) -> pd.Series:
    return df["balls_before"].astype(str) + "-" + df["strikes_before"].astype(str)


def heatmap(ax, matrix: pd.DataFrame, title: str, cmap: str = "viridis", fmt: str = ".3f") -> None:
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = values[i, j]
            if np.isfinite(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=6, color="white" if val < np.nanmean(values) else "black")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)


def plot_target_drift(train: pd.DataFrame) -> pd.DataFrame:
    by_season = train.groupby("season")[TARGET].agg(n="size", target_mean="mean").reset_index()
    save_table("target_by_season.csv", by_season)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(by_season["season"], by_season["target_mean"], marker="o", lw=2.4, color="#2a78d6")
    ax.axhline(train[TARGET].mean(), color="#555555", lw=1, ls="--", label="global mean")
    for _, r in by_season.iterrows():
        ax.text(r["season"], r["target_mean"] + 0.004, f"{r['target_mean']:.3f}", ha="center", fontsize=8)
    ax.set_title("Control Success Rate Drift By Season", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("season")
    ax.set_ylabel("target mean")
    ax.set_ylim(0.45, 0.59)
    ax.legend(frameon=False)
    savefig("01_target_drift_by_season.png")
    return by_season


def plot_count_heatmap(train: pd.DataFrame) -> pd.DataFrame:
    train = train.copy()
    train["count_state"] = make_count_state(train)
    tab = train.groupby(["season", "count_state"], observed=True)[TARGET].agg(n="size", target_mean="mean").reset_index()
    save_table("target_by_season_count_state.csv", tab)
    pivot = tab.pivot(index="season", columns="count_state", values="target_mean").reindex(columns=COUNT_ORDER)

    fig, ax = plt.subplots(figsize=(11, 5))
    heatmap(ax, pivot, "Target Mean By Season And Count", cmap="magma", fmt=".3f")
    ax.set_xlabel("count before pitch")
    ax.set_ylabel("season")
    savefig("02_target_heatmap_season_count.png")
    return tab


def plot_asof_curves(train: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_batter_success_rate",
    ]
    rows = []
    for col in cols:
        tmp = train[[col, TARGET]].dropna().copy()
        tmp["decile"] = pd.qcut(tmp[col], 10, labels=False, duplicates="drop")
        g = tmp.groupby("decile", observed=True).agg(n=(TARGET, "size"), x_mean=(col, "mean"), target_mean=(TARGET, "mean")).reset_index()
        g["column"] = col
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    save_table("asof_decile_curves.csv", out)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.ravel(), cols):
        g = out[out["column"] == col]
        ax.plot(g["x_mean"], g["target_mean"], marker="o", lw=2.2, color="#2a78d6")
        ax.set_title(col, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("feature decile mean")
        ax.set_ylabel("target mean")
    savefig("03_asof_rate_reliability_curves.png")
    return out


def plot_cold_start(train: pd.DataFrame) -> pd.DataFrame:
    bins = [-1, 0, 10, 50, 100, 200, 500, 1000, 2000, 5000, math.inf]
    labels = ["0", "1-10", "11-50", "51-100", "101-200", "201-500", "501-1k", "1k-2k", "2k-5k", "5k+"]
    cols = ["asof_pitcher_n", "asof_batter_n"]
    rows = []
    for col in cols:
        tmp = train[[col, TARGET]].copy()
        tmp["count_bin"] = pd.cut(tmp[col], bins=bins, labels=labels)
        g = tmp.groupby("count_bin", observed=True)[TARGET].agg(n="size", target_mean="mean").reset_index()
        g["column"] = col
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    save_table("cold_start_curves.csv", out)

    fig, ax = plt.subplots(figsize=(10, 5))
    for col, color in [("asof_pitcher_n", "#2a78d6"), ("asof_batter_n", "#eb6834")]:
        g = out[out["column"] == col]
        ax.plot(np.arange(len(g)), g["target_mean"], marker="o", lw=2.2, label=col, color=color)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.axhline(train[TARGET].mean(), color="#555555", lw=1, ls="--")
    ax.set_title("Cold-Start / Exposure Curves", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("pre-pitch historical count bin")
    ax.set_ylabel("target mean")
    ax.legend(frameon=False)
    savefig("04_cold_start_exposure_curves.png")
    return out


def plot_pitcher_profile(train: pd.DataFrame) -> pd.DataFrame:
    prof = (
        train.groupby("pitcher_id")
        .agg(
            n=(TARGET, "size"),
            observed_success=(TARGET, "mean"),
            mean_asof_success=("asof_pitcher_success_rate", "mean"),
            mean_asof_reverse=("asof_pitcher_reverse_rate", "mean"),
            mean_asof_middle=("asof_pitcher_middle_rate", "mean"),
        )
        .reset_index()
    )
    prof = prof[prof["n"] >= 500].copy()
    prof["log_n"] = np.log1p(prof["n"])
    save_table("pitcher_profile_n500.csv", prof.sort_values("n", ascending=False))

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sc = ax.scatter(
        prof["mean_asof_success"],
        prof["observed_success"],
        s=8 + 7 * prof["log_n"],
        c=prof["mean_asof_reverse"],
        cmap="coolwarm",
        alpha=0.68,
        edgecolors="none",
    )
    lo = min(prof["mean_asof_success"].min(), prof["observed_success"].min())
    hi = max(prof["mean_asof_success"].max(), prof["observed_success"].max())
    ax.plot([lo, hi], [lo, hi], color="#555555", lw=1, ls="--")
    ax.set_title("Pitcher Profile: Prior Success vs Observed Success", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("mean asof pitcher success rate")
    ax.set_ylabel("observed train success rate")
    cb = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("mean asof reverse rate")
    savefig("05_pitcher_profile_asof_vs_observed.png")
    return prof


def load_trackman() -> pd.DataFrame:
    cols = [
        "trackman_id",
        "season",
        "inning",
        "balls_before",
        "strikes_before",
        "pitcher_hand",
        "batter_hand",
        "pitch_type_group",
        "rel_speed",
        "spin_rate",
        "induced_vert_break",
        "horz_break",
        "extension",
        "rel_height",
        "rel_side",
        "zone_speed",
    ]
    tm = pd.read_csv(DATA / "trackman_history.csv", usecols=cols, low_memory=False, encoding="utf-8-sig")
    tm["count_state"] = make_count_state(tm)
    tm["inning_phase"] = pd.cut(
        tm["inning"],
        bins=[-np.inf, 3, 6, 9, np.inf],
        labels=["early_1_3", "mid_4_6", "late_7_9", "extra_10p"],
    ).astype(str)
    return tm


def plot_trackman_pitch_mix(tm: pd.DataFrame, target_by_season: pd.DataFrame) -> pd.DataFrame:
    mix = tm.groupby(["season", "pitch_type_group"], observed=True).agg(n=("trackman_id", "size")).reset_index()
    mix["share"] = mix["n"] / mix.groupby("season")["n"].transform("sum")
    save_table("trackman_pitch_group_share_by_season.csv", mix)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    for p in PITCH_ORDER:
        g = mix[mix["pitch_type_group"] == p].sort_values("season")
        ax1.plot(g["season"], g["share"], marker="o", lw=2.2, label=p, color=PITCH_COLORS[p])
    ax1.set_ylabel("Trackman pitch group share")
    ax1.set_xlabel("season")
    ax1.set_title("Pitch-Mix Drift And Main Target Drift", loc="left", fontsize=12, fontweight="bold")
    ax2 = ax1.twinx()
    ax2.plot(target_by_season["season"], target_by_season["target_mean"], marker="s", lw=2, color="#111111", label="target mean")
    ax2.set_ylabel("main train target mean")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, ncol=3, fontsize=8, loc="upper right")
    savefig("06_trackman_pitch_mix_vs_target_drift.png")
    return mix


def stratified_sample(df: pd.DataFrame, group_col: str, per_group: int, seed: int = 20260811) -> pd.DataFrame:
    parts = []
    for _, part in df.groupby(group_col, observed=True):
        if len(part) > per_group:
            parts.append(part.sample(per_group, random_state=seed))
        else:
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def plot_trackman_movement(tm: pd.DataFrame) -> None:
    cols = ["horz_break", "induced_vert_break", "pitch_type_group"]
    tmp = tm[cols].dropna()
    tmp = tmp[tmp["pitch_type_group"].isin(PITCH_ORDER)]
    smp = stratified_sample(tmp, "pitch_type_group", 35000)
    save_table("trackman_movement_sample.csv", smp)

    fig, ax = plt.subplots(figsize=(8, 7))
    for p in PITCH_ORDER:
        g = smp[smp["pitch_type_group"] == p]
        ax.scatter(g["horz_break"], g["induced_vert_break"], s=2, alpha=0.12, color=PITCH_COLORS[p], label=p, rasterized=True)
    ax.set_title("Trackman Movement Atlas", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("horizontal break")
    ax.set_ylabel("induced vertical break")
    ax.legend(markerscale=5, frameon=False)
    savefig("07_trackman_movement_atlas.png")


def plot_trackman_release(tm: pd.DataFrame) -> None:
    cols = ["rel_side", "rel_height", "pitch_type_group"]
    tmp = tm[cols].dropna()
    tmp = tmp[tmp["pitch_type_group"].isin(PITCH_ORDER)]
    smp = stratified_sample(tmp, "pitch_type_group", 30000)
    save_table("trackman_release_sample.csv", smp)

    fig, ax = plt.subplots(figsize=(8, 7))
    for p in PITCH_ORDER:
        g = smp[smp["pitch_type_group"] == p]
        ax.scatter(g["rel_side"], g["rel_height"], s=2, alpha=0.10, color=PITCH_COLORS[p], label=p, rasterized=True)
    ax.set_title("Trackman Release Point Atlas", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("release side")
    ax.set_ylabel("release height")
    ax.legend(markerscale=5, frameon=False)
    savefig("08_trackman_release_atlas.png")


def plot_trackman_policy(tm: pd.DataFrame) -> pd.DataFrame:
    policy = tm.groupby(["count_state", "pitch_type_group"], observed=True).agg(n=("trackman_id", "size")).reset_index()
    policy["share"] = policy["n"] / policy.groupby("count_state")["n"].transform("sum")
    save_table("trackman_pitch_policy_by_count.csv", policy)
    pivot = (
        policy.pivot(index="count_state", columns="pitch_type_group", values="share")
        .reindex(index=COUNT_ORDER, columns=PITCH_ORDER)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(7.5, 6))
    heatmap(ax, pivot, "Trackman Pitch Group Share By Count", cmap="viridis", fmt=".2f")
    ax.set_xlabel("pitch group")
    ax.set_ylabel("count before pitch")
    savefig("09_trackman_pitch_policy_count_heatmap.png")
    return policy


def plot_trackman_physics_summary(tm: pd.DataFrame) -> pd.DataFrame:
    phys_cols = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side", "zone_speed"]
    summary = tm.groupby(["season", "pitch_type_group"], observed=True)[phys_cols].agg(["count", "mean", "std"]).reset_index()
    summary.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in summary.columns]
    save_table("trackman_physics_summary.csv", summary)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics = ["rel_speed_mean", "spin_rate_mean", "extension_mean", "zone_speed_mean"]
    titles = ["Release Speed", "Spin Rate", "Extension", "Zone Speed"]
    for ax, metric, title in zip(axes.ravel(), metrics, titles):
        for p in PITCH_ORDER:
            g = summary[summary["pitch_type_group"] == p].sort_values("season")
            ax.plot(g["season"], g[metric], marker="o", lw=2, label=p, color=PITCH_COLORS[p])
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("season")
    axes[0, 0].legend(frameon=False, fontsize=8)
    savefig("10_trackman_physics_summary_by_season.png")
    return summary


def main() -> None:
    t0 = time.time()
    setup_style()
    ensure_out()
    print(f"OUT={OUT}", flush=True)

    train_cols = [
        "season",
        "balls_before",
        "strikes_before",
        "pitcher_id",
        "pitcher_hand",
        "batter_hand",
        "asof_pitcher_n",
        "asof_batter_n",
        "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_batter_success_rate",
        TARGET,
    ]
    train = pd.read_csv(DATA / "train.csv", usecols=train_cols, low_memory=False, encoding="utf-8-sig")

    target_by_season = plot_target_drift(train)
    plot_count_heatmap(train)
    plot_asof_curves(train)
    plot_cold_start(train)
    plot_pitcher_profile(train)

    tm = load_trackman()
    plot_trackman_pitch_mix(tm, target_by_season)
    plot_trackman_movement(tm)
    plot_trackman_release(tm)
    plot_trackman_policy(tm)
    plot_trackman_physics_summary(tm)

    summary = {
        "output": str(OUT),
        "runtime_seconds": round(time.time() - t0, 3),
        "figures": sorted([p.name for p in FIG.glob("*.png")]),
        "tables": sorted([p.name for p in TAB.glob("*.csv")]),
        "rule_notes": [
            "Hidden 2025 test was not used.",
            "Distributed 5-row test.csv was not used for distributional analysis.",
            "Trackman visualizations use only 2019-2024 historical data.",
            "True pitch trajectory and strike-zone heatmaps are not generated because plate coordinates and trajectory points are not provided.",
        ],
    }
    write_json("summary.json", summary)

    findings = [
        "# Visual EDA Atlas V1",
        "",
        f"run: `{OUT}`",
        f"runtime seconds: `{summary['runtime_seconds']}`",
        "",
        "## Scope",
        "",
        "This atlas visualizes official train target structure and 2019-2024 Trackman physics/context policy.",
        "It does not use hidden 2025 test rows or test-row aggregation.",
        "Actual pitch trajectories and strike-zone heatmaps are not possible because the data does not include plate coordinates or trajectory points.",
        "",
        "## Figures",
        "",
    ]
    for fig in summary["figures"]:
        findings.append(f"- `figures/{fig}`")
    findings.extend(
        [
            "",
            "## Tables",
            "",
        ]
    )
    for table in summary["tables"]:
        findings.append(f"- `tables/{table}`")
    (OUT / "findings.md").write_text("\n".join(findings), encoding="utf-8")

    print(f"runtime_seconds={summary['runtime_seconds']}", flush=True)


if __name__ == "__main__":
    main()
