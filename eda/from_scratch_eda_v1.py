import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("LG_AIMERS_ROOT", "."))
DATA = ROOT / "data"
RUNS = ROOT / "runs"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = RUNS / f"from_scratch_eda_v1_{STAMP}"

TARGET = "control_success"
ID_COL = "row_id"

COUNT_BINS = [-1, 0, 10, 50, 100, 200, 500, 1000, 2000, 5000, math.inf]
COUNT_LABELS = ["0", "1-10", "11-50", "51-100", "101-200", "201-500", "501-1k", "1k-2k", "2k-5k", "5k+"]


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "tables").mkdir()


def write_json(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_table(name: str, df: pd.DataFrame) -> None:
    df.to_csv(OUT / "tables" / name, index=False)


def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    view = df.head(max_rows).copy()
    if view.empty:
        return "_empty_"
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append(f"| ... | {len(df) - max_rows} more rows |" + " |" * max(0, len(cols) - 2))
    return "\n".join(lines)


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["count_state"] = out["balls_before"].astype(str) + "-" + out["strikes_before"].astype(str)
    out["platoon_matchup"] = out["pitcher_hand"].astype(str) + "v" + out["batter_hand"].astype(str)
    out["inning_phase"] = pd.cut(
        out["inning"],
        bins=[-np.inf, 3, 6, 9, np.inf],
        labels=["early_1_3", "mid_4_6", "late_7_9", "extra_10p"],
    ).astype(str)
    out["score_abs_bin"] = pd.cut(
        out["score_diff_pitcher_team"].abs(),
        bins=[-np.inf, 0, 1, 3, 6, np.inf],
        labels=["tie", "1", "2_3", "4_6", "7p"],
    ).astype(str)
    out["li_bin"] = pd.cut(
        out["li"],
        bins=[-np.inf, 0.5, 1.0, 2.0, 4.0, np.inf],
        labels=["li_0_0.5", "li_0.5_1", "li_1_2", "li_2_4", "li_4p"],
    ).astype(str)
    out["runners_scoring"] = ((out["runner_on_2b"] == 1) | (out["runner_on_3b"] == 1)).astype(int)
    out["late_close"] = ((out["inning"] >= 7) & (out["score_diff_pitcher_team"].abs() <= 1)).astype(int)
    return out


def summarize_target(df: pd.DataFrame, group_cols: list[str], name: str) -> pd.DataFrame:
    g = (
        df.groupby(group_cols, dropna=False, observed=True)[TARGET]
        .agg(n="size", target_mean="mean", target_std="std")
        .reset_index()
    )
    g["lift_vs_global"] = g["target_mean"] - df[TARGET].mean()
    g = g.sort_values(["n", "target_mean"], ascending=[False, False])
    save_table(name, g)
    return g


def numeric_profile(df: pd.DataFrame, input_cols: list[str]) -> pd.DataFrame:
    rows = []
    y = df[TARGET].astype(float)
    for col in input_cols:
        if col == ID_COL:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        row = {
            "column": col,
            "n": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "missing_rate": float(s.isna().mean()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "p01": float(s.quantile(0.01)),
            "p50": float(s.quantile(0.50)),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
        }
        if s.nunique(dropna=True) > 1:
            row["pearson_target"] = float(s.corr(y))
        else:
            row["pearson_target"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("pearson_target", key=lambda x: x.abs(), ascending=False)
    save_table("numeric_profile.csv", out)
    return out


def categorical_profile(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    global_mean = df[TARGET].mean()
    for col in cols:
        if col not in df.columns:
            continue
        g = df.groupby(col, dropna=False, observed=True)[TARGET].agg(n="size", target_mean="mean").reset_index()
        g["column"] = col
        g["lift_abs"] = (g["target_mean"] - global_mean).abs()
        g = g.sort_values(["lift_abs", "n"], ascending=[False, False])
        rows.append(g[["column", col, "n", "target_mean", "lift_abs"]].rename(columns={col: "value"}))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    save_table("categorical_profile.csv", out)
    return out


def bin_rate_feature(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = pd.to_numeric(df[col], errors="coerce")
    tmp = df[[TARGET]].copy()
    tmp[col] = s
    tmp = tmp[tmp[col].notna()].copy()
    if tmp[col].nunique() <= 1:
        return pd.DataFrame()
    tmp["bin"] = pd.qcut(tmp[col], q=10, duplicates="drop")
    out = (
        tmp.groupby("bin", observed=True)
        .agg(n=(TARGET, "size"), feature_mean=(col, "mean"), target_mean=(TARGET, "mean"))
        .reset_index()
    )
    out["column"] = col
    out["bin"] = out["bin"].astype(str)
    return out[["column", "bin", "n", "feature_mean", "target_mean"]]


def asof_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rate_cols = [
        c
        for c in df.columns
        if c.startswith("asof_")
        and c.endswith("_rate")
        and c not in {"asof_pitcher_pitchmix_n"}
    ]
    frames = [bin_rate_feature(df, c) for c in rate_cols]
    out = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    save_table("asof_rate_decile_target_means.csv", out)
    return out


def cold_start_analysis(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for col in ["asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"]:
        tmp = df[[col, TARGET]].copy()
        tmp["count_bin"] = pd.cut(tmp[col], bins=COUNT_BINS, labels=COUNT_LABELS)
        g = (
            tmp.groupby("count_bin", observed=True)[TARGET]
            .agg(n="size", target_mean="mean")
            .reset_index()
        )
        g["column"] = col
        frames.append(g[["column", "count_bin", "n", "target_mean"]])
    out = pd.concat(frames, ignore_index=True)
    save_table("cold_start_count_bins.csv", out)
    return out


def year_to_year_pitcher_stability(df: pd.DataFrame) -> pd.DataFrame:
    ps = (
        df.groupby(["pitcher_id", "season"], observed=True)[TARGET]
        .agg(n="size", rate="mean")
        .reset_index()
    )
    ps = ps[ps["n"] >= 100].copy()
    rows = []
    for season in sorted(ps["season"].unique()):
        nxt = season + 1
        a = ps[ps["season"] == season][["pitcher_id", "rate", "n"]].rename(columns={"rate": "rate_y", "n": "n_y"})
        b = ps[ps["season"] == nxt][["pitcher_id", "rate", "n"]].rename(columns={"rate": "rate_next", "n": "n_next"})
        m = a.merge(b, on="pitcher_id")
        if len(m) >= 20:
            rows.append(
                {
                    "season": int(season),
                    "next_season": int(nxt),
                    "pitchers": int(len(m)),
                    "corr_rate_to_next": float(m["rate_y"].corr(m["rate_next"])),
                    "mean_rate_y": float(m["rate_y"].mean()),
                    "mean_rate_next": float(m["rate_next"].mean()),
                }
            )
    out = pd.DataFrame(rows)
    save_table("pitcher_year_to_year_stability.csv", out)
    return out


def trackman_eda() -> dict:
    tm_path = DATA / "trackman_history.csv"
    tm = pd.read_csv(tm_path, low_memory=False, encoding="utf-8-sig")
    tm["count_state"] = tm["balls_before"].astype(str) + "-" + tm["strikes_before"].astype(str)
    tm["inning_phase"] = pd.cut(
        tm["inning"],
        bins=[-np.inf, 3, 6, 9, np.inf],
        labels=["early_1_3", "mid_4_6", "late_7_9", "extra_10p"],
    ).astype(str)

    contract = {
        "rows": int(len(tm)),
        "columns": int(len(tm.columns)),
        "seasons": sorted([int(x) for x in tm["season"].dropna().unique()]),
        "columns_list": list(tm.columns),
    }

    miss = (
        tm.isna()
        .sum()
        .rename("missing")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    miss["missing_rate"] = miss["missing"] / len(tm)
    miss = miss.sort_values("missing", ascending=False)
    save_table("trackman_missing.csv", miss)

    pitch_group = (
        tm.groupby(["season", "pitch_type_group"], dropna=False, observed=True)
        .agg(n=("trackman_id", "size"))
        .reset_index()
    )
    pitch_group["share_in_season"] = pitch_group["n"] / pitch_group.groupby("season")["n"].transform("sum")
    save_table("trackman_pitch_type_group_by_season.csv", pitch_group)

    phys_cols = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side", "zone_speed"]
    phys = (
        tm.groupby(["season", "pitch_type_group"], dropna=False, observed=True)[phys_cols]
        .agg(["count", "mean", "std"])
    )
    phys.columns = ["_".join(c) for c in phys.columns]
    phys = phys.reset_index()
    save_table("trackman_physics_by_season_pitch_group.csv", phys)

    policy = (
        tm.groupby(["count_state", "pitcher_hand", "batter_hand", "inning_phase", "pitch_type_group"], dropna=False, observed=True)
        .agg(n=("trackman_id", "size"))
        .reset_index()
    )
    total = policy.groupby(["count_state", "pitcher_hand", "batter_hand", "inning_phase"])["n"].transform("sum")
    policy["share"] = policy["n"] / total
    policy = policy.sort_values(["count_state", "pitcher_hand", "batter_hand", "inning_phase", "share"], ascending=[True, True, True, True, False])
    save_table("trackman_context_pitch_policy.csv", policy)

    return contract


def main() -> None:
    t0 = time.time()
    ensure_out()
    print(f"OUT={OUT}", flush=True)

    file_contract = {}
    for name in ["train.csv", "test.csv", "sample_submission.csv", "trackman_history.csv", "../data_description.md"]:
        p = (DATA / name).resolve() if name.endswith(".csv") else (ROOT / "data_description.md")
        file_contract[p.name] = {"path": str(p), "size_bytes": int(p.stat().st_size)}

    train = pd.read_csv(DATA / "train.csv", low_memory=False, encoding="utf-8-sig")
    test_sample = pd.read_csv(DATA / "test.csv", low_memory=False, encoding="utf-8-sig")
    sample_sub = pd.read_csv(DATA / "sample_submission.csv", low_memory=False, encoding="utf-8-sig")

    input_cols = [c for c in train.columns if c != TARGET]
    contract = {
        "output": str(OUT),
        "files": file_contract,
        "train_rows": int(len(train)),
        "train_n_columns": int(len(train.columns)),
        "test_sample_rows": int(len(test_sample)),
        "test_sample_n_columns": int(len(test_sample.columns)),
        "sample_submission_rows": int(len(sample_sub)),
        "sample_submission_n_columns": int(len(sample_sub.columns)),
        "target": TARGET,
        "train_row_id_unique": bool(train[ID_COL].is_unique),
        "test_sample_row_id_unique": bool(test_sample[ID_COL].is_unique),
        "input_columns_match_test_sample": input_cols == list(test_sample.columns),
        "sample_submission_columns": list(sample_sub.columns),
        "train_columns": list(train.columns),
        "test_sample_columns": list(test_sample.columns),
        "note": "test.csv and sample_submission.csv in the distributed data are 5-row schema samples, not hidden 2025 evaluation data.",
    }
    write_json("stage0_contract.json", contract)

    train = add_basic_features(train)
    global_mean = float(train[TARGET].mean())
    target_summary = {
        "target_mean": global_mean,
        "positives": int(train[TARGET].sum()),
        "negatives": int((1 - train[TARGET]).sum()),
        "constant_brier_at_global_mean": float(np.mean((train[TARGET] - global_mean) ** 2)),
    }
    write_json("target_summary.json", target_summary)

    season = summarize_target(train, ["season"], "target_by_season.csv")
    summarize_target(train, ["season", "game_month"], "target_by_season_month.csv")
    summarize_target(train, ["count_state"], "target_by_count_state.csv")
    summarize_target(train, ["base_state"], "target_by_base_state.csv")
    summarize_target(train, ["inning_phase"], "target_by_inning_phase.csv")
    summarize_target(train, ["li_bin"], "target_by_li_bin.csv")
    summarize_target(train, ["score_abs_bin"], "target_by_score_abs_bin.csv")
    summarize_target(train, ["platoon_matchup"], "target_by_platoon_matchup.csv")
    summarize_target(train, ["runner_on_1b", "runner_on_2b", "runner_on_3b"], "target_by_base_occupancy_flags.csv")
    summarize_target(train, ["season", "count_state"], "target_by_season_count_state.csv")

    numeric = numeric_profile(train, input_cols)
    cat_cols = [
        "top_bottom",
        "game_type",
        "base_state",
        "pitcher_hand",
        "batter_hand",
        "pitcher_team_id",
        "batter_team_id",
        "count_state",
        "platoon_matchup",
        "inning_phase",
        "li_bin",
        "score_abs_bin",
    ]
    categorical = categorical_profile(train, cat_cols)
    asof = asof_analysis(train)
    cold = cold_start_analysis(train)
    stability = year_to_year_pitcher_stability(train)

    # Value-range and dependency checks that matter for modeling contracts.
    checks = {
        "asof_pitcher_pitchmix_n_equals_asof_pitcher_n": bool((train["asof_pitcher_pitchmix_n"] == train["asof_pitcher_n"]).all()),
        "home_away_win_expectancy_sum_min": float((train["home_win_expectancy"] + train["away_win_expectancy"]).min()),
        "home_away_win_expectancy_sum_max": float((train["home_win_expectancy"] + train["away_win_expectancy"]).max()),
        "target_values": sorted([int(x) for x in train[TARGET].dropna().unique()]),
    }
    write_json("contract_checks.json", checks)

    # Release train memory before Trackman.
    del train

    trackman_contract = trackman_eda()
    write_json("trackman_contract.json", trackman_contract)

    runtime = {"seconds": round(time.time() - t0, 3)}
    write_json("runtime.json", runtime)

    findings = []
    findings.append("# From Scratch EDA V1 Findings")
    findings.append("")
    findings.append(f"run: `{OUT}`")
    findings.append(f"runtime seconds: `{runtime['seconds']}`")
    findings.append("")
    findings.append("## Data Contract")
    findings.append("")
    findings.append(f"- train: `{contract['train_rows']:,}` rows x `{contract['train_n_columns']}` columns")
    findings.append(f"- test sample: `{contract['test_sample_rows']}` rows x `{contract['test_sample_n_columns']}` columns")
    findings.append(f"- input columns match test sample: `{contract['input_columns_match_test_sample']}`")
    findings.append(f"- target mean: `{target_summary['target_mean']:.6f}`")
    findings.append(f"- constant Brier at global mean: `{target_summary['constant_brier_at_global_mean']:.9f}`")
    findings.append("")
    findings.append("## Season Drift")
    findings.append("")
    findings.append(md_table(season.sort_values("season"), 10))
    findings.append("")
    findings.append("## Strongest Numeric Single-Column Signals")
    findings.append("")
    findings.append(md_table(numeric[["column", "missing_rate", "mean", "std", "pearson_target"]].head(20), 20))
    findings.append("")
    findings.append("## Count State Target Means")
    findings.append("")
    cnt = pd.read_csv(OUT / "tables/target_by_count_state.csv")
    findings.append(md_table(cnt.sort_values("target_mean"), 20))
    findings.append("")
    findings.append("## Cold Start Count Bins")
    findings.append("")
    findings.append(md_table(cold, 40))
    findings.append("")
    findings.append("## Pitcher Year-To-Year Stability")
    findings.append("")
    findings.append(md_table(stability, 10))
    findings.append("")
    findings.append("## Trackman Contract")
    findings.append("")
    findings.append(f"- rows: `{trackman_contract['rows']:,}`")
    findings.append(f"- seasons: `{trackman_contract['seasons']}`")
    findings.append("- Trackman is analyzed as historical context/policy data, not as a 1:1 join target table.")
    findings.append("")
    findings.append("## Generated Tables")
    findings.append("")
    for p in sorted((OUT / "tables").glob("*.csv")):
        findings.append(f"- `tables/{p.name}`")
    findings.append("")
    findings.append("## Rule Notes")
    findings.append("")
    findings.append("- Hidden 2025 test data was not used.")
    findings.append("- Distributed 5-row `test.csv` was used only for schema contract checks.")
    findings.append("- No test-row aggregation or test distribution feature was created.")
    findings.append("- Future deployment artifacts must be fit with season-forward cutoffs.")
    (OUT / "findings.md").write_text("\n".join(findings), encoding="utf-8")

    print(f"OUT={OUT}", flush=True)
    print(f"runtime_seconds={runtime['seconds']}", flush=True)


if __name__ == "__main__":
    main()
