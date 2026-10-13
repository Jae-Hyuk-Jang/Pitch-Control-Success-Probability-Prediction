from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


_CACHE = {}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1.0e-6, 1.0 - 1.0e-6)
    return np.log(p / (1.0 - p))


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def _string(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("__MISSING__", index=frame.index, dtype=str)
    return frame[column].astype("object").where(frame[column].notna(), "__MISSING__").astype(str)


def _attach_anchor_state(frame: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    try:
        from .asym_m1_features import build_deployment_features, load_history_tables

        out = build_deployment_features(frame.copy(), load_history_tables(model_dir))
        out["strict_prior_rate"] = pd.to_numeric(out.get("preseason_rate", 0.0), errors="coerce")
        return out
    except Exception:
        out = frame.copy()
        table_path = model_dir / "current_state_preseason_2025.csv"
        if not table_path.exists():
            out["hist_known"] = 0.0
            out["strict_prior_rate"] = np.nan
            out["current_minus_preseason_rate"] = 0.0
            out["current_season_reliability"] = 0.0
            out["current_season_n_log"] = 0.0
            out["state_recent_minus_long"] = 0.0
            return out
        tab = pd.read_csv(table_path, low_memory=False)
        tab["pitcher_id"] = tab["pitcher_id"].astype(str)
        key = _string(out, "pitcher_id")
        prior = pd.Series(pd.to_numeric(tab["pre_success_rate"], errors="coerce").to_numpy(float), index=tab["pitcher_id"])
        prior_n = pd.Series(pd.to_numeric(tab["preseason_n"], errors="coerce").fillna(0.0).to_numpy(float), index=tab["pitcher_id"])
        p0 = key.map(prior).to_numpy(float)
        n0 = key.map(prior_n).fillna(0.0).to_numpy(float)
        asof_n = _numeric(out, "asof_pitcher_n").to_numpy(float)
        rate = pd.to_numeric(out["asof_pitcher_success_rate"], errors="coerce").to_numpy(float)
        current_n = np.maximum(asof_n - n0, 0.0)
        ok = (n0 > 0.0) & (current_n > 0.0) & np.isfinite(rate) & np.isfinite(p0)
        out["hist_known"] = (n0 > 0.0).astype(float)
        out["strict_prior_rate"] = p0
        out["current_minus_preseason_rate"] = np.where(ok, rate - p0, 0.0)
        out["current_season_reliability"] = current_n / (current_n + 100.0)
        out["current_season_n_log"] = np.log1p(current_n)
        out["state_recent_minus_long"] = 0.0
        return out


def _add_factor_features(frame: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    out = _attach_anchor_state(frame, model_dir).copy()
    eps = 1.0e-6

    asof_n = _numeric(out, "asof_pitcher_n")
    asof_rate = pd.to_numeric(out.get("asof_pitcher_success_rate"), errors="coerce")
    prior = pd.to_numeric(out.get("strict_prior_rate"), errors="coerce")
    dev = _numeric(out, "current_minus_preseason_rate")
    rel = _numeric(out, "current_season_reliability")
    current_n_log = _numeric(out, "current_season_n_log")

    out["asof_success_center"] = np.where(np.isfinite(asof_rate), asof_rate - 0.52, 0.0)
    out["asof_n_log"] = np.log1p(np.maximum(asof_n.to_numpy(float), 0.0))
    out["prior_success_center"] = np.where(np.isfinite(prior), prior - 0.52, 0.0)
    out["current_dev"] = dev
    out["current_dev_x_rel"] = dev * rel
    out["current_dev_x_log_n"] = dev * current_n_log
    out["prior_x_rel"] = out["prior_success_center"].astype(float) * rel
    out["dev_z_sqrt_n"] = dev * np.sqrt(np.maximum(np.expm1(current_n_log), 0.0).clip(0, 5000))

    for k in (1, 3, 5):
        prev = pd.to_numeric(out.get(f"asof_pitcher_prev{k}_game_success_rate"), errors="coerce")
        out[f"prev{k}_gap"] = np.where(np.isfinite(prev) & np.isfinite(asof_rate), prev - asof_rate, 0.0)

    mix_n = _numeric(out, "asof_pitcher_pitchmix_n")
    fb = pd.to_numeric(out.get("asof_pitcher_fastball_rate"), errors="coerce").clip(0.0, 1.0)
    br = pd.to_numeric(out.get("asof_pitcher_breaking_rate"), errors="coerce").clip(0.0, 1.0)
    off = pd.to_numeric(out.get("asof_pitcher_offspeed_rate"), errors="coerce").clip(0.0, 1.0)
    known_mix = (mix_n > 0) & np.isfinite(fb) & np.isfinite(br) & np.isfinite(off)
    a = 2.0
    c_fb = np.where(known_mix, mix_n * fb + a, a)
    c_br = np.where(known_mix, mix_n * br + a, a)
    c_off = np.where(known_mix, mix_n * off + a, a)
    denom = c_fb + c_br + c_off
    q_fb, q_br, q_off = c_fb / denom, c_br / denom, c_off / denom
    out["mix_known"] = known_mix.astype(float)
    out["mix_precision_log"] = np.log1p(np.maximum(mix_n.to_numpy(float), 0.0))
    out["mix_fb_vs_nonfb"] = np.log(c_fb / np.sqrt(c_br * c_off))
    out["mix_brk_vs_off"] = np.log(c_br / c_off)
    out["mix_entropy"] = -(q_fb * np.log(q_fb + eps) + q_br * np.log(q_br + eps) + q_off * np.log(q_off + eps))
    out["mix_max_share"] = np.maximum.reduce([q_fb, q_br, q_off])

    balls = _numeric(out, "balls_before", -1.0).astype(int)
    strikes = _numeric(out, "strikes_before", -1.0).astype(int)
    inning = _numeric(out, "inning")
    li = _numeric(out, "li")
    score = _numeric(out, "score_diff_pitcher_team")
    pairs = pd.Series(list(zip(balls, strikes)), index=out.index)
    out["two_strike"] = (strikes == 2).astype(float)
    out["three_ball"] = (balls == 3).astype(float)
    out["full_count"] = ((balls == 3) & (strikes == 2)).astype(float)
    out["pitcher_adv_count"] = pairs.isin([(0, 1), (0, 2), (1, 2)]).astype(float)
    out["batter_adv_count"] = pairs.isin([(2, 0), (3, 0), (3, 1)]).astype(float)
    out["late"] = (inning >= 7).astype(float)
    out["high_li"] = (li >= 1.5).astype(float)
    out["close_game"] = ((score.abs() <= 2) & (inning >= 7)).astype(float)
    out["risp"] = ((_numeric(out, "runner_on_2b").astype(int) == 1) | (_numeric(out, "runner_on_3b").astype(int) == 1)).astype(float)
    out["li_log"] = np.log1p(np.maximum(li.to_numpy(float), 0.0))
    out["score_abs"] = np.abs(score.to_numpy(float))
    out["hand_diff"] = (_string(out, "pitcher_hand") != _string(out, "batter_hand")).astype(float)
    out["team13_involved"] = ((_string(out, "pitcher_team_id") == "13") | (_string(out, "batter_team_id") == "13")).astype(float)

    out["mix_fb_vs_nonfb_x_two_strike"] = out["mix_fb_vs_nonfb"] * out["two_strike"]
    out["mix_brk_vs_off_x_two_strike"] = out["mix_brk_vs_off"] * out["two_strike"]
    out["mix_fb_vs_nonfb_x_hand_diff"] = out["mix_fb_vs_nonfb"] * out["hand_diff"]
    out["mix_brk_vs_off_x_hand_diff"] = out["mix_brk_vs_off"] * out["hand_diff"]
    out["mix_entropy_x_batter_adv"] = out["mix_entropy"] * out["batter_adv_count"]
    out["current_dev_x_two_strike"] = out["current_dev"] * out["two_strike"]
    out["current_dev_x_high_li"] = out["current_dev"] * out["high_li"]
    out["current_dev_x_risp"] = out["current_dev"] * out["risp"]
    out["current_dev_x_mix_fb"] = out["current_dev"] * out["mix_fb_vs_nonfb"]
    out["prior_x_mix_brk_off"] = out["prior_success_center"] * out["mix_brk_vs_off"]
    return out


def _standardize_train_test(train_raw: np.ndarray, test_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(train_raw, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    tr = np.where(np.isfinite(train_raw), train_raw, med)
    te = np.where(np.isfinite(test_raw), test_raw, med)
    mean = tr.mean(axis=0)
    std = tr.std(axis=0, ddof=1)
    std = np.where(np.isfinite(std) & (std > 0.0), std, 1.0)
    return ((tr - mean) / std).astype(np.float32), ((te - mean) / std).astype(np.float32)


def _fit_bundle(model_dir: Path):
    key = str(Path(model_dir).resolve())
    if key in _CACHE:
        return _CACHE[key]

    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    manifest = json.loads((model_dir / "component_aux_corr_manifest.json").read_text(encoding="utf-8"))
    data = np.load(model_dir / "component_aux_train_2024.npz")
    factor_train = data["factor_raw"].astype(np.float32)
    y_comp = {name: data[f"y_{name}"].astype(float) for name in manifest["components"]}
    z = data["z_resid"].astype(np.float32)
    w = data["weight"].astype(np.float32)

    dummy_test = factor_train[:1].copy()
    x_comp_train, _ = _standardize_train_test(factor_train, dummy_test)
    comp_train_pred = {}
    comp_models = {}
    for comp in manifest["components"]:
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.035,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=250,
            l2_regularization=10.0,
            random_state=2025,
            early_stopping=False,
        )
        model.fit(x_comp_train, y_comp[comp])
        comp_models[comp] = model
        comp_train_pred[comp] = model.predict_proba(x_comp_train)[:, 1].astype(np.float32)

    train_shape = _component_shape_frame(comp_train_pred).to_numpy(np.float32)
    x_resid_raw = np.concatenate([factor_train, train_shape], axis=1)
    x_resid_train, _ = _standardize_train_test(x_resid_raw, x_resid_raw[:1].copy())
    resid_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.035,
        max_iter=160,
        max_leaf_nodes=15,
        min_samples_leaf=350,
        l2_regularization=15.0,
        random_state=0,
        early_stopping=False,
    )
    resid_model.fit(x_resid_train, z, sample_weight=w)
    train_pred = resid_model.predict(x_resid_train).astype(float)
    bundle = {
        "manifest": manifest,
        "factor_train": factor_train,
        "comp_models": comp_models,
        "resid_model": resid_model,
        "resid_train_raw": x_resid_raw.astype(np.float32),
        "train_pred_mean": float(train_pred.mean()),
    }
    _CACHE[key] = bundle
    return bundle


def _component_shape_frame(pred: dict[str, np.ndarray]) -> pd.DataFrame:
    arr = np.column_stack([pred["reverse"], pred["middle"], pred["ball"], pred["strike"]])
    return pd.DataFrame(
        {
            "pred_fail_sum": pred["reverse"] + pred["middle"] + pred["ball"],
            "pred_reverse_minus_middle": pred["reverse"] - pred["middle"],
            "pred_ball_minus_strike": pred["ball"] - pred["strike"],
            "pred_execution_spread": np.std(arr, axis=1, ddof=1),
        }
    )


def apply_component_aux_corr(probability: np.ndarray, frame: pd.DataFrame, model_dir: Path) -> np.ndarray:
    model_dir = Path(model_dir)
    manifest_path = model_dir / "component_aux_corr_manifest.json"
    data_path = model_dir / "component_aux_train_2024.npz"
    if not manifest_path.exists() or not data_path.exists():
        return probability
    bundle = _fit_bundle(model_dir)
    manifest = bundle["manifest"]

    features = _add_factor_features(frame, model_dir)
    factor_cols = list(manifest["factor_cols"])
    factor_test = features.reindex(columns=factor_cols).astype(float).to_numpy(np.float32)
    _, x_comp_test = _standardize_train_test(bundle["factor_train"], factor_test)
    comp_test_pred = {
        comp: bundle["comp_models"][comp].predict_proba(x_comp_test)[:, 1].astype(np.float32)
        for comp in manifest["components"]
    }
    test_shape = _component_shape_frame(comp_test_pred).to_numpy(np.float32)
    x_test_raw = np.concatenate([factor_test, test_shape], axis=1)
    _, x_resid_test = _standardize_train_test(bundle["resid_train_raw"], x_test_raw)
    corr = bundle["resid_model"].predict(x_resid_test).astype(float) - float(bundle["train_pred_mean"])

    out = np.asarray(probability, dtype=float).copy()
    z = _logit(out) + float(manifest["scale"]) * corr
    out = _sigmoid(z)
    return np.clip(out, 0.0, 1.0)
