"""Fixed row-local logit terms applied to R rows.

logit(p') = logit(p) + scale * sum_j gamma_j * x_j(row)
No other evaluation row participates; the only table consulted is a frozen train-derived membership / prior table."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .asym_m1_features import key_series as _tail_key_series, _map_index as _tail_map_index


def _tail_cold_mask(frame: pd.DataFrame, tables) -> np.ndarray:
    """cold = pitcher absent from the frozen train-history table (same rule as the M1 feature hist_known == 0)."""
    all_n = _tail_map_index(tables.pitcher_all, "n", _tail_key_series(frame, "pitcher_id"))
    return all_n <= 0.0


def _tail_prior_rate(frame: pd.DataFrame, model_dir) -> np.ndarray:
    """Frozen preseason (through-2024) success rate per pitcher from the anchor's own current-state artifact; NaN if absent/n<=0."""
    tab = pd.read_csv(model_dir / "current_state_preseason_2025.csv", low_memory=False)
    tab["pitcher_id"] = tab["pitcher_id"].astype("object").where(tab["pitcher_id"].notna(), "__NA__").astype(str)
    ok = pd.to_numeric(tab["preseason_n"], errors="coerce").fillna(0.0) > 0.0
    rate = pd.Series(pd.to_numeric(tab["pre_success_rate"], errors="coerce").to_numpy(dtype=float), index=tab["pitcher_id"].to_numpy())[ok.to_numpy()]
    return _tail_key_series(frame, "pitcher_id").map(rate).to_numpy(dtype=float)


def _term_value(frame: pd.DataFrame, term: dict, cold: np.ndarray, prior: np.ndarray = None) -> np.ndarray:
    """Row-local design value for one term. Uses only the row's own official columns
    plus the frozen train-history membership flag (cold/known) already used by the anchor's M1 features."""
    value = _term_core(frame, term, prior)
    mask = term.get("mask")
    if mask == "cold":
        value = np.where(cold, value, 0.0)
    elif mask == "known":
        value = np.where(cold, 0.0, value)
    elif mask is not None:
        raise ValueError(f"unknown term mask: {mask}")
    if "n_min" in term or "n_max" in term:
        count_n = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        keep = np.ones(len(count_n), dtype=bool)
        if "n_min" in term:
            keep &= count_n >= float(term["n_min"])
        if "n_max" in term:
            keep &= count_n < float(term["n_max"])
        value = np.where(keep, value, 0.0)
    return value


def _term_core(frame: pd.DataFrame, term: dict, prior: np.ndarray = None) -> np.ndarray:
    kind = term["type"]
    if kind == "asof_minus_prior":
        value = pd.to_numeric(frame[term["column"]], errors="coerce").to_numpy(dtype=float)
        count_n = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ok = np.isfinite(value) & (count_n > 0.0) & np.isfinite(prior)
        return np.where(ok, np.nan_to_num(value) - np.nan_to_num(prior), 0.0)
    if kind == "count":
        balls = pd.to_numeric(frame["balls_before"], errors="coerce").to_numpy(dtype=float)
        strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").to_numpy(dtype=float)
        return ((balls == float(term["balls"])) & (strikes == float(term["strikes"]))).astype(float)
    value = pd.to_numeric(frame[term["column"]], errors="coerce").to_numpy(dtype=float)
    known = np.isfinite(value)
    if term.get("requires_asof_n", True):
        count_n = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        known = known & (count_n > 0.0)
    if kind == "linear":
        return np.where(known, np.nan_to_num(value) - float(term["center"]), 0.0)
    if kind == "hinge":
        return np.where(known, np.maximum(np.nan_to_num(value) - float(term["threshold"]), 0.0), 0.0)
    raise ValueError(f"unknown term type: {kind}")


def apply_tail_terms(final: np.ndarray, frame: pd.DataFrame, is_r: np.ndarray, spec: dict, cold: np.ndarray, prior: np.ndarray = None) -> np.ndarray:
    """Row-local logit shift on R rows: logit(p') = logit(p) + scale * sum(gamma_j * x_j)."""
    scale = float(spec["scale"])
    shift = np.zeros(len(frame), dtype=float)
    for term in spec["terms"]:
        shift += float(term["gamma"]) * _term_value(frame, term, cold, prior)
    p = np.clip(final[is_r], 1.0e-6, 1.0 - 1.0e-6)
    z = np.log(p / (1.0 - p)) + scale * shift[is_r]
    out = np.array(final, dtype=float, copy=True)
    out[is_r] = 1.0 / (1.0 + np.exp(-z))
    f_intercept = float(spec.get("f_intercept", 0.0))
    if f_intercept != 0.0:
        # constant logit shift on F (non-R) rows: logit(p') = logit(p) + scale * f_intercept
        is_f = ~is_r
        pf = np.clip(final[is_f], 1.0e-6, 1.0 - 1.0e-6)
        out[is_f] = 1.0 / (1.0 + np.exp(-(np.log(pf / (1.0 - pf)) + scale * f_intercept)))
    return out

