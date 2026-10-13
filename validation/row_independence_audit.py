"""Row-independence audit for any ``predict(frame) -> 1-D array`` function.

The competition required that every evaluation row be predicted independently: the value for row A may not change
with the other rows present in the batch. Before a candidate was packaged it had to pass all of these checks:

  single_row                 sampled rows predicted one at a time == the same rows inside the full batch
  chunks                     the full batch evaluated in several chunk sizes
  permutation                a shuffled batch, restored to the original order
  duplicates                 sampled rows appended again; both copies must equal the original
  unrelated_rows_scrambled   every field of *all other* rows replaced by draws from a shifted distribution
                             (upper half of each numeric column, modal value of each categorical one), so
                             batch means, spreads and frequencies change drastically
  repeat                     the same batch twice in a row (no state change between calls)

Each entry is the maximum absolute difference in the predicted probability. ``passed`` is true when all of them are
within ``tol`` (0.0 for the CatBoost-only graph; ~1e-6 once torch float32 members are involved).

Usage as a library::

    from validation.row_independence_audit import audit
    report = audit(predictor.predict, pseudo_test_frame, tol=1e-6)

Command line (needs the frozen artifacts, which are not part of this repository)::

    python -m validation.row_independence_audit --model-dir model --frame pseudo_test.csv --rows 2000
"""
from __future__ import annotations

import argparse
import json
from typing import Callable, Sequence

import numpy as np
import pandas as pd

Predict = Callable[[pd.DataFrame], Sequence[float]]


def _as_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _max_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Largest absolute difference; a non-finite prediction on either side counts as an infinite difference."""
    diff = np.abs(a - b)
    return float(np.max(np.where(np.isfinite(a) & np.isfinite(b), diff, np.inf)))


def audit(
    predict: Predict,
    frame: pd.DataFrame,
    n_probe: int = 64,
    chunk_sizes: Sequence[int] = (1, 37, 1024),
    seed: int = 0,
    tol: float = 1e-6,
) -> dict:
    rng = np.random.default_rng(seed)
    frame = frame.reset_index(drop=True)
    n = len(frame)
    if n < 3:
        raise ValueError("need at least three rows to audit")
    base = _as_array(predict(frame))
    if base.shape != (n,):
        raise ValueError(f"predict returned shape {base.shape}, expected {(n,)}")
    probe = np.sort(rng.choice(n, size=min(n_probe, n), replace=False))
    report: dict = {"rows": int(n), "probe_rows": int(len(probe))}

    single = np.array([_as_array(predict(frame.iloc[[i]].reset_index(drop=True)))[0] for i in probe])
    report["single_row"] = _max_diff(single, base[probe])

    worst = 0.0
    for size in chunk_sizes:
        parts = [_as_array(predict(frame.iloc[s:s + size].reset_index(drop=True))) for s in range(0, n, int(size))]
        worst = max(worst, _max_diff(np.concatenate(parts), base))
    report["chunks"] = worst

    perm = rng.permutation(n)
    shuffled = _as_array(predict(frame.iloc[perm].reset_index(drop=True)))
    restored = np.empty(n)
    restored[perm] = shuffled
    report["permutation"] = _max_diff(restored, base)

    doubled = _as_array(predict(pd.concat([frame, frame.iloc[probe]], ignore_index=True)))
    report["duplicates"] = max(_max_diff(doubled[:n], base), _max_diff(doubled[n:], base[probe]))

    others = np.setdiff1d(np.arange(n), probe)
    scrambled = frame.copy()
    for column in frame.columns:
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            pool = values[values >= values.median()].to_numpy()       # upper half: shifts mean and spread
        else:
            pool = values.mode().to_numpy()                            # modal value: collapses the category mix
        if len(pool):
            scrambled.loc[others, column] = rng.choice(pool, size=len(others))
    after = _as_array(predict(scrambled))
    report["unrelated_rows_scrambled"] = _max_diff(after[probe], base[probe])

    report["repeat"] = _max_diff(_as_array(predict(frame)), base)
    checks = ["single_row", "chunks", "permutation", "duplicates", "unrelated_rows_scrambled", "repeat"]
    report["max_abs_diff"] = max(report[name] for name in checks)
    report["tol"] = float(tol)
    report["passed"] = bool(report["max_abs_diff"] <= tol)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Row-independence audit of the packaged predictor")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--frame", required=True, help="CSV with the official input columns (no labels needed)")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    from pitch_control.predictor import Predictor  # imported late: needs torch and the frozen artifacts

    frame = pd.read_csv(args.frame, encoding="utf-8-sig", low_memory=False).head(args.rows)
    predictor = Predictor(args.model_dir, threads=args.threads)
    report = audit(predictor.predict, frame, tol=args.tol)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
