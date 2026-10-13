"""Command line entry point: python -m pitch_control --model-dir model --data-dir data --output-dir output

Reads <data-dir>/test.csv and <data-dir>/sample_submission.csv, writes <output-dir>/submission.csv.
The frozen artifacts in --model-dir are produced by the training stages and are not part of this repository.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .predictor import Predictor, ROW_ID, TARGET


def run(model_dir: Path, data_dir: Path, output_dir: Path, threads: int = 6) -> Path:
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    if not test_path.is_file() or not sample_path.is_file():
        raise FileNotFoundError("data/test.csv or data/sample_submission.csv missing")
    test = pd.read_csv(test_path, encoding="utf-8-sig", low_memory=False)
    sample = pd.read_csv(sample_path, encoding="utf-8-sig", low_memory=False)
    if list(sample.columns) != [ROW_ID, TARGET]:
        raise ValueError(f"unexpected sample columns: {list(sample.columns)}")
    if test[ROW_ID].isna().any() or test[ROW_ID].astype(str).duplicated().any():
        raise ValueError("test row_id must be unique and non-null")
    if sample[ROW_ID].isna().any() or sample[ROW_ID].astype(str).duplicated().any():
        raise ValueError("sample row_id must be unique and non-null")
    if len(test) != len(sample):
        raise ValueError("test/sample row count mismatch")

    prediction = Predictor(model_dir, threads=threads).predict(test)
    mapping = pd.Series(prediction, index=test[ROW_ID].astype(str))
    aligned = sample[ROW_ID].astype(str).map(mapping)
    if aligned.isna().any():
        raise RuntimeError("sample/test row_id mismatch")
    output = sample[[ROW_ID, TARGET]].copy()
    output[TARGET] = aligned.to_numpy(float)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "submission.csv"
    output.to_csv(target, index=False, encoding="utf-8")
    print(f"saved={target} rows={len(output)}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Row-independent control-success inference")
    parser.add_argument("--model-dir", type=Path, default=Path("model"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    run(args.model_dir, args.data_dir, args.output_dir, args.threads)


if __name__ == "__main__":
    main()
