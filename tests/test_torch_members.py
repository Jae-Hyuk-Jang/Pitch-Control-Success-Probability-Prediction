"""Neural members (asymmetric M1, DL D4) with random weights: shapes, ranges, and batch invariance."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from pitch_control import asym_m1_predictor as m1  # noqa: E402
from pitch_control import dl_convex_predictor as dl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_embedding_dimensions_follow_the_cardinality_rules():
    assert m1.embedding_dimension(5, "pitcher_id") == 6            # floor at 6
    assert m1.embedding_dimension(10_000, "pitcher_id") == 24      # cap at 24
    assert m1.embedding_dimension(10_000, "batter_id") == 16
    assert m1.embedding_dimension(400, "pitcher_team_id") == 8
    assert m1.embedding_dimension(3, "base_state") == 2


def test_m1_network_outputs_probabilities_and_auxiliary_heads():
    torch.manual_seed(0)
    net = m1.ControlSuccessNet(numeric_count=7, cardinalities=[11, 5], embedding_dims=[4, 3]).eval()
    numeric = torch.randn(32, 7)
    categorical = torch.stack([torch.randint(0, 11, (32,)), torch.randint(0, 5, (32,))], dim=1)
    with torch.no_grad():
        p, aux = net(numeric, categorical)
        alone = torch.stack([net(numeric[i:i + 1], categorical[i:i + 1])[0][0] for i in range(32)])
    assert p.shape == (32,) and aux.shape == (32, 3)
    assert ((p > 0) & (p < 1)).all() and ((aux > 0) & (aux < 1)).all()
    assert torch.allclose(p, alone, atol=1e-6)                     # eval mode: no batch statistics


def test_m1_feature_transform_is_row_local_and_clips_outliers():
    manifest = {
        "numeric_names": ["x1", "x2"], "categorical_names": ["team"],
        "numeric_medians": {"x1": 1.0, "x2": 0.0}, "numeric_means": {"x1": 1.0, "x2": 0.0},
        "numeric_scales": {"x1": 2.0, "x2": 1.0}, "categorical_vocabularies": {"team": ["a", "b"]},
    }
    frame = pd.DataFrame({"x1": [np.nan, 1000.0, 3.0], "x2": [0.5, -1000.0, np.inf], "team": ["a", "z", None]})
    numeric, categorical = m1.transform_features(frame, manifest)
    assert numeric[0, 0] == 0.0                                    # missing -> training median -> standardised 0
    assert numeric[1, 0] == 12.0 and numeric[1, 1] == -12.0        # clipped
    assert categorical[:, 0].tolist() == [1, 0, 0]                 # unseen category and missing map to 0
    again, _ = m1.transform_features(frame.iloc[[1]], manifest)
    np.testing.assert_array_equal(again, numeric[[1]])


def test_dl_member_is_batch_composition_independent_in_float64():
    torch.manual_seed(1)
    net = dl.DLNet({"pitcher_id": 20, "team": 5}, n_num=6, width=(16, 8)).double().eval()
    xnum = torch.randn(64, 6, dtype=torch.float64)
    xcat = torch.stack([torch.randint(0, 21, (64,)), torch.randint(0, 6, (64,))], dim=1)
    with torch.no_grad():
        full = torch.sigmoid(net(xnum, xcat))
        single = torch.stack([torch.sigmoid(net(xnum[i:i + 1], xcat[i:i + 1]))[0] for i in range(64)])
        reversed_ = torch.sigmoid(net(xnum.flip(0), xcat.flip(0))).flip(0)
    assert float((full - single).abs().max()) < 1e-12
    assert float((full - reversed_).abs().max()) < 1e-12


def test_integer_ids_are_keyed_without_decimals():
    assert dl._key_str(pd.Series([1.0, 2.0, 30.0])).tolist() == ["1", "2", "30"]
    assert dl._key_str(pd.Series(["a", "b"])).tolist() == ["a", "b"]
    assert dl._key_str(pd.Series([1.5, 2.0])).tolist() == ["1.5", "2.0"]


def test_command_line_entry_point_loads():
    done = subprocess.run([sys.executable, "-m", "pitch_control", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "--model-dir" in done.stdout
