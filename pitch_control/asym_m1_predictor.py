from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .asym_m1_features import build_deployment_features, key_series, load_history_tables


def embedding_dimension(cardinality: int, name: str) -> int:
    if name == "pitcher_id":
        return min(24, max(6, int(math.ceil(math.sqrt(cardinality)))))
    if name == "batter_id":
        return min(16, max(6, int(math.ceil(math.sqrt(cardinality)))))
    if name in {"pitcher_team_id", "batter_team_id"}:
        return min(8, max(3, int(math.ceil(math.sqrt(cardinality)))))
    return min(8, max(2, int(math.ceil(math.sqrt(cardinality)))))


class ControlSuccessNet(nn.Module):
    def __init__(
        self,
        numeric_count: int,
        cardinalities: list[int],
        embedding_dims: list[int],
        hidden: tuple[int, int] = (256, 128),
        dropout: float = 0.10,
        use_auxiliary: bool = True,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, dimension)
                for cardinality, dimension in zip(cardinalities, embedding_dims)
            ]
        )
        input_dimension = int(numeric_count) + sum(embedding_dims)
        self.trunk = nn.Sequential(
            nn.Linear(input_dimension, hidden[0]),
            nn.SiLU(),
            nn.LayerNorm(hidden[0]),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.SiLU(),
            nn.LayerNorm(hidden[1]),
            nn.Dropout(dropout),
        )
        self.shared_head = nn.Linear(hidden[1], 1)
        self.auxiliary_shared = nn.Linear(hidden[1], 3) if use_auxiliary else None

    def forward(
        self, numeric: torch.Tensor, categorical: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        embedded = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        hidden = self.trunk(torch.cat([numeric, *embedded], dim=1))
        probability = torch.sigmoid(self.shared_head(hidden).squeeze(1))
        auxiliary = (
            torch.sigmoid(self.auxiliary_shared(hidden))
            if self.auxiliary_shared is not None
            else None
        )
        return probability, auxiliary


def transform_features(
    frame: pd.DataFrame, manifest: dict
) -> tuple[np.ndarray, np.ndarray]:
    numeric_names = list(manifest["numeric_names"])
    categorical_names = list(manifest["categorical_names"])
    missing = sorted(set(numeric_names + categorical_names) - set(frame.columns))
    if missing:
        raise ValueError(f"M1 feature columns missing: {missing}")

    numeric_frame = frame[numeric_names].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    medians = pd.Series(manifest["numeric_medians"], dtype=float).reindex(numeric_names)
    means = pd.Series(manifest["numeric_means"], dtype=float).reindex(numeric_names)
    scales = pd.Series(manifest["numeric_scales"], dtype=float).reindex(numeric_names)
    numeric = np.clip(
        ((numeric_frame.fillna(medians) - means) / scales).to_numpy("float32"),
        -12.0,
        12.0,
    )

    categorical = np.zeros((len(frame), len(categorical_names)), dtype=np.int64)
    vocabularies = manifest["categorical_vocabularies"]
    for index, column in enumerate(categorical_names):
        values = key_series(frame, column)
        vocabulary = list(vocabularies[column])
        mapping = pd.Series(
            np.arange(1, len(vocabulary) + 1, dtype=np.int64), index=vocabulary
        )
        categorical[:, index] = values.map(mapping).fillna(0).to_numpy(np.int64)
    return numeric, categorical


class AsymmetricM1Predictor:
    def __init__(
        self,
        model_dir: Path,
        device: str | None = None,
        batch_size: int = 16384,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.manifest = json.loads(
            (self.model_dir / "asym_m1_manifest.json").read_text(encoding="utf-8")
        )
        self.tables = load_history_tables(self.model_dir)
        selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(selected)
        self.batch_size = int(batch_size)
        cardinalities = [int(value) for value in self.manifest["cardinalities"]]
        embedding_dims = [int(value) for value in self.manifest["embedding_dims"]]
        hidden = tuple(int(value) for value in self.manifest["hidden"])
        self.models: list[ControlSuccessNet] = []
        for seed in self.manifest["seeds"]:
            model = ControlSuccessNet(
                numeric_count=len(self.manifest["numeric_names"]),
                cardinalities=cardinalities,
                embedding_dims=embedding_dims,
                hidden=hidden,
                dropout=float(self.manifest["dropout"]),
                use_auxiliary=True,
            ).to(self.device)
            state = torch.load(
                self.model_dir / f"asym_m1_seed{int(seed)}.pt",
                map_location=self.device,
                weights_only=True,
            )
            model.load_state_dict(state, strict=True)
            model.eval()
            self.models.append(model)

    def predict(self, raw: pd.DataFrame) -> np.ndarray:
        if not len(raw):
            return np.empty(0, dtype=float)
        features = build_deployment_features(raw.reset_index(drop=True), self.tables)
        numeric, categorical = transform_features(features, self.manifest)
        dataset = TensorDataset(torch.from_numpy(numeric), torch.from_numpy(categorical))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        seed_predictions: list[np.ndarray] = []
        with torch.no_grad():
            for model in self.models:
                parts: list[np.ndarray] = []
                for numeric_batch, categorical_batch in loader:
                    probability, _ = model(
                        numeric_batch.to(self.device, non_blocking=True),
                        categorical_batch.to(self.device, non_blocking=True),
                    )
                    parts.append(probability.cpu().numpy())
                seed_predictions.append(np.concatenate(parts))
        prediction = np.mean(np.stack(seed_predictions), axis=0)
        if prediction.shape != (len(raw),) or not np.isfinite(prediction).all():
            raise RuntimeError("invalid M1 prediction")
        return np.clip(prediction.astype(float), 1.0e-6, 1.0 - 1.0e-6)
