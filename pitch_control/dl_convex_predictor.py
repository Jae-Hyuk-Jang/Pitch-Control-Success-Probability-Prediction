"""Row-local deployable DL member (D4): embeddings + MLP on official row columns plus explicit season-to-date success-rate
state reconstructed from the row's own career as-of values and a frozen train-derived (through-2024) table.
Inference is stateless per row: no other evaluator row participates."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class DLNet(nn.Module):
    def __init__(self, cat_sizes: dict, n_num: int, width=(512, 256, 128), emb_id: int = 16, emb_other: int = 4) -> None:
        super().__init__()
        dims = {c: (emb_id if c in ("pitcher_id", "batter_id") else emb_other) for c in cat_sizes}
        self.embs = nn.ModuleDict({c: nn.Embedding(sz + 1, dims[c]) for c, sz in cat_sizes.items()})
        layers, d = [], n_num + sum(dims.values())
        for wdt in width:
            layers += [nn.Linear(d, wdt), nn.SiLU(), nn.Dropout(0.10)]; d = wdt
        self.trunk = nn.Sequential(*layers[:-1])
        self.head = nn.Linear(d, 1)

    def forward(self, xnum: torch.Tensor, xcat: torch.Tensor) -> torch.Tensor:
        h = torch.cat([xnum] + [self.embs[c](xcat[:, i]) for i, c in enumerate(self.embs)], dim=1)
        return self.head(self.trunk(h)).squeeze(1)


def _key_str(s: pd.Series) -> pd.Series:
    """canonical categorical key: integers render without decimals (matches training's astype(str) on int64 columns)."""
    v = pd.to_numeric(s, errors="coerce")
    if v.notna().all() and np.all(np.mod(v.to_numpy(float), 1.0) == 0.0):
        return v.astype("int64").astype(str)
    return s.astype(str)


class DLConvexPredictor:
    def __init__(self, model_dir: Path, threads: int = 4) -> None:
        d = Path(model_dir) / "dl"
        self.meta = json.loads((d / "dl_meta.json").read_text(encoding="utf-8"))
        self.numeric = list(self.meta["numeric"]); self.cat = list(self.meta["cat"])
        self.mu = pd.Series(self.meta["mu"]).reindex(self.numeric).astype(float).fillna(0.0)
        self.sd = pd.Series(self.meta["sd"]).reindex(self.numeric).astype(float).replace(0.0, 1.0).fillna(1.0)
        self.vocab = {c: dict(self.meta["vocab"][c]) for c in self.cat}
        self.tables = {}
        for tag, fn, idc in (("P", "frozen_pitcher_2025.csv", "pitcher_id"), ("B", "frozen_batter_2025.csv", "batter_id")):
            t = pd.read_csv(d / fn, low_memory=False)
            key = _key_str(t[idc])
            self.tables[tag] = (pd.Series(t["n"].to_numpy(float), index=key.to_numpy()), pd.Series(t["s"].to_numpy(float), index=key.to_numpy()))
        torch.set_num_threads(max(1, int(threads)))
        self.device = torch.device("cpu")
        self.models = []
        for seed in self.meta["seeds"]:
            m = DLNet({c: int(self.meta["cat_sizes"][c]) for c in self.cat}, len(self.numeric), tuple(self.meta.get("width", (512, 256, 128))), int(self.meta.get("emb_id", 16)), int(self.meta.get("emb_other", 4)))
            m.load_state_dict(torch.load(d / f"dl_model_seed{int(seed)}.pt", map_location=self.device))
            m.double().eval(); self.models.append(m)  # float64 inference: batch-composition-independent to ~1e-15

    def _state(self, frame: pd.DataFrame) -> pd.DataFrame:
        d = pd.DataFrame(index=frame.index)
        for tag, idc, ncol, rcol in (("P", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"), ("B", "batter_id", "asof_batter_n", "asof_batter_success_rate")):
            tn, ts = self.tables[tag]; key = _key_str(frame[idc])
            n_pr = key.map(tn).fillna(0.0).to_numpy(float); s_pr = key.map(ts).fillna(0.0).to_numpy(float)
            n_all = pd.to_numeric(frame[ncol], errors="coerce").fillna(0.0).to_numpy(float)
            r_all = pd.to_numeric(frame[rcol], errors="coerce").to_numpy(float); n_se = n_all - n_pr
            r_pr = np.where(n_pr > 0, s_pr / np.where(n_pr > 0, n_pr, 1.0), np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                r_se = np.where(n_se > 0, (np.nan_to_num(r_all) * n_all - s_pr) / np.where(n_se > 0, n_se, 1.0), np.nan)
            d[f"{tag}_prior_n_log"] = np.log1p(np.maximum(n_pr, 0)); d[f"{tag}_season_n_log"] = np.log1p(np.maximum(n_se, 0))
            d[f"{tag}_season_rel"] = np.maximum(n_se, 0) / (np.maximum(n_se, 0) + 1000.0); d[f"{tag}_has_prior"] = (n_pr > 0).astype(float)
            d[f"X_{tag}_prior"] = r_pr; d[f"X_{tag}_season"] = r_se
            d[f"X_{tag}_dev"] = np.where((n_pr > 0) & (n_se > 0) & np.isfinite(r_all), r_se - r_pr, np.nan)
        return d

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if len(frame) == 0:
            return np.zeros(0, dtype=float)
        st = self._state(frame)
        cols = {}
        for c in self.numeric:
            cols[c] = st[c].to_numpy(float) if c in st.columns else pd.to_numeric(frame[c], errors="coerce").to_numpy(float)
        xn = pd.DataFrame(cols, index=frame.index)[self.numeric]
        xn = ((xn - self.mu) / self.sd).fillna(0.0).to_numpy(np.float64)
        cat_cols = []
        for c in self.cat:
            if c == "count_cat":
                s = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(0).astype(int).astype(str) + "-" + pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(0).astype(int).astype(str)
            else:
                s = _key_str(frame[c])
            cat_cols.append(s.map(self.vocab[c]).fillna(0).to_numpy(np.int64))
        xc = np.column_stack(cat_cols)
        out = np.zeros(len(frame), dtype=float)
        with torch.no_grad():
            for i in range(0, len(frame), 65536):
                a = torch.tensor(xn[i:i + 65536]); b = torch.tensor(xc[i:i + 65536])
                out[i:i + 65536] = np.mean([torch.sigmoid(m(a, b)).numpy().astype(np.float64) for m in self.models], axis=0)
        return np.clip(out, 1e-6, 1 - 1e-6)
