"""Echo State Network + MLP readout for multi-horizon TTC prediction.

The reservoir is a fixed, sparse random recurrent layer with leaky-integrator
units; only the MLP readout is trained, which keeps training cheap and the
model small enough for the Kria's ARM cores. The readout sees [features, state]
and predicts log(TTC) at each horizon (log makes the loss a relative error).
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class EsnConfig:
    reservoir_size: int = 300
    spectral_radius: float = 0.9
    leak_rate: float = 0.3
    input_scale: float = 0.5
    density: float = 0.1
    hidden: int = 64
    dropout: float = 0.1
    seed: int = 0


class Reservoir:
    def __init__(self, n_inputs: int, cfg: EsnConfig):
        rng = np.random.default_rng(cfg.seed)
        n = cfg.reservoir_size
        w = rng.uniform(-1, 1, (n, n)) * (rng.random((n, n)) < cfg.density)
        w *= cfg.spectral_radius / np.max(np.abs(np.linalg.eigvals(w)))
        self.w = w.astype(np.float32)
        self.w_in = (rng.uniform(-1, 1, (n, n_inputs + 1)) * cfg.input_scale).astype(np.float32)  # +1 bias
        self.leak = cfg.leak_rate

    @property
    def size(self) -> int:
        return self.w.shape[0]

    def initial_state(self) -> np.ndarray:
        return np.zeros(self.size, np.float32)

    def step(self, u: np.ndarray, x: np.ndarray) -> np.ndarray:
        pre = self.w_in @ np.concatenate(([1.0], u)).astype(np.float32) + self.w @ x
        return (1 - self.leak) * x + self.leak * np.tanh(pre)

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """States for a contiguous input sequence [T, n_inputs], starting from rest."""
        x, states = self.initial_state(), np.empty((len(inputs), self.size), np.float32)
        for t, u in enumerate(inputs):
            x = self.step(u, x)
            states[t] = x
        return states


class Readout(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


class EsnTtcModel:
    """Standardiser + reservoir + readout. `use_reservoir=False` gives the memoryless MLP ablation."""

    def __init__(self, n_features: int, horizons_s: tuple[float, ...], cfg: EsnConfig, use_reservoir: bool = True):
        self.cfg, self.horizons_s, self.use_reservoir = cfg, tuple(horizons_s), use_reservoir
        self.n_features = n_features
        self.reservoir = Reservoir(n_features, cfg) if use_reservoir else None
        self.mean = np.zeros(n_features, np.float32)
        self.std = np.ones(n_features, np.float32)
        n_in = n_features + (self.reservoir.size if use_reservoir else 0)
        torch.manual_seed(cfg.seed)
        self.readout = Readout(n_in, len(horizons_s), cfg.hidden, cfg.dropout)

    def fit_scaler(self, features: np.ndarray) -> None:
        self.mean = np.nanmean(features, axis=0).astype(np.float32)
        self.std = (np.nanstd(features, axis=0) + 1e-6).astype(np.float32)

    def normalise(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) / self.std).astype(np.float32)

    def design_matrix(self, segment_features: np.ndarray) -> np.ndarray:
        """Readout inputs for one contiguous segment of raw features [T, F]."""
        u = self.normalise(segment_features)
        return np.hstack([u, self.reservoir.run(u)]) if self.use_reservoir else u

    @torch.no_grad()
    def predict_log_ttc(self, design: np.ndarray) -> np.ndarray:
        self.readout.eval()
        return self.readout(torch.from_numpy(design)).numpy()

    def save(self, path) -> None:
        torch.save({
            "cfg": asdict(self.cfg), "horizons_s": self.horizons_s, "use_reservoir": self.use_reservoir,
            "n_features": self.n_features, "mean": self.mean, "std": self.std,
            "readout": self.readout.state_dict(),
            # Stored explicitly: regenerating from the seed is not bit-stable across numpy builds.
            "reservoir": None if self.reservoir is None else {"w": self.reservoir.w, "w_in": self.reservoir.w_in},
        }, path)

    @classmethod
    def load(cls, path) -> "EsnTtcModel":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(blob["n_features"], blob["horizons_s"], EsnConfig(**blob["cfg"]), blob["use_reservoir"])
        model.mean, model.std = blob["mean"], blob["std"]
        model.readout.load_state_dict(blob["readout"])
        if model.reservoir is not None:
            model.reservoir.w, model.reservoir.w_in = blob["reservoir"]["w"], blob["reservoir"]["w_in"]
        return model


class OnlineTtcPredictor:
    """Per-track streaming inference for the live pipeline (reservoir state per track id)."""

    def __init__(self, model: EsnTtcModel):
        self.model = model
        self.states: dict[int, np.ndarray] = {}

    def update(self, track_id: int, features: np.ndarray) -> np.ndarray:
        """Return predicted TTC (s) at each horizon for this track's newest frame."""
        u = self.model.normalise(features[None])[0]
        if self.model.use_reservoir:
            x = self.model.reservoir.step(u, self.states.get(track_id, self.model.reservoir.initial_state()))
            self.states[track_id] = x
            u = np.concatenate([u, x])
        return np.exp(self.model.predict_log_ttc(u[None].astype(np.float32))[0])

    def forget_missing(self, live_ids) -> None:
        self.states = {tid: s for tid, s in self.states.items() if tid in live_ids}
