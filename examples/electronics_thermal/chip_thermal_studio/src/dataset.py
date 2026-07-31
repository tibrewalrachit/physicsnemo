# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Design-library storage and model input/output encoding.

The "library of completed designs" is a single NPZ file holding power
maps, package parameters, and solved temperature fields. Normalization
statistics are computed from the library and stored with checkpoints so
the surrogate can be served standalone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from thermal import ChipSpec


#: Gaussian blur scales (in grid cells) for the multi-scale spreading
#: features. They bracket the thermal healing length lambda = sqrt(kt/h)
#: (typically 6-22 cells for the default design space at grid=64).
BLUR_SIGMAS: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0)

#: Number of local input channels: power, T0, blurred T0 per sigma, x, y.
N_LOCAL_FEATURES: int = 2 + len(BLUR_SIGMAS) + 2

#: Number of global features: [kt, h, lambda].
N_GLOBAL_FEATURES: int = 3


@dataclass
class NormStats:
    """Standardization statistics for inputs and outputs."""

    power_mean: float
    power_std: float
    globals_mean: np.ndarray  # (3,) for [kt, h, lambda]
    globals_std: np.ndarray
    temp_mean: float
    temp_std: float

    def to_dict(self) -> dict:
        return {
            "power_mean": self.power_mean,
            "power_std": self.power_std,
            "globals_mean": self.globals_mean.tolist(),
            "globals_std": self.globals_std.tolist(),
            "temp_mean": self.temp_mean,
            "temp_std": self.temp_std,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NormStats":
        return cls(
            power_mean=float(d["power_mean"]),
            power_std=float(d["power_std"]),
            globals_mean=np.asarray(d["globals_mean"], dtype=np.float32),
            globals_std=np.asarray(d["globals_std"], dtype=np.float32),
            temp_mean=float(d["temp_mean"]),
            temp_std=float(d["temp_std"]),
        )

    @classmethod
    def from_library(cls, power, kt, h, temp) -> "NormStats":
        lam = np.sqrt(kt / h)
        g = np.stack([kt, h, lam], axis=-1)
        return cls(
            power_mean=float(power.mean()),
            power_std=float(power.std() + 1e-12),
            globals_mean=g.mean(axis=0).astype(np.float32),
            globals_std=(g.std(axis=0) + 1e-12).astype(np.float32),
            temp_mean=float(temp.mean()),
            temp_std=float(temp.std() + 1e-12),
        )


def save_library(
    path: str | Path, spec: ChipSpec, power, kt, h, temp
) -> None:
    """Persist a design library to NPZ."""
    np.savez_compressed(
        path,
        power=power.astype(np.float32),
        kt=kt.astype(np.float32),
        h=h.astype(np.float32),
        temp=temp.astype(np.float32),
        grid=spec.grid,
        die_size=spec.die_size,
        total_power=spec.total_power,
    )


def load_library(path: str | Path):
    """Load a design library; returns ``(spec, power, kt, h, temp)``."""
    data = np.load(path)
    spec = ChipSpec(
        grid=int(data["grid"]),
        die_size=float(data["die_size"]),
        total_power=float(data["total_power"]),
    )
    return spec, data["power"], data["kt"], data["h"], data["temp"]


def _coordinate_grid(g: int) -> np.ndarray:
    """Normalized (x, y) channels in [-1, 1], shape (g, g, 2)."""
    axis = np.linspace(-1.0, 1.0, g, dtype=np.float32)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    return np.stack([x, y], axis=-1)


_KERNEL_CACHE: dict[tuple[float, str], torch.Tensor] = {}


def _gaussian_blur(field: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur with replicate padding, (B, H, W) -> same."""
    key = (sigma, str(field.device))
    if key not in _KERNEL_CACHE:
        radius = max(1, int(3.0 * sigma))
        x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=field.device)
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        _KERNEL_CACHE[key] = (k / k.sum()).view(1, 1, 1, -1)
    k = _KERNEL_CACHE[key]
    radius = (k.shape[-1] - 1) // 2
    t = field.unsqueeze(1)
    t = torch.nn.functional.pad(t, (radius, radius, 0, 0), mode="replicate")
    t = torch.nn.functional.conv2d(t, k)
    t = torch.nn.functional.pad(t, (0, 0, radius, radius), mode="replicate")
    t = torch.nn.functional.conv2d(t, k.transpose(2, 3))
    return t.squeeze(1)


def encode_inputs(
    power: np.ndarray | torch.Tensor,
    kt: np.ndarray | torch.Tensor,
    h: np.ndarray | torch.Tensor,
    stats: NormStats,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build physics-informed model inputs from raw design data.

    Local channels: normalized power, the analytic no-spreading solution
    ``T0 = q / h`` (in target-standardized units), Gaussian-blurred ``T0``
    at :data:`BLUR_SIGMAS` scales (cheap spreading proxies), and (x, y).
    Global channels: standardized ``[kt, h, lambda]`` with the healing
    length ``lambda = sqrt(kt / h)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(local_embedding, global_embedding)`` with shapes
        ``(B, g, g, N_LOCAL_FEATURES)`` and ``(B, 1, N_GLOBAL_FEATURES)``.
    """
    power = torch.as_tensor(power, dtype=torch.float32, device=device)
    if power.ndim == 2:
        power = power.unsqueeze(0)
    b, g, _ = power.shape

    kt = torch.as_tensor(kt, dtype=torch.float32, device=device).reshape(b, 1)
    h = torch.as_tensor(h, dtype=torch.float32, device=device).reshape(b, 1)

    p_norm = (power - stats.power_mean) / stats.power_std
    t0 = power / h.view(b, 1, 1)
    t0_norm = (t0 - stats.temp_mean) / stats.temp_std

    channels = [p_norm, t0_norm]
    channels += [_gaussian_blur(t0_norm, s) for s in BLUR_SIGMAS]
    coords = torch.from_numpy(_coordinate_grid(g)).to(device)
    coords = coords.unsqueeze(0).expand(b, -1, -1, -1)
    local = torch.cat(
        [torch.stack(channels, dim=-1), coords], dim=-1
    )

    lam = torch.sqrt(kt / h)
    g_raw = torch.cat([kt, h, lam], dim=-1)
    g_mean = torch.from_numpy(stats.globals_mean).to(device)
    g_std = torch.from_numpy(stats.globals_std).to(device)
    global_emb = ((g_raw - g_mean) / g_std).unsqueeze(1)

    return local, global_emb


def decode_temperature(pred: torch.Tensor, stats: NormStats) -> torch.Tensor:
    """Invert output standardization -> temperature rise [K], (B, g, g)."""
    return pred.squeeze(-1) * stats.temp_std + stats.temp_mean


class ChipThermalDataset(Dataset):
    """Torch dataset over a design library with standardized targets."""

    def __init__(self, power, kt, h, temp, stats: NormStats):
        self.power = power
        self.kt = kt
        self.h = h
        self.temp = temp
        self.stats = stats

    def __len__(self) -> int:
        return self.power.shape[0]

    def __getitem__(self, idx: int):
        local, global_emb = encode_inputs(
            self.power[idx], self.kt[idx : idx + 1], self.h[idx : idx + 1], self.stats
        )
        target = (
            torch.from_numpy(self.temp[idx]).float() - self.stats.temp_mean
        ) / self.stats.temp_std
        return local[0], global_emb[0], target.unsqueeze(-1)
