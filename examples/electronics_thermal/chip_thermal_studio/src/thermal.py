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

"""Chip floorplans and a finite-difference die thermal solver.

This module plays the role of the "high-fidelity simulator" in the
SeaScape-style workflow: it generates randomized chip floorplans (power
maps) and solves the steady-state compact thermal model

.. math::

    -k_{eff}\\, t_{die}\\, \\nabla^2 \\Delta T + h_{eff}\\, \\Delta T = q(x, y)

on the die plane, where :math:`q` is the areal power density [W/m^2],
:math:`k_{eff} t_{die}` the effective lateral heat-spreading conductance
[W/K], and :math:`h_{eff}` the effective through-package conductance to
ambient per unit area [W/(m^2 K)]. Lateral die edges are adiabatic
(Neumann). The operator is symmetric positive definite, so the system is
solved with conjugate gradients in pure PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class ChipSpec:
    """Design-space specification for generated chip floorplans."""

    grid: int = 64  # cells per side
    die_size: float = 0.010  # die edge length [m] (10 mm)
    total_power: float = 100.0  # total dissipated power [W]
    n_blocks_range: tuple[int, int] = (4, 12)  # power macros per design
    block_frac_range: tuple[float, float] = (0.08, 0.45)  # block edge / die edge
    background_frac: float = 0.15  # fraction of power spread as leakage
    # Effective lateral spreading conductance k_eff * t_die [W/K]:
    kt_range: tuple[float, float] = (0.04, 0.12)
    # Effective through-package conductance to ambient [W/(m^2 K)]:
    h_range: tuple[float, float] = (1.0e4, 4.0e4)

    @property
    def dx(self) -> float:
        return self.die_size / self.grid

    @property
    def cell_area(self) -> float:
        return self.dx * self.dx


@dataclass
class ChipDesign:
    """A single chip design: floorplan blocks plus package parameters."""

    power_map: np.ndarray  # (grid, grid) areal power density [W/m^2]
    kt: float  # k_eff * t_die [W/K]
    h: float  # h_eff [W/(m^2 K)]
    blocks: list = field(default_factory=list)  # (x0, y0, w, h_, watts)

    @property
    def total_power(self) -> float:
        return float(self.power_map.sum())


def random_design(spec: ChipSpec, rng: np.random.Generator) -> ChipDesign:
    """Sample a random floorplan meeting the specification.

    Blocks are axis-aligned rectangles with power split randomly among
    them (Dirichlet); a uniform background accounts for leakage/global
    routing power. The power map integrates exactly to ``total_power``.
    """
    g = spec.grid
    n_blocks = int(rng.integers(spec.n_blocks_range[0], spec.n_blocks_range[1] + 1))

    block_power = spec.total_power * (1.0 - spec.background_frac)
    shares = rng.dirichlet(np.ones(n_blocks)) * block_power

    density = np.zeros((g, g), dtype=np.float64)
    blocks = []
    lo, hi = spec.block_frac_range
    for watts in shares:
        w = max(2, int(rng.uniform(lo, hi) * g))
        h_ = max(2, int(rng.uniform(lo, hi) * g))
        x0 = int(rng.integers(0, g - w + 1))
        y0 = int(rng.integers(0, g - h_ + 1))
        area = w * h_ * spec.cell_area
        density[y0 : y0 + h_, x0 : x0 + w] += watts / area
        blocks.append((x0, y0, w, h_, float(watts)))

    background = spec.total_power * spec.background_frac
    density += background / (g * g * spec.cell_area)

    kt = float(rng.uniform(*spec.kt_range))
    h = float(rng.uniform(*spec.h_range))
    # Return power per cell -> density is W/m^2; power_map stores density.
    return ChipDesign(power_map=density.astype(np.float32), kt=kt, h=h, blocks=blocks)


def _laplacian_neumann(t: torch.Tensor, dx: float) -> torch.Tensor:
    """5-point Laplacian with zero-flux (replicate) boundary conditions.

    Parameters
    ----------
    t : torch.Tensor
        Field of shape ``(B, H, W)``.
    dx : float
        Grid spacing.
    """
    p = torch.nn.functional.pad(t.unsqueeze(1), (1, 1, 1, 1), mode="replicate")[:, 0]
    return (
        p[:, 1:-1, :-2] + p[:, 1:-1, 2:] + p[:, :-2, 1:-1] + p[:, 2:, 1:-1]
        - 4.0 * t
    ) / (dx * dx)


def solve_temperature(
    power_density: torch.Tensor,
    kt: torch.Tensor,
    h: torch.Tensor,
    dx: float,
    tol: float = 1.0e-7,
    max_iter: int = 2000,
) -> torch.Tensor:
    """Solve the compact thermal model with conjugate gradients.

    Parameters
    ----------
    power_density : torch.Tensor
        Areal power density ``q`` of shape ``(B, H, W)`` [W/m^2].
    kt : torch.Tensor
        Per-design spreading conductance ``k_eff * t_die`` of shape ``(B,)``.
    h : torch.Tensor
        Per-design package conductance of shape ``(B,)``.
    dx : float
        Grid spacing [m].
    tol : float
        Relative residual tolerance.
    max_iter : int
        Maximum CG iterations.

    Returns
    -------
    torch.Tensor
        Temperature rise above ambient ``dT`` of shape ``(B, H, W)`` [K].
    """
    q = power_density.to(torch.float64)
    kt = kt.to(torch.float64).view(-1, 1, 1)
    h = h.to(torch.float64).view(-1, 1, 1)

    def matvec(t: torch.Tensor) -> torch.Tensor:
        return -kt * _laplacian_neumann(t, dx) + h * t

    x = q / h  # good initial guess: no-spreading solution
    r = q - matvec(x)
    p = r.clone()
    rs = (r * r).sum(dim=(1, 2), keepdim=True)
    b_norm = (q * q).sum(dim=(1, 2), keepdim=True).sqrt() + 1.0e-30

    for _ in range(max_iter):
        ap = matvec(p)
        alpha = rs / ((p * ap).sum(dim=(1, 2), keepdim=True) + 1.0e-30)
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = (r * r).sum(dim=(1, 2), keepdim=True)
        if bool((rs_new.sqrt() / b_norm).max() < tol):
            break
        p = r + (rs_new / (rs + 1.0e-30)) * p
        rs = rs_new

    return x.to(torch.float32)


def solve_design(design: ChipDesign, spec: ChipSpec, device: str = "cpu") -> np.ndarray:
    """Convenience wrapper: solve a single :class:`ChipDesign`."""
    q = torch.from_numpy(design.power_map).unsqueeze(0).to(device)
    kt = torch.tensor([design.kt], device=device)
    h = torch.tensor([design.h], device=device)
    return solve_temperature(q, kt, h, spec.dx)[0].cpu().numpy()
