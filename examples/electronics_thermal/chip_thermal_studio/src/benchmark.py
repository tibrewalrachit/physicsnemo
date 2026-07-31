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

"""Benchmark: AI surrogate vs high-fidelity solver across design sizes.

The solver's conjugate-gradient cost grows roughly 8x per grid doubling
(4x cells and ~2x iterations from the worsening condition number), while
the surrogate's forward pass grows only with the token count - so the
speedup opens up on larger designs.

Usage::

    python benchmark.py --grids 64 128 256 --designs 8 --batch 8
    python benchmark.py --grids 64 128 256 --checkpoint ../outputs/surrogate.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from dataset import NormStats, encode_inputs
from thermal import ChipSpec, random_design, solve_temperature
from train import build_model


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_solver(power, kt, h, spec, device, repeats: int = 1) -> float:
    """High-fidelity solve time per design [s] (batched)."""
    q = torch.from_numpy(power).to(device)
    kt_t = torch.from_numpy(kt).to(device)
    h_t = torch.from_numpy(h).to(device)
    solve_temperature(q[:1], kt_t[:1], h_t[:1], spec.dx)  # warmup
    _sync(device)
    start = time.time()
    for _ in range(repeats):
        solve_temperature(q, kt_t, h_t, spec.dx)
    _sync(device)
    return (time.time() - start) / (repeats * power.shape[0])


@torch.inference_mode()
def time_surrogate(model, stats, power, kt, h, device, repeats: int = 3) -> float:
    """Surrogate inference time per design [s] (batched, incl. encoding)."""
    model.eval()
    local, global_emb = encode_inputs(power, kt, h, stats, device=device)
    model(local, global_embedding=global_emb)  # warmup
    _sync(device)
    start = time.time()
    for _ in range(repeats):
        local, global_emb = encode_inputs(power, kt, h, stats, device=device)
        model(local, global_embedding=global_emb)
    _sync(device)
    return (time.time() - start) / (repeats * power.shape[0])


def placeholder_stats() -> NormStats:
    """Stats used when no checkpoint is given (timing does not depend on them)."""
    return NormStats(
        power_mean=1.0e6,
        power_std=1.0e6,
        globals_mean=np.array([0.08, 2.5e4, 1.8e-3], dtype=np.float32),
        globals_std=np.array([0.03, 1.0e4, 5.0e-4], dtype=np.float32),
        temp_mean=50.0,
        temp_std=30.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--designs", type=int, default=8, help="Designs per grid")
    parser.add_argument("--batch", type=int, default=8, help="Surrogate batch size")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional trained checkpoint; its model config sizes the timing "
        "model (weights only matter for its own grid)",
    )
    parser.add_argument("--model-config", default="../conf/config_demo.yaml")
    parser.add_argument("--json", default=None, help="Write results JSON here")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model_cfg = OmegaConf.create(ckpt["model_cfg"])
        stats = NormStats.from_dict(ckpt["stats"])
    else:
        model_cfg = OmegaConf.load(args.model_config).model
        stats = placeholder_stats()

    rng = np.random.default_rng(0)
    rows = []
    print(f"device: {device}")
    print(f"{'grid':>6} {'cells':>8} {'solver ms':>10} {'AI ms':>8} {'speedup':>8}")
    for grid in args.grids:
        spec = ChipSpec(grid=grid)
        n = min(args.designs, args.batch)
        designs = [random_design(spec, rng) for _ in range(n)]
        power = np.stack([d.power_map for d in designs])
        kt = np.array([d.kt for d in designs], dtype=np.float32)
        h = np.array([d.h for d in designs], dtype=np.float32)

        solver_s = time_solver(power, kt, h, spec, device)

        model = build_model(model_cfg, grid).to(device)
        surrogate_s = time_surrogate(model, stats, power, kt, h, device)
        del model

        speedup = solver_s / surrogate_s
        rows.append(
            {
                "grid": grid,
                "cells": grid * grid,
                "solver_ms": solver_s * 1000,
                "surrogate_ms": surrogate_s * 1000,
                "speedup": speedup,
            }
        )
        print(
            f"{grid:>6} {grid * grid:>8} {solver_s * 1000:>10.2f} "
            f"{surrogate_s * 1000:>8.2f} {speedup:>7.1f}x"
        )

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"device": str(device), "rows": rows}, indent=2))
        print(f"results written to {out}")


if __name__ == "__main__":
    main()
