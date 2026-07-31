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

"""Generate the design library ("completed high-fidelity simulations").

Samples random chip floorplans and solves each with the finite-difference
thermal solver, mimicking a library of signed-off thermal analyses that a
design team accumulates over time.

Usage::

    python generate_dataset.py --n 2000 --out ../data/library.npz
    python generate_dataset.py --n 200 --grid 64 --seed 7   # quick demo
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from dataset import save_library
from thermal import ChipSpec, random_design, solve_temperature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000, help="Number of designs")
    parser.add_argument("--grid", type=int, default=64, help="Grid cells per side")
    parser.add_argument(
        "--total-power", type=float, default=100.0, help="Power per design [W]"
    )
    parser.add_argument("--out", default="../data/library.npz", help="Output NPZ")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=64, help="Solver batch size")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    spec = ChipSpec(grid=args.grid, total_power=args.total_power)
    rng = np.random.default_rng(args.seed)

    designs = [random_design(spec, rng) for _ in range(args.n)]
    power = np.stack([d.power_map for d in designs])
    kt = np.array([d.kt for d in designs], dtype=np.float32)
    h = np.array([d.h for d in designs], dtype=np.float32)

    temps = []
    start = time.time()
    for i in range(0, args.n, args.batch):
        sl = slice(i, min(i + args.batch, args.n))
        dt = solve_temperature(
            torch.from_numpy(power[sl]).to(device),
            torch.from_numpy(kt[sl]).to(device),
            torch.from_numpy(h[sl]).to(device),
            spec.dx,
        )
        temps.append(dt.cpu().numpy())
        print(f"solved {sl.stop}/{args.n} designs ({time.time() - start:.1f}s)")
    temp = np.concatenate(temps)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_library(out, spec, power, kt, h, temp)

    per_design = (time.time() - start) / args.n
    print(
        f"library: {args.n} designs @ {args.grid}x{args.grid} -> {out} "
        f"({per_design * 1000:.1f} ms/design solve time)"
    )
    print(
        f"peak dT range: {temp.max(axis=(1, 2)).min():.1f} - "
        f"{temp.max(axis=(1, 2)).max():.1f} K"
    )


if __name__ == "__main__":
    main()
