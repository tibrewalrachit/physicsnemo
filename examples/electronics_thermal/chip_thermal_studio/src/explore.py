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

"""AI-accelerated thermal design exploration.

Uses the trained surrogate to sweep floorplan candidates and rank them by
predicted peak temperature - the "AI engine for robust design
exploration" of the SeaScape workflow - then validates the winners (and a
random control subset) against the high-fidelity solver.

Usage::

    python explore.py --checkpoint ../outputs/surrogate.pt \
        --candidates 2000 --top-k 5 --report ../outputs/exploration.png
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import NormStats, decode_temperature, encode_inputs
from thermal import ChipSpec, random_design, solve_design
from train import build_model
from omegaconf import OmegaConf


def load_surrogate(checkpoint: Path, device):
    """Load a trained surrogate checkpoint -> (model, stats, spec)."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    stats = NormStats.from_dict(ckpt["stats"])
    spec = ChipSpec(
        grid=int(ckpt["grid"]),
        die_size=float(ckpt["die_size"]),
        total_power=float(ckpt["total_power"]),
    )
    model = build_model(OmegaConf.create(ckpt["model_cfg"]), spec.grid).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, stats, spec


@torch.no_grad()
def surrogate_predict(model, stats, designs, device, batch: int = 64):
    """Predict temperature fields for a list of designs -> (N, g, g) [K]."""
    fields = []
    for i in range(0, len(designs), batch):
        chunk = designs[i : i + batch]
        local, global_emb = encode_inputs(
            np.stack([d.power_map for d in chunk]),
            np.array([d.kt for d in chunk]),
            np.array([d.h for d in chunk]),
            stats,
            device=device,
        )
        pred = model(local, global_embedding=global_emb)
        fields.append(decode_temperature(pred, stats).cpu().numpy())
    return np.concatenate(fields)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="../outputs/surrogate.pt")
    parser.add_argument("--candidates", type=int, default=2000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--validate", type=int, default=20, help="Extra random candidates to validate"
    )
    parser.add_argument("--total-power", type=float, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--csv", default="../outputs/exploration.csv")
    parser.add_argument("--report", default="../outputs/exploration.png")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, stats, spec = load_surrogate(Path(args.checkpoint), device)
    if args.total_power is not None:
        spec.total_power = args.total_power
    rng = np.random.default_rng(args.seed)

    # --- 1. Candidate sweep with the AI engine ---
    designs = [random_design(spec, rng) for _ in range(args.candidates)]
    t0 = time.time()
    fields = surrogate_predict(model, stats, designs, device)
    ai_time = time.time() - t0
    peaks = fields.max(axis=(1, 2))
    order = np.argsort(peaks)

    # --- 2. High-fidelity validation of winners + random controls ---
    t0 = time.time()
    _ = solve_design(designs[0], spec, device=str(device))
    solver_time = time.time() - t0

    check_idx = list(order[: args.top_k]) + list(
        rng.choice(args.candidates, size=min(args.validate, args.candidates), replace=False)
    )
    check_idx = list(dict.fromkeys(int(i) for i in check_idx))
    true_peaks = {}
    for i in check_idx:
        true_peaks[i] = float(solve_design(designs[i], spec, device=str(device)).max())

    # --- 3. Report ---
    best = int(order[0])
    ai_ms = ai_time / args.candidates * 1000
    speedup = solver_time / (ai_time / args.candidates)
    print(
        f"swept {args.candidates} candidates in {ai_time:.2f}s "
        f"({ai_ms:.2f} ms/design vs {solver_time * 1000:.0f} ms/design "
        f"high-fidelity -> {speedup:.0f}x speedup)"
    )
    print(
        f"best design #{best}: predicted peak dT {peaks[best]:.1f} K, "
        f"solver-verified {true_peaks[best]:.1f} K "
        f"(kt={designs[best].kt:.3f} W/K, h={designs[best].h:.0f} W/m2K)"
    )
    errs = [abs(peaks[i] - true_peaks[i]) for i in check_idx]
    print(
        f"validation on {len(check_idx)} designs: "
        f"peak-dT MAE {np.mean(errs):.2f} K, max {np.max(errs):.2f} K"
    )

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["rank", "design", "predicted_peak_dT_K", "solver_peak_dT_K", "kt_W_per_K", "h_W_per_m2K"]
        )
        for rank, i in enumerate(order):
            writer.writerow(
                [
                    rank,
                    int(i),
                    f"{peaks[i]:.3f}",
                    f"{true_peaks.get(int(i), ''):.3f}" if int(i) in true_peaks else "",
                    f"{designs[i].kt:.4f}",
                    f"{designs[i].h:.0f}",
                ]
            )
    print(f"ranking written to {csv_path}")

    true_best = solve_design(designs[best], spec, device=str(device))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    extent = [0, spec.die_size * 1000, 0, spec.die_size * 1000]

    im = axes[0, 0].imshow(designs[best].power_map / 1e6, cmap="inferno", extent=extent)
    axes[0, 0].set_title(f"Best floorplan #{best}: power density")
    fig.colorbar(im, ax=axes[0, 0], label="MW/m$^2$")

    im = axes[0, 1].imshow(fields[best], cmap="turbo", extent=extent)
    axes[0, 1].set_title(f"AI-predicted ΔT (peak {peaks[best]:.1f} K)")
    fig.colorbar(im, ax=axes[0, 1], label="K")

    im = axes[0, 2].imshow(true_best, cmap="turbo", extent=extent)
    axes[0, 2].set_title(f"High-fidelity ΔT (peak {true_best.max():.1f} K)")
    fig.colorbar(im, ax=axes[0, 2], label="K")

    im = axes[1, 0].imshow(fields[best] - true_best, cmap="coolwarm", extent=extent)
    axes[1, 0].set_title("AI − solver error")
    fig.colorbar(im, ax=axes[1, 0], label="K")

    xs = [true_peaks[i] for i in check_idx]
    ys = [peaks[i] for i in check_idx]
    lo, hi = min(xs + ys), max(xs + ys)
    axes[1, 1].scatter(xs, ys, s=28, c="#76b900", edgecolors="k", linewidths=0.4)
    axes[1, 1].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[1, 1].set_xlabel("solver peak ΔT [K]")
    axes[1, 1].set_ylabel("AI peak ΔT [K]")
    axes[1, 1].set_title(f"Validation ({len(check_idx)} designs)")

    axes[1, 2].hist(peaks, bins=40, color="#4fc3f7", edgecolor="k", linewidth=0.3)
    axes[1, 2].axvline(peaks[best], color="#76b900", linewidth=2, label="best design")
    axes[1, 2].set_xlabel("predicted peak ΔT [K]")
    axes[1, 2].set_ylabel("candidates")
    axes[1, 2].set_title(f"Design space ({args.candidates} candidates)")
    axes[1, 2].legend()

    for ax in axes.flat[:4]:
        ax.set_xlabel("mm")
        ax.set_ylabel("mm")
    fig.suptitle(
        f"AI-accelerated thermal design exploration — {speedup:.0f}x faster than "
        f"the high-fidelity solver ({ai_ms:.2f} ms vs {solver_time * 1000:.0f} ms per design)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(report, dpi=130)
    print(f"report written to {report}")


if __name__ == "__main__":
    main()
