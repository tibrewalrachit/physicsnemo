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

"""Train a GeoTransolver thermal surrogate on the design library.

The model runs in structured 2D mode (``structured_shape=(g, g)``): the
local embedding is the normalized power map plus (x, y) coordinates, the
global embedding carries the package parameters ``[kt, h]``, and the
output is the standardized temperature-rise field.

Usage::

    python train.py --config ../conf/config.yaml
    python train.py --config ../conf/config_demo.yaml   # tiny CPU demo
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataset import (
    N_GLOBAL_FEATURES,
    N_LOCAL_FEATURES,
    ChipThermalDataset,
    NormStats,
    load_library,
)
from physicsnemo.experimental.models.geotransolver import GeoTransolver


def build_model(model_cfg, grid: int) -> GeoTransolver:
    """Instantiate the structured-2D GeoTransolver from config."""
    return GeoTransolver(
        functional_dim=N_LOCAL_FEATURES,  # power, T0, blurred T0s, x, y
        out_dim=1,  # standardized temperature rise
        global_dim=N_GLOBAL_FEATURES,  # [kt, h, lambda]
        structured_shape=(grid, grid),
        n_layers=int(model_cfg.n_layers),
        n_hidden=int(model_cfg.n_hidden),
        n_head=int(model_cfg.n_head),
        slice_num=int(model_cfg.slice_num),
        mlp_ratio=int(model_cfg.mlp_ratio),
        use_te=bool(model_cfg.get("use_te", False)),
    )


def evaluate(model, loader, stats, device) -> dict:
    """Validation metrics in physical units."""
    model.eval()
    l2_num = l2_den = 0.0
    peak_err = []
    with torch.no_grad():
        for local, global_emb, target in loader:
            local = local.to(device)
            global_emb = global_emb.to(device)
            target = target.to(device)
            pred = model(local, global_embedding=global_emb)
            t_pred = pred.squeeze(-1) * stats.temp_std + stats.temp_mean
            t_true = target.squeeze(-1) * stats.temp_std + stats.temp_mean
            l2_num += ((t_pred - t_true) ** 2).sum().item()
            l2_den += (t_true**2).sum().item()
            peak_p = t_pred.amax(dim=(1, 2))
            peak_t = t_true.amax(dim=(1, 2))
            peak_err.extend((peak_p - peak_t).abs().cpu().tolist())
    return {
        "rel_l2": (l2_num / max(l2_den, 1e-30)) ** 0.5,
        "peak_mae_K": float(np.mean(peak_err)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="../conf/config.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg_dir = Path(args.config).resolve().parent
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(int(cfg.training.get("seed", 0)))

    library_path = (cfg_dir / cfg.data.library).resolve()
    spec, power, kt, h, temp = load_library(library_path)
    n = power.shape[0]
    n_val = max(1, int(n * float(cfg.data.val_fraction)))
    print(f"library: {n} designs @ {spec.grid}x{spec.grid} ({library_path})")

    stats = NormStats.from_library(
        power[: n - n_val], kt[: n - n_val], h[: n - n_val], temp[: n - n_val]
    )
    train_ds = ChipThermalDataset(
        power[: n - n_val], kt[: n - n_val], h[: n - n_val], temp[: n - n_val], stats
    )
    val_ds = ChipThermalDataset(
        power[n - n_val :], kt[n - n_val :], h[n - n_val :], temp[n - n_val :], stats
    )
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg.training.batch_size), shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=int(cfg.training.batch_size))

    model = build_model(cfg.model, spec.grid).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: GeoTransolver structured ({n_params:,} parameters) on {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.get("weight_decay", 1e-5)),
    )
    epochs = int(cfg.training.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    out_dir = (cfg_dir / cfg.training.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        start = time.time()
        for local, global_emb, target in train_loader:
            local = local.to(device)
            global_emb = global_emb.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred = model(local, global_embedding=global_emb)
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        metrics = evaluate(model, val_loader, stats, device)
        print(
            f"epoch {epoch + 1}/{epochs} loss={epoch_loss / len(train_loader):.5f} "
            f"val_rel_l2={metrics['rel_l2']:.4f} "
            f"val_peak_mae={metrics['peak_mae_K']:.2f}K "
            f"({time.time() - start:.1f}s)"
        )

        if metrics["rel_l2"] < best_val:
            best_val = metrics["rel_l2"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_cfg": OmegaConf.to_container(cfg.model),
                    "grid": spec.grid,
                    "die_size": spec.die_size,
                    "total_power": spec.total_power,
                    "stats": stats.to_dict(),
                    "val_metrics": metrics,
                    "encoding_version": 2,
                },
                out_dir / "surrogate.pt",
            )

    print(f"best val rel L2: {best_val:.4f}; checkpoint: {out_dir / 'surrogate.pt'}")
    with open(out_dir / "train_summary.json", "w") as f:
        json.dump({"best_val_rel_l2": best_val, "n_params": n_params}, f, indent=2)


if __name__ == "__main__":
    main()
