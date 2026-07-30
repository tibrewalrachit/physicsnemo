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

"""Batch CLI for GeoTransolver Aero Studio.

Evaluate one or many geometries from the command line - the programmatic
counterpart to the web UI, intended for design-space sweeps.

Usage::

    python cli.py designs/*.stl --config conf/config.yaml \
        --velocity 30.0 --density 1.205 \
        --csv sweep_results.csv --output-dir predictions/
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometries", nargs="+", help="STL/VTP geometry files")
    parser.add_argument("--config", default=None, help="App config YAML")
    parser.add_argument(
        "--velocity", type=float, default=None, help="Free-stream velocity [m/s]"
    )
    parser.add_argument(
        "--density", type=float, default=None, help="Air density [kg/m^3]"
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=0,
        help="MC-Dropout passes for uncertainty (needs a concrete-dropout checkpoint)",
    )
    parser.add_argument("--csv", default=None, help="Write a summary CSV here")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write per-geometry VTP predictions into this directory",
    )
    parser.add_argument("--device", default=None, help="Torch device override")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from aero_studio.config import load_app_config
    from aero_studio.engine import AeroPredictor

    cfg = load_app_config(args.config)
    velocity = (
        args.velocity
        if args.velocity is not None
        else float(cfg.physics.get("default_stream_velocity", 30.0))
    )
    density = (
        args.density
        if args.density is not None
        else float(cfg.physics.get("default_air_density", 1.205))
    )

    predictor = AeroPredictor(cfg, device=args.device)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = 0
    for geom in args.geometries:
        geom_path = Path(geom)
        if not geom_path.exists():
            print(f"[skip] {geom_path}: not found", file=sys.stderr)
            failures += 1
            continue
        try:
            result, mesh = predictor.predict(
                geom_path,
                stream_velocity=velocity,
                air_density=density,
                mc_samples=args.mc_samples,
            )
        except Exception as exc:
            print(f"[fail] {geom_path}: {exc}", file=sys.stderr)
            failures += 1
            continue

        row = {"geometry": geom_path.name, **result.summary()}
        rows.append(row)
        print(
            f"{geom_path.name}: Cd={result.drag_coefficient:.4f} "
            f"Cl={result.lift_coefficient:.4f} "
            f"A_front={result.frontal_area:.3f} m^2 "
            f"({result.n_cells} cells, {result.inference_seconds:.2f}s)"
        )

        if output_dir:
            out_vtp = output_dir / f"{geom_path.stem}_prediction.vtp"
            mesh.save(str(out_vtp))
            print(f"  -> {out_vtp}")

    if args.csv and rows:
        csv_path = Path(args.csv)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} results to {csv_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
