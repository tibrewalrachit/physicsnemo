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

"""Run the full chip-thermal workflow on a Modal GPU (A100 by default).

Generates the design library, trains the surrogate, benchmarks it
against the high-fidelity solver across grid sizes, and runs a design
exploration - all remotely - then downloads the artifacts to
``../outputs/modal/``.

Usage::

    modal run run_on_modal.py                     # defaults: A100, grid 128
    AERO_STUDIO_GPU=H100 modal run run_on_modal.py --grid 256 --designs 1600
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
try:
    _REPO_ROOT = _HERE.parents[3]
except IndexError:  # inside the container; unused there
    _REPO_ROOT = _HERE

GPU = os.environ.get("AERO_STUDIO_GPU", "A100")

app = modal.App("chip-thermal-studio")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.10",
        "torchvision",
        "numpy",
        "requests",
        "importlib-metadata",
        "matplotlib",
        "omegaconf",
        "hydra-core",
        "jaxtyping",
        "einops",
        "tqdm",
        "h5py",
        "pandas",
        "warp-lang>=1.14.0",
        "termcolor",
        "psutil",
        "treelib",
        "nvtx",
        "tensordict",
        "onnx",
        "s3fs",
        "cftime",
        "timm",
        "GitPython",
    )
    .add_local_dir(
        str(_REPO_ROOT / "physicsnemo"),
        "/root/physicsnemo",
        ignore=["**/__pycache__"],
    )
    .add_local_dir(str(_HERE), "/root/chipsrc", ignore=["**/__pycache__"])
    .add_local_dir(str(_HERE.parent / "conf"), "/root/conf")
)


@app.function(image=image, gpu=GPU, timeout=3600)
def run_workflow(
    grid: int = 128,
    designs: int = 1600,
    epochs: int = 40,
    batch_size: int = 16,
    candidates: int = 2000,
    bench_grids: str = "64,128,256,512",
) -> dict:
    """Generate -> train -> benchmark -> explore on the GPU; return artifacts."""
    import json
    import subprocess
    import sys

    src = "/root/chipsrc"
    out = Path("/root/outputs")
    data = Path("/root/data")
    out.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)

    cfg_path = Path("/root/train_config.yaml")
    cfg_path.write_text(
        f"""
data:
  library: {data}/library.npz
  val_fraction: 0.1
model:
  n_layers: 4
  n_hidden: 96
  n_head: 8
  slice_num: 48
  mlp_ratio: 2
  use_te: false
training:
  seed: 0
  epochs: {epochs}
  batch_size: {batch_size}
  lr: 1.0e-3
  weight_decay: 1.0e-5
  output_dir: {out}
"""
    )

    def run(step: str, *cmd: str) -> str:
        print(f"=== {step}: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(
            [sys.executable, *cmd], cwd=src, capture_output=True, text=True
        )
        print(proc.stdout[-4000:], flush=True)
        if proc.returncode != 0:
            print(proc.stderr[-4000:], flush=True)
            raise RuntimeError(f"{step} failed with code {proc.returncode}")
        return proc.stdout

    gen_log = run(
        "generate",
        "generate_dataset.py",
        "--n", str(designs),
        "--grid", str(grid),
        "--batch", "200",
        "--out", str(data / "library.npz"),
    )
    train_log = run("train", "train.py", "--config", str(cfg_path))
    bench_log = run(
        "benchmark",
        "benchmark.py",
        "--grids", *bench_grids.split(","),
        "--designs", "8",
        "--batch", "8",
        "--checkpoint", str(out / "surrogate.pt"),
        "--json", str(out / "benchmark.json"),
    )
    explore_log = run(
        "explore",
        "explore.py",
        "--checkpoint", str(out / "surrogate.pt"),
        "--candidates", str(candidates),
        "--top-k", "5",
        "--validate", "25",
        "--csv", str(out / "exploration.csv"),
        "--report", str(out / "exploration.png"),
    )

    return {
        "gpu": GPU,
        "grid": grid,
        "logs": {
            "generate": gen_log[-2000:],
            "train": train_log[-3000:],
            "benchmark": bench_log[-2000:],
            "explore": explore_log[-2000:],
        },
        "benchmark": json.loads((out / "benchmark.json").read_text()),
        "files": {
            "exploration.png": (out / "exploration.png").read_bytes(),
            "exploration.csv": (out / "exploration.csv").read_bytes(),
            "surrogate.pt": (out / "surrogate.pt").read_bytes(),
        },
    }


@app.local_entrypoint()
def main(
    grid: int = 128,
    designs: int = 1600,
    epochs: int = 40,
    candidates: int = 2000,
    bench_grids: str = "64,128,256,512",
):
    result = run_workflow.remote(
        grid=grid,
        designs=designs,
        epochs=epochs,
        candidates=candidates,
        bench_grids=bench_grids,
    )
    out_dir = _HERE.parent / "outputs" / "modal"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, blob in result["files"].items():
        (out_dir / name).write_bytes(blob)
        print(f"saved {out_dir / name} ({len(blob)} bytes)")
    print(f"\n=== {result['gpu']} benchmark (grid sweep) ===")
    for row in result["benchmark"]["rows"]:
        print(
            f"grid {row['grid']:>4}: solver {row['solver_ms']:8.2f} ms  "
            f"AI {row['surrogate_ms']:6.2f} ms  speedup {row['speedup']:6.1f}x"
        )
    print("\n=== training tail ===")
    print("\n".join(result["logs"]["train"].strip().splitlines()[-4:]))
    print("\n=== exploration tail ===")
    print(result["logs"]["explore"].strip())
