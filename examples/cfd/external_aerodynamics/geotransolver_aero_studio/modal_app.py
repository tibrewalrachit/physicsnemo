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

"""Modal GPU backend for GeoTransolver Aero Studio.

Deploys the aero inference engine as a serverless GPU service on
`Modal <https://modal.com>`_, so the web studio / CLI can run on any
laptop while inference executes on an A100 or H100.

Deploy (defaults: A100, ``conf/config_modal.yaml``)::

    modal deploy modal_app.py

Deploy variants via environment variables (read at deploy time)::

    AERO_STUDIO_GPU=H100 modal deploy modal_app.py
    AERO_STUDIO_GPU=A100-80GB AERO_STUDIO_CONFIG=conf/config_demo.yaml \\
        AERO_STUDIO_MODAL_APP=geotransolver-aero-studio-demo \\
        modal deploy modal_app.py

Upload a trained checkpoint to the persistent volume once::

    modal volume put aero-studio-checkpoints /path/to/checkpoints /geotransolver

Then point the local server at the deployment with
``inference.backend: modal`` in its config.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[3]

# Deploy-time knobs (baked into the deployment, not read in the container):
APP_NAME = os.environ.get("AERO_STUDIO_MODAL_APP", "geotransolver-aero-studio")
GPU = os.environ.get("AERO_STUDIO_GPU", "A100")  # A100 | A100-80GB | H100 | ...
REMOTE_CONFIG = os.environ.get("AERO_STUDIO_CONFIG", "conf/config_modal.yaml")

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    # VTK (pyvista) needs system GL/X libraries even for file I/O.
    .apt_install("libgl1", "libglu1-mesa", "libxrender1", "libglib2.0-0")
    .pip_install(
        "torch>=2.10",
        "torchvision",
        "numpy",
        "pyvista",
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
    .env({"AERO_STUDIO_CONFIG": REMOTE_CONFIG})
    # Ship the local PhysicsNeMo source and the studio engine into the image
    # so remote inference exactly matches the local checkout.
    .add_local_dir(
        str(_REPO_ROOT / "physicsnemo"),
        "/root/physicsnemo",
        ignore=["**/__pycache__"],
    )
    .add_local_dir(
        str(_HERE / "aero_studio"), "/root/aero_studio", ignore=["**/__pycache__"]
    )
    .add_local_dir(str(_HERE / "conf"), "/root/conf")
)

# Persistent storage for trained checkpoints; mounted at /checkpoints.
checkpoints = modal.Volume.from_name("aero-studio-checkpoints", create_if_missing=True)


@app.cls(
    image=image,
    gpu=GPU,
    volumes={"/checkpoints": checkpoints},
    timeout=1800,
    scaledown_window=300,
)
class AeroGPU:
    """GPU-resident aero predictor service."""

    @modal.enter()
    def load(self) -> None:
        """Load the model once per container (survives across calls)."""
        from aero_studio.config import load_app_config
        from aero_studio.engine import AeroPredictor

        config_path = Path("/root") / os.environ["AERO_STUDIO_CONFIG"]
        cfg = load_app_config(config_path)
        self.gpu_kind = GPU
        self.predictor = AeroPredictor(cfg)

    @modal.method()
    def info(self) -> dict:
        """Backend metadata for the studio's health endpoint."""
        import torch

        return {
            "model": self.predictor.model_name,
            "trained": self.predictor.trained,
            "device": str(self.predictor.device),
            "deviceName": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            ),
            "gpu": self.gpu_kind,
        }

    @modal.method()
    def predict(
        self,
        stl_bytes: bytes,
        suffix: str = ".stl",
        stream_velocity: float = 30.0,
        air_density: float = 1.205,
        mc_samples: int = 0,
    ) -> dict:
        """Run inference on raw geometry bytes; return the prediction as a dict.

        The return value is ``dataclasses.asdict(AeroPrediction)``, so the
        client can rebuild the same dataclass losslessly (numpy arrays
        included).
        """
        import dataclasses
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=suffix) as f:
            f.write(stl_bytes)
            f.flush()
            result, _mesh = self.predictor.predict(
                f.name,
                stream_velocity=stream_velocity,
                air_density=air_density,
                mc_samples=mc_samples,
            )
        return dataclasses.asdict(result)


@app.local_entrypoint()
def smoke(stl: str, velocity: float = 30.0, density: float = 1.205):
    """Quick remote test: ``modal run modal_app.py --stl path/to/car.stl``."""
    data = Path(stl).read_bytes()
    service = AeroGPU()
    print(service.info.remote())
    out = service.predict.remote(
        data, Path(stl).suffix, stream_velocity=velocity, air_density=density
    )
    print(
        f"Cd={out['drag_coefficient']:.4f} Cl={out['lift_coefficient']:.4f} "
        f"cells={out['n_cells']} t={out['inference_seconds']:.2f}s"
    )
