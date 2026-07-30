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

"""Remote GPU predictor backed by a deployed Modal app.

Drop-in replacement for :class:`~aero_studio.engine.AeroPredictor`: same
``predict`` signature, but the heavy lifting (datapipe + GeoTransolver
forward passes) runs on a serverless A100/H100 deployed with
``modal deploy modal_app.py``. Only the geometry bytes travel up and the
predicted fields travel back; mesh handling for visualization and VTP
export stays local.
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from .engine import AeroPrediction, attach_prediction_to_mesh
from .geometry import read_surface_mesh

logger = logging.getLogger("aero_studio.remote")


class ModalAeroPredictor:
    """Client for the Modal-deployed :class:`AeroGPU` service.

    Parameters
    ----------
    cfg : DictConfig
        Application config. Reads ``modal.app_name`` / ``modal.class_name``
        (defaults match ``modal_app.py``). Requires the ``modal`` package
        and an authenticated Modal token (``modal token set ...``).
    """

    def __init__(self, cfg: DictConfig) -> None:
        try:
            import modal
        except ImportError as exc:
            raise ImportError(
                "inference.backend=modal requires the 'modal' package: pip install modal"
            ) from exc

        modal_cfg = cfg.get("modal", None) or {}
        app_name = modal_cfg.get("app_name", "geotransolver-aero-studio")
        class_name = modal_cfg.get("class_name", "AeroGPU")

        try:
            service_cls = modal.Cls.from_name(app_name, class_name)
            self._service = service_cls()
            info = self._service.info.remote()
        except Exception as exc:
            raise RuntimeError(
                f"Could not reach Modal app '{app_name}' (class '{class_name}'). "
                f"Deploy it first with 'modal deploy modal_app.py' and make sure "
                f"'modal token set' has been run. Original error: {exc}"
            ) from exc

        self.model_name = info["model"]
        self.trained = info["trained"]
        self.device = f"modal/{info['gpu']} ({info.get('deviceName', 'gpu')})"
        logger.info(
            f"Connected to Modal app '{app_name}': {self.model_name} on "
            f"{self.device}, trained={self.trained}"
        )

    def predict(
        self,
        stl_path: str | Path,
        stream_velocity: float = 30.0,
        air_density: float = 1.205,
        mc_samples: int = 0,
    ) -> tuple[AeroPrediction, "pv.PolyData"]:  # noqa: F821
        """Predict surface aerodynamics via the remote GPU service.

        Mirrors :meth:`aero_studio.engine.AeroPredictor.predict`.
        """
        stl_path = Path(stl_path)
        data = self._service.predict.remote(
            stl_path.read_bytes(),
            stl_path.suffix,
            stream_velocity=float(stream_velocity),
            air_density=float(air_density),
            mc_samples=int(mc_samples),
        )
        result = AeroPrediction(**data)

        mesh = read_surface_mesh(stl_path)
        if mesh.n_cells != result.n_cells:
            raise RuntimeError(
                f"Remote prediction has {result.n_cells} cells but the local "
                f"mesh read produced {mesh.n_cells}. Local and remote pyvista "
                f"versions likely triangulate differently - align the pyvista "
                f"versions."
            )
        attach_prediction_to_mesh(mesh, result)
        return result, mesh
