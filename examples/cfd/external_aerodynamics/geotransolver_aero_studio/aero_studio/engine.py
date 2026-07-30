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

"""Inference engine turning raw STL geometry into aerodynamic predictions.

The :class:`AeroPredictor` wraps a trained GeoTransolver (or Transolver)
surface model behind a single call: STL in, surface pressure / wall shear
stress / force coefficients (and optional MC-Dropout uncertainty) out.
It reuses the exact preprocessing from
:class:`physicsnemo.datapipes.cae.transolver_datapipe.TransolverDataPipe`
so predictions are consistent with the training recipe in
``examples/cfd/external_aerodynamics/transformer_models``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import hydra.utils
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from physicsnemo.datapipes.cae.transolver_datapipe import TransolverDataPipe
from physicsnemo.nn import ConcreteDropout, get_concrete_dropout_rates
from physicsnemo.utils import load_checkpoint

from .geometry import frontal_area, load_stl_surface

logger = logging.getLogger("aero_studio.engine")


@dataclass
class AeroPrediction:
    """Container for the results of a single aerodynamic evaluation.

    All per-cell arrays are aligned with the cells of the (triangulated)
    input mesh. Fields are dimensional (SI units) assuming the mesh is in
    meters, velocity in m/s, and density in kg/m^3.
    """

    # Per-cell surface fields:
    pressure: np.ndarray  # (N,) static pressure [Pa]
    wall_shear_stress: np.ndarray  # (N, 3) [Pa]

    # Optional per-cell MC-Dropout uncertainty (std across samples):
    pressure_std: np.ndarray | None = None
    wall_shear_stress_std: np.ndarray | None = None

    # Integrated quantities:
    drag_coefficient: float = 0.0
    lift_coefficient: float = 0.0
    drag_force: float = 0.0  # [N]
    lift_force: float = 0.0  # [N]
    drag_pressure_component: float = 0.0  # Cd contribution from pressure
    drag_friction_component: float = 0.0  # Cd contribution from friction
    # CdA-style raw force coefficients (F / (rho * U^2)), matching the
    # convention reported by the training/inference recipe:
    raw_drag_coefficient: float = 0.0
    raw_lift_coefficient: float = 0.0

    # Geometry / conditions:
    frontal_area: float = 0.0  # [m^2]
    total_surface_area: float = 0.0  # [m^2]
    n_cells: int = 0
    stream_velocity: float = 0.0
    air_density: float = 0.0

    # Bookkeeping:
    inference_seconds: float = 0.0
    mc_samples: int = 0
    model_name: str = ""
    trained: bool = True
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """Return the scalar results as a JSON-friendly dictionary."""
        return {
            "dragCoefficient": self.drag_coefficient,
            "liftCoefficient": self.lift_coefficient,
            "dragForceN": self.drag_force,
            "liftForceN": self.lift_force,
            "dragPressureComponent": self.drag_pressure_component,
            "dragFrictionComponent": self.drag_friction_component,
            "rawDragCoefficient": self.raw_drag_coefficient,
            "rawLiftCoefficient": self.raw_lift_coefficient,
            "frontalAreaM2": self.frontal_area,
            "totalSurfaceAreaM2": self.total_surface_area,
            "nCells": self.n_cells,
            "streamVelocity": self.stream_velocity,
            "airDensity": self.air_density,
            "inferenceSeconds": self.inference_seconds,
            "mcSamples": self.mc_samples,
            "modelName": self.model_name,
            "trained": self.trained,
            "meanUncertaintyPa": (
                float(np.mean(self.pressure_std))
                if self.pressure_std is not None
                else None
            ),
        }


class AeroPredictor:
    """STL-to-aerodynamics prediction engine backed by GeoTransolver.

    Parameters
    ----------
    cfg : DictConfig
        Application config (see ``conf/config.yaml``). Must contain a
        ``model`` node instantiable by hydra and a ``data`` node with the
        datapipe options used at training time.
    device : str | torch.device | None
        Compute device. Defaults to CUDA when available.
    """

    def __init__(
        self,
        cfg: DictConfig,
        device: str | torch.device | None = None,
    ) -> None:
        self.cfg = cfg
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = hydra.utils.instantiate(cfg.model)
        self.model_name = str(cfg.model.get("_target_", "model")).rsplit(".", 1)[-1]

        # The GeoTransolver path is selected when the datapipe emits a
        # geometry tensor; the plain Transolver path otherwise.
        self.uses_geometry = bool(cfg.data.get("include_geometry", False))

        self.trained = self._load_checkpoint()
        self.model.to(self.device)
        self.model.eval()

        self.surface_factors = self._load_normalization()
        self.datapipe = self._build_datapipe()

        self.chunk_size = int(cfg.inference.get("chunk_size", 200_000))
        self.n_output_fields = int(cfg.model.get("out_dim", 4))

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> bool:
        ckpt_dir = self.cfg.inference.get("checkpoint_dir", None)
        if ckpt_dir:
            ckpt_dir = Path(hydra.utils.to_absolute_path(str(ckpt_dir)))
            if ckpt_dir.exists() and any(ckpt_dir.iterdir()):
                epoch = load_checkpoint(
                    path=str(ckpt_dir), models=self.model, device=self.device
                )
                logger.info(f"Loaded checkpoint from {ckpt_dir} (epoch {epoch})")
                return True
            logger.warning(f"Checkpoint directory {ckpt_dir} is missing or empty.")

        if not self.cfg.inference.get("allow_untrained_model", False):
            raise FileNotFoundError(
                "No usable checkpoint found. Set inference.checkpoint_dir to a "
                "directory of checkpoints produced by the transformer_models "
                "training recipe, or set inference.allow_untrained_model=true "
                "to run the full pipeline with random weights (demo only)."
            )
        logger.warning(
            "Running with RANDOM (untrained) weights - predictions are "
            "meaningless. This mode only exercises the pipeline."
        )
        return False

    def _load_normalization(self) -> dict[str, torch.Tensor] | None:
        norm_file = self.cfg.data.get("normalization_file", None)
        if not norm_file:
            return None
        norm_file = Path(hydra.utils.to_absolute_path(str(norm_file)))
        if not norm_file.exists():
            raise FileNotFoundError(f"Normalization file not found: {norm_file}")
        norm_data = np.load(norm_file)
        factors = {
            "mean": torch.from_numpy(norm_data["mean"]).to(self.device),
            "std": torch.from_numpy(norm_data["std"]).to(self.device),
        }
        logger.info(f"Loaded surface normalization from {norm_file}")
        return factors

    def _build_datapipe(self) -> TransolverDataPipe:
        data_cfg = self.cfg.data
        overrides = {}
        for key in (
            "include_normals",
            "include_sdf",
            "broadcast_global_features",
            "include_geometry",
            "geometry_sampling",
            "translational_invariance",
            "reference_origin",
            "scale_invariance",
            "reference_scale",
        ):
            if data_cfg.get(key, None) is not None:
                value = data_cfg[key]
                overrides[key] = (
                    OmegaConf.to_container(value)
                    if isinstance(value, (DictConfig, list)) or OmegaConf.is_config(value)
                    else value
                )

        datapipe = TransolverDataPipe(
            input_path=None,
            model_type="surface",
            resolution=None,  # full mesh; we chunk manually
            surface_factors=self.surface_factors,
            scaling_type="mean_std_scaling" if self.surface_factors else None,
            return_mesh_features=True,
            **overrides,
        )
        if (
            datapipe.config.scale_invariance
            and datapipe.config.reference_scale is not None
        ):
            datapipe.config.reference_scale = datapipe.config.reference_scale.to(
                self.device
            )
        return datapipe

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _forward_chunks(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the model over the full point cloud in memory-sized chunks.

        Points are visited in a random order (chunk composition then
        matches the random subsampling seen during training) and results
        are scattered back to the original ordering.
        """
        embeddings = batch["embeddings"]
        fx = batch.get("fx", None)
        geometry = batch.get("geometry", None)

        n_points = embeddings.shape[1]
        indices = torch.randperm(n_points, device=embeddings.device)
        blocks = torch.split(indices, self.chunk_size)

        outputs = []
        for block in blocks:
            emb = embeddings[:, block]
            if fx is not None and fx.shape[1] == n_points:
                fx_block = fx[:, block]
            else:
                fx_block = fx

            if geometry is not None:
                pred = self.model(
                    local_embedding=emb,
                    local_positions=emb[:, :, :3],
                    global_embedding=fx_block,
                    geometry=geometry,
                )
            else:
                pred = self.model(fx=fx_block, embedding=emb)
            outputs.append(pred)

        stacked = torch.cat(outputs, dim=1)
        inverse = torch.empty_like(indices)
        inverse[indices] = torch.arange(indices.size(0), device=indices.device)
        return stacked[:, inverse]

    def _enable_mc_dropout(self) -> bool:
        """Put ConcreteDropout layers (if any) into stochastic mode."""
        self.model.eval()
        found = False
        for module in self.model.modules():
            if isinstance(module, ConcreteDropout):
                module.train()
                found = True
        if found:
            rates = list(get_concrete_dropout_rates(self.model).values())
            if rates:
                logger.info(
                    f"MC-Dropout active: learned rates "
                    f"min={min(rates):.4f} max={max(rates):.4f}"
                )
        return found

    @torch.no_grad()
    def predict(
        self,
        stl_path: str | Path,
        stream_velocity: float = 30.0,
        air_density: float = 1.205,
        mc_samples: int = 0,
    ) -> tuple[AeroPrediction, "pv.PolyData"]:  # noqa: F821
        """Predict surface aerodynamics for a geometry file.

        Parameters
        ----------
        stl_path : str | Path
            Path to the vehicle surface geometry (STL preferred).
        stream_velocity : float
            Free-stream velocity magnitude in m/s (flow along +x).
        air_density : float
            Free-stream air density in kg/m^3.
        mc_samples : int
            When > 0 and the model was trained with Concrete Dropout, run
            this many stochastic forward passes and report per-point
            uncertainty; the mean across passes is the prediction.

        Returns
        -------
        tuple[AeroPrediction, pv.PolyData]
            The prediction results and the triangulated input mesh with
            predicted fields attached as cell data.
        """
        start = time.time()

        data_dict, mesh = load_stl_surface(
            stl_path, device=self.device, n_output_fields=self.n_output_fields
        )
        data_dict["air_density"] = torch.tensor(
            float(air_density), device=self.device, dtype=torch.float32
        )
        data_dict["stream_velocity"] = torch.tensor(
            float(stream_velocity), device=self.device, dtype=torch.float32
        )

        # The geometry encoding samples STL vertices without replacement;
        # clamp the sample size for meshes coarser than the configured count.
        configured_sampling = self.datapipe.config.geometry_sampling
        n_stl_points = int(data_dict["stl_coordinates"].shape[0])
        if configured_sampling is not None and configured_sampling >= n_stl_points:
            self.datapipe.config.geometry_sampling = None
        try:
            batch = self.datapipe(data_dict)
        finally:
            self.datapipe.config.geometry_sampling = configured_sampling

        effective_mc = 0
        if mc_samples and mc_samples > 0:
            if self._enable_mc_dropout():
                effective_mc = int(mc_samples)
            else:
                logger.warning(
                    "mc_samples > 0 but the model has no ConcreteDropout "
                    "layers; falling back to a deterministic pass."
                )

        if effective_mc > 0:
            samples = [self._forward_chunks(batch) for _ in range(effective_mc)]
            stacked = torch.stack(samples, dim=0)
            preds_scaled = stacked.mean(dim=0)
            std_scaled = stacked.std(dim=0)
            self.model.eval()
        else:
            self.model.eval()
            preds_scaled = self._forward_chunks(batch)
            std_scaled = None

        # Undo the training normalization -> nondimensional p/(rho U^2), wss/(rho U^2)
        preds = self.datapipe.unscale_model_targets(preds_scaled, factor_type="surface")
        dynamic_scale = float(air_density) * float(stream_velocity) ** 2
        fields = (preds[0] * dynamic_scale).to(torch.float32)

        pressure = fields[:, 0]
        wss = fields[:, 1:4]

        pressure_std = None
        wss_std = None
        if std_scaled is not None:
            # std is invariant to the mean shift; scale by std factors only.
            if self.surface_factors is not None:
                std_fields = std_scaled[0] * self.surface_factors["std"]
            else:
                std_fields = std_scaled[0]
            std_fields = (std_fields * dynamic_scale).to(torch.float32)
            pressure_std = std_fields[:, 0]
            wss_std = std_fields[:, 1:4]

        normals = data_dict["surface_normals"]
        areas = data_dict["surface_areas"]

        result = self._integrate_forces(
            pressure=pressure,
            wss=wss,
            normals=normals,
            areas=areas,
            stream_velocity=float(stream_velocity),
            air_density=float(air_density),
        )

        result.pressure = pressure.cpu().numpy()
        result.wall_shear_stress = wss.cpu().numpy()
        if pressure_std is not None:
            result.pressure_std = pressure_std.cpu().numpy()
            result.wall_shear_stress_std = wss_std.cpu().numpy()

        result.n_cells = int(pressure.shape[0])
        result.inference_seconds = time.time() - start
        result.mc_samples = effective_mc
        result.model_name = self.model_name
        result.trained = self.trained

        # Attach fields to the mesh for export / visualization.
        mesh.cell_data["PredictedPressure"] = result.pressure
        mesh.cell_data["PredictedWallShearStress"] = result.wall_shear_stress
        if result.pressure_std is not None:
            mesh.cell_data["UncertaintyPressureStd"] = result.pressure_std
            mesh.cell_data["UncertaintyWallShearStressStd"] = (
                result.wall_shear_stress_std
            )

        return result, mesh

    def _integrate_forces(
        self,
        pressure: torch.Tensor,
        wss: torch.Tensor,
        normals: torch.Tensor,
        areas: torch.Tensor,
        stream_velocity: float,
        air_density: float,
    ) -> AeroPrediction:
        """Integrate surface fields into forces and coefficients.

        Follows the same sign conventions as the training recipe
        (``compute_force_coefficients`` in ``inference_on_zarr.py``):
        pressure force along ``d`` is ``sum((n . d) * A * p)`` and friction
        force is ``-sum((wss . d) * A)``.
        """
        drag_dir = torch.tensor(
            self.cfg.physics.get("drag_direction", [1.0, 0.0, 0.0]),
            device=pressure.device,
            dtype=pressure.dtype,
        )
        lift_dir = torch.tensor(
            self.cfg.physics.get("lift_direction", [0.0, 0.0, 1.0]),
            device=pressure.device,
            dtype=pressure.dtype,
        )

        def force(direction: torch.Tensor) -> tuple[float, float, float]:
            f_p = torch.sum((normals @ direction) * areas * pressure)
            f_f = -torch.sum((wss @ direction) * areas)
            return float(f_p + f_f), float(f_p), float(f_f)

        drag_total, drag_p, drag_f = force(drag_dir)
        lift_total, _, _ = force(lift_dir)

        a_frontal = frontal_area(normals, areas, tuple(drag_dir.tolist()))
        q = 0.5 * air_density * stream_velocity**2
        q_a = max(q * a_frontal, 1e-12)
        rho_u2 = max(air_density * stream_velocity**2, 1e-12)

        return AeroPrediction(
            pressure=np.empty(0),
            wall_shear_stress=np.empty(0),
            drag_coefficient=drag_total / q_a,
            lift_coefficient=lift_total / q_a,
            drag_force=drag_total,
            lift_force=lift_total,
            drag_pressure_component=drag_p / q_a,
            drag_friction_component=drag_f / q_a,
            raw_drag_coefficient=drag_total / rho_u2,
            raw_lift_coefficient=lift_total / rho_u2,
            frontal_area=a_frontal,
            total_surface_area=float(areas.sum()),
            stream_velocity=stream_velocity,
            air_density=air_density,
        )
