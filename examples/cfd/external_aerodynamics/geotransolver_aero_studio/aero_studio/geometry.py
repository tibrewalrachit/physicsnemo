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

"""Geometry ingestion utilities for the aero studio.

Reads raw STL (or any pyvista-readable surface) files and produces the
tensors expected by :class:`physicsnemo.datapipes.cae.transolver_datapipe.TransolverDataPipe`
in ``surface`` mode, plus helper quantities such as the projected frontal
area used to normalize force coefficients.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import torch


def read_surface_mesh(path: str | Path) -> pv.PolyData:
    """Read a surface mesh file and return it as triangulated PolyData.

    Parameters
    ----------
    path : str | Path
        Path to an STL / VTP / OBJ / PLY file readable by pyvista.

    Returns
    -------
    pv.PolyData
        Triangulated surface mesh.
    """
    mesh = pv.read(str(path))
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    mesh = mesh.triangulate()
    if mesh.n_cells == 0:
        raise ValueError(f"Mesh {path} contains no cells")
    return mesh


def load_stl_surface(
    path: str | Path,
    device: torch.device | str = "cpu",
    n_output_fields: int = 4,
) -> tuple[dict[str, torch.Tensor], pv.PolyData]:
    """Build the surface-mode data dictionary for the Transolver datapipe.

    The returned dictionary mirrors the layout produced by the DrivAerML
    preprocessing used for training: cell centers, unit cell normals, cell
    areas, plus the raw STL vertices/faces/centers used for the geometry
    encoding. Surface fields are filled with zeros since they are unknown
    at inference time.

    Parameters
    ----------
    path : str | Path
        Path to the surface geometry file.
    device : torch.device | str
        Device to place tensors on.
    n_output_fields : int
        Number of predicted surface fields (default 4: pressure + 3 wall
        shear stress components).

    Returns
    -------
    tuple[dict[str, torch.Tensor], pv.PolyData]
        ``(data_dict, mesh)`` where ``data_dict`` feeds the datapipe and
        ``mesh`` is the triangulated pyvista mesh (used later for export
        and visualization).
    """
    mesh = read_surface_mesh(path)

    device = torch.device(device)

    points = np.asarray(mesh.points, dtype=np.float32)
    faces = mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32)

    cell_centers = np.asarray(mesh.cell_centers().points, dtype=np.float32)

    normals = np.asarray(
        mesh.compute_normals(cell_normals=True, point_normals=False).cell_data[
            "Normals"
        ],
        dtype=np.float32,
    )
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

    sized = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    areas = np.asarray(sized.cell_data["Area"], dtype=np.float32)

    n_cells = cell_centers.shape[0]

    data_dict = {
        # Surface tensors (the model's input point cloud):
        "surface_mesh_centers": torch.from_numpy(cell_centers).to(device),
        "surface_normals": torch.from_numpy(normals).to(device),
        "surface_areas": torch.from_numpy(areas).to(device),
        "surface_fields": torch.zeros(
            (n_cells, n_output_fields), dtype=torch.float32, device=device
        ),
        # STL tensors (for the geometry encoding / center of mass):
        "stl_coordinates": torch.from_numpy(points).to(device),
        "stl_faces": torch.from_numpy(faces.flatten()).to(device),
        "stl_centers": torch.from_numpy(cell_centers).to(device),
    }

    return data_dict, mesh


def frontal_area(
    normals: torch.Tensor,
    areas: torch.Tensor,
    flow_direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> float:
    """Approximate the projected frontal area of a closed surface.

    For a watertight mesh, projecting every cell onto the plane normal to
    the flow direction counts the frontal silhouette twice (front-facing
    and back-facing cells), so half the summed absolute projection is a
    good estimate of the frontal area.

    Parameters
    ----------
    normals : torch.Tensor
        Unit cell normals of shape ``(N, 3)``.
    areas : torch.Tensor
        Cell areas of shape ``(N,)``.
    flow_direction : tuple[float, float, float]
        Free-stream flow direction (default +x, the DrivAerML convention).

    Returns
    -------
    float
        Estimated frontal area in the mesh's length units squared.
    """
    d = torch.tensor(flow_direction, dtype=normals.dtype, device=normals.device)
    d = d / torch.norm(d)
    projected = torch.abs(normals @ d) * areas
    return float(0.5 * projected.sum())


def build_visualization_payload(
    mesh: pv.PolyData,
    cell_fields: dict[str, np.ndarray],
    max_cells: int = 120_000,
) -> dict:
    """Build a compact JSON-serializable mesh payload for the web viewer.

    Cell-centered predictions are averaged to the mesh points so the viewer
    can render smoothly interpolated colors, and the mesh is decimated if
    it exceeds ``max_cells`` to keep the payload light.

    Parameters
    ----------
    mesh : pv.PolyData
        Triangulated surface mesh.
    cell_fields : dict[str, np.ndarray]
        Scalar fields defined on cells, shape ``(n_cells,)`` each.
    max_cells : int
        Decimation target for the viewer mesh.

    Returns
    -------
    dict
        Dictionary with ``positions`` (flat xyz list), ``indices`` (flat
        triangle indices), ``fields`` (per-vertex scalars), and per-field
        ``ranges``.
    """
    viz = mesh.copy(deep=True)
    viz.clear_data()
    for name, values in cell_fields.items():
        viz.cell_data[name] = np.asarray(values, dtype=np.float32)

    # Move scalars to points for smooth shading in the viewer.
    viz = viz.cell_data_to_point_data()

    if viz.n_cells > max_cells:
        reduction = 1.0 - max_cells / viz.n_cells
        try:
            viz = viz.decimate(reduction, attribute_error=True)
        except Exception:
            # Decimation is a nicety; fall back to the full mesh on failure.
            pass

    positions = np.asarray(viz.points, dtype=np.float32)
    indices = viz.faces.reshape(-1, 4)[:, 1:].astype(np.int64)

    fields = {}
    ranges = {}
    for name in cell_fields:
        values = np.asarray(viz.point_data[name], dtype=np.float32)
        fields[name] = values.tolist()
        finite = values[np.isfinite(values)]
        if finite.size:
            ranges[name] = [float(finite.min()), float(finite.max())]
        else:
            ranges[name] = [0.0, 0.0]

    bounds = viz.bounds
    return {
        "positions": positions.flatten().tolist(),
        "indices": indices.flatten().tolist(),
        "fields": fields,
        "ranges": ranges,
        "bounds": list(bounds),
        "nCells": int(viz.n_cells),
    }
