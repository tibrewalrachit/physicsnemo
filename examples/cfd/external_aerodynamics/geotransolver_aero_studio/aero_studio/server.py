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

"""FastAPI application exposing the aero studio web UI and REST API.

Endpoints
---------
- ``GET  /``                          - interactive web UI
- ``GET  /api/v1/health``             - model / server info
- ``POST /api/v1/jobs``               - upload a geometry and queue a prediction
- ``GET  /api/v1/jobs``               - list jobs
- ``GET  /api/v1/jobs/{id}``          - job status + scalar results
- ``GET  /api/v1/jobs/{id}/surface``  - mesh + fields for the 3D viewer
- ``GET  /api/v1/jobs/{id}/result.vtp`` - download predictions as VTP

Launch with::

    python serve.py --config conf/config.yaml
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import DictConfig

from .engine import AeroPredictor
from .geometry import build_visualization_payload
from .jobs import JobManager, JobRecord

logger = logging.getLogger("aero_studio.server")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
ALLOWED_SUFFIXES = {".stl", ".vtp", ".obj", ".ply"}


def create_app(cfg: DictConfig, predictor: AeroPredictor | None = None) -> FastAPI:
    """Build the FastAPI app around a (possibly shared) predictor."""
    app = FastAPI(
        title="GeoTransolver Aero Studio",
        description=(
            "AI external-aerodynamics predictions from raw vehicle geometry, "
            "powered by NVIDIA PhysicsNeMo GeoTransolver."
        ),
        version="1.0.0",
    )

    if predictor is None:
        predictor = AeroPredictor(cfg)

    workdir = Path(cfg.server.get("workdir", "aero_studio_jobs"))
    manager = JobManager(workdir)
    upload_dir = workdir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    max_upload_bytes = int(cfg.server.get("max_upload_mb", 300)) * 1024 * 1024
    viz_max_cells = int(cfg.server.get("viz_max_cells", 120_000))

    def run_job(job: JobRecord, stl_path: Path) -> None:
        result, mesh = predictor.predict(
            stl_path,
            stream_velocity=job.stream_velocity,
            air_density=job.air_density,
            mc_samples=job.mc_samples,
        )
        job.summary = result.summary()

        vtp_path = workdir / f"{job.job_id}.vtp"
        mesh.save(str(vtp_path))
        job.vtp_path = vtp_path

        cell_fields = {
            "pressure": result.pressure,
            "wallShearStressMagnitude": np.linalg.norm(
                result.wall_shear_stress, axis=1
            ),
        }
        if result.pressure_std is not None:
            cell_fields["pressureUncertainty"] = result.pressure_std
        job.viz_payload = build_visualization_payload(
            mesh, cell_fields, max_cells=viz_max_cells
        )

    @app.get("/api/v1/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model": predictor.model_name,
            "trained": predictor.trained,
            "device": str(predictor.device),
            "defaults": {
                "streamVelocity": cfg.physics.get("default_stream_velocity", 30.0),
                "airDensity": cfg.physics.get("default_air_density", 1.205),
                "mcSamples": cfg.inference.get("default_mc_samples", 0),
            },
        }

    @app.post("/api/v1/jobs")
    async def create_job(
        file: UploadFile = File(...),
        stream_velocity: float = Form(None),
        air_density: float = Form(None),
        mc_samples: int = Form(None),
    ) -> JSONResponse:
        suffix = Path(file.filename or "geometry.stl").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{suffix}'. "
                f"Allowed: {sorted(ALLOWED_SUFFIXES)}",
            )

        contents = await file.read()
        if len(contents) > max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (> {max_upload_bytes // (1024 * 1024)} MB)",
            )
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        stl_path = upload_dir / f"{uuid.uuid4().hex}{suffix}"
        stl_path.write_bytes(contents)

        if stream_velocity is None:
            stream_velocity = float(cfg.physics.get("default_stream_velocity", 30.0))
        if air_density is None:
            air_density = float(cfg.physics.get("default_air_density", 1.205))
        if mc_samples is None:
            mc_samples = int(cfg.inference.get("default_mc_samples", 0))
        if stream_velocity <= 0 or air_density <= 0:
            raise HTTPException(
                status_code=400,
                detail="stream_velocity and air_density must be positive",
            )
        mc_samples = max(0, min(int(mc_samples), 100))

        job = manager.submit(
            run_job,
            filename=file.filename or stl_path.name,
            stl_path=stl_path,
            stream_velocity=float(stream_velocity),
            air_density=float(air_density),
            mc_samples=mc_samples,
        )
        return JSONResponse(job.public(), status_code=202)

    @app.get("/api/v1/jobs")
    def list_jobs() -> list[dict]:
        return manager.list()

    def _get_job_or_404(job_id: str) -> JobRecord:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id}")
        return job

    @app.get("/api/v1/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        return _get_job_or_404(job_id).public()

    @app.get("/api/v1/jobs/{job_id}/surface")
    def job_surface(job_id: str) -> dict:
        job = _get_job_or_404(job_id)
        if job.status != "done" or job.viz_payload is None:
            raise HTTPException(
                status_code=409, detail=f"Job {job_id} is not finished (status: {job.status})"
            )
        return job.viz_payload

    @app.get("/api/v1/jobs/{job_id}/result.vtp")
    def job_vtp(job_id: str) -> FileResponse:
        job = _get_job_or_404(job_id)
        if job.status != "done" or job.vtp_path is None or not job.vtp_path.exists():
            raise HTTPException(
                status_code=409, detail=f"Job {job_id} has no VTP result yet"
            )
        return FileResponse(
            str(job.vtp_path),
            media_type="application/octet-stream",
            filename=f"{Path(job.filename).stem}_prediction.vtp",
        )

    @app.get("/")
    def index() -> FileResponse:
        index_file = FRONTEND_DIR / "index.html"
        if not index_file.exists():
            raise HTTPException(status_code=404, detail="Frontend not found")
        return FileResponse(str(index_file), media_type="text/html")

    if FRONTEND_DIR.exists():
        app.mount(
            "/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static"
        )

    return app
