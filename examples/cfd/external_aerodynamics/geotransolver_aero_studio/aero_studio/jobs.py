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

"""In-process job queue for aerodynamic evaluations.

Evaluations run one at a time on a worker thread (the model typically owns
a single GPU), while the API stays responsive. Results are kept in memory
and exported artifacts (VTP files) are written under the server workdir.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("aero_studio.jobs")


@dataclass
class JobRecord:
    """State of a single evaluation job."""

    job_id: str
    filename: str
    stream_velocity: float
    air_density: float
    mc_samples: int
    status: str = "queued"  # queued | running | done | failed
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    summary: dict | None = None
    viz_payload: dict | None = None
    vtp_path: Path | None = None

    def public(self) -> dict:
        """JSON-friendly job state (without the heavy viz payload)."""
        return {
            "jobId": self.job_id,
            "filename": self.filename,
            "streamVelocity": self.stream_velocity,
            "airDensity": self.air_density,
            "mcSamples": self.mc_samples,
            "status": self.status,
            "createdAt": self.created_at,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "error": self.error,
            "summary": self.summary,
        }


class JobManager:
    """Runs evaluations sequentially on a background thread."""

    def __init__(self, workdir: Path) -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def submit(
        self,
        run_fn,
        filename: str,
        stl_path: Path,
        stream_velocity: float,
        air_density: float,
        mc_samples: int,
    ) -> JobRecord:
        """Queue an evaluation.

        ``run_fn(job, stl_path)`` performs the work and fills the record's
        ``summary`` / ``viz_payload`` / ``vtp_path`` fields.
        """
        job = JobRecord(
            job_id=uuid.uuid4().hex[:12],
            filename=filename,
            stream_velocity=stream_velocity,
            air_density=air_density,
            mc_samples=mc_samples,
        )
        with self._lock:
            self._jobs[job.job_id] = job

        def _run() -> None:
            job.status = "running"
            job.started_at = time.time()
            try:
                run_fn(job, stl_path)
                job.status = "done"
            except Exception as exc:  # surfaced through the API
                logger.error(f"Job {job.job_id} failed: {exc}")
                traceback.print_exc()
                job.status = "failed"
                job.error = str(exc)
            finally:
                job.finished_at = time.time()
                stl_path.unlink(missing_ok=True)

        self._executor.submit(_run)
        return job

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: -j.created_at)
        return [j.public() for j in jobs]
