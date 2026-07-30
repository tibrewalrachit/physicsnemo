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

"""Configuration loading for the aero studio."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = APP_ROOT / "conf" / "config.yaml"


def load_app_config(config_path: str | Path | None = None) -> DictConfig:
    """Load the application config, resolving relative paths.

    ``data.normalization_file`` and ``inference.checkpoint_dir`` may be
    given relative to the config file's directory; they are rewritten to
    absolute paths so the engine and server can run from any working
    directory.

    Parameters
    ----------
    config_path : str | Path | None
        Path to a YAML config. Defaults to ``conf/config.yaml`` next to
        this package.

    Returns
    -------
    DictConfig
        The resolved application configuration.
    """
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    base = config_path.resolve().parent

    def _absolutize(node: str, key: str) -> None:
        value = OmegaConf.select(cfg, f"{node}.{key}")
        if value:
            p = Path(str(value))
            if not p.is_absolute():
                candidate = (base / p).resolve()
                # Fall back to CWD-relative if nothing exists next to the config.
                if candidate.exists() or not (Path.cwd() / p).exists():
                    p = candidate
                else:
                    p = (Path.cwd() / p).resolve()
            OmegaConf.update(cfg, f"{node}.{key}", str(p))

    _absolutize("data", "normalization_file")
    _absolutize("inference", "checkpoint_dir")

    return cfg
