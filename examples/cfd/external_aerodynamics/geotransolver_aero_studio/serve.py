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

"""Launch the GeoTransolver Aero Studio web server.

Usage::

    python serve.py --config conf/config.yaml
    python serve.py --config conf/config_demo.yaml --port 8080
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from aero_studio.config import load_app_config
from aero_studio.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=None, help="Path to the app config YAML"
    )
    parser.add_argument("--host", default=None, help="Override server host")
    parser.add_argument("--port", type=int, default=None, help="Override server port")
    parser.add_argument(
        "--device", default=None, help="Torch device override (e.g. cuda:0, cpu)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cfg = load_app_config(args.config)

    from aero_studio.engine import AeroPredictor

    predictor = AeroPredictor(cfg, device=args.device)
    app = create_app(cfg, predictor=predictor)

    host = args.host or cfg.server.get("host", "0.0.0.0")
    port = args.port or int(cfg.server.get("port", 8000))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
