// SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/** Piecewise-linear colormaps. Each stop is [t, r, g, b] with rgb in 0-255. */
const STOPS = {
  turbo: [
    [0.0, 48, 18, 227], [0.125, 62, 108, 254], [0.25, 37, 199, 253],
    [0.375, 35, 235, 176], [0.5, 106, 253, 105], [0.625, 189, 235, 54],
    [0.75, 251, 185, 56], [0.875, 246, 96, 22], [1.0, 210, 31, 38],
  ],
  coolwarm: [
    [0.0, 59, 76, 192], [0.25, 124, 159, 249], [0.5, 221, 221, 221],
    [0.75, 245, 156, 125], [1.0, 180, 4, 38],
  ],
  viridis: [
    [0.0, 68, 1, 84], [0.25, 59, 82, 139], [0.5, 33, 145, 140],
    [0.75, 94, 201, 98], [1.0, 253, 231, 37],
  ],
  greys: [
    [0.0, 40, 48, 60], [1.0, 235, 240, 248],
  ],
};

export const COLORMAP_NAMES = Object.keys(STOPS);

/**
 * Sample a named colormap.
 * @param {string} name colormap name
 * @param {number} t position in [0, 1]
 * @returns {[number, number, number]} rgb in 0-1
 */
export function sample(name, t) {
  const stops = STOPS[name] ?? STOPS.turbo;
  t = Math.min(1, Math.max(0, t));
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const [t0, r0, g0, b0] = stops[i - 1];
      const [t1, r1, g1, b1] = stops[i];
      const f = t1 === t0 ? 0 : (t - t0) / (t1 - t0);
      return [
        (r0 + f * (r1 - r0)) / 255,
        (g0 + f * (g1 - g0)) / 255,
        (b0 + f * (b1 - b0)) / 255,
      ];
    }
  }
  const last = stops[stops.length - 1];
  return [last[1] / 255, last[2] / 255, last[3] / 255];
}

/** Paint a vertical colorbar (max at top) onto a canvas. */
export function paintColorbar(canvas, name) {
  const ctx = canvas.getContext('2d');
  for (let y = 0; y < canvas.height; y++) {
    const t = 1 - y / (canvas.height - 1);
    const [r, g, b] = sample(name, t);
    ctx.fillStyle = `rgb(${r * 255},${g * 255},${b * 255})`;
    ctx.fillRect(0, y, canvas.width, 1);
  }
}
