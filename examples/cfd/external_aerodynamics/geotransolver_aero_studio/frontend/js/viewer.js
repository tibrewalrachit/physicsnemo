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

/** Three.js scene manager for the aero studio viewer. */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { sample } from './colormaps.js';

export class AeroViewer {
  /**
   * @param {HTMLElement} container element the canvas is appended to
   * @param {(info: {value: number, x: number, y: number} | null) => void} onProbe
   *   called with the field value under the cursor (null to hide)
   */
  constructor(container, onProbe = null) {
    this.container = container;
    this.onProbe = onProbe;

    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x10141a);
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 5000);
    this.camera.position.set(6, 4, 6);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x223044, 1.1));
    const dir = new THREE.DirectionalLight(0xffffff, 1.4);
    dir.position.set(5, 10, 7);
    this.scene.add(dir);
    this.grid = new THREE.GridHelper(20, 20, 0x2b3648, 0x1e2632);
    this.scene.add(this.grid);

    this.mesh = null; // active result or preview mesh
    this.surface = null; // result payload (fields per vertex)
    this.fieldName = null;
    this.colormapName = 'turbo';
    this.range = null; // [min, max] currently applied

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this._probeActive = false;

    window.addEventListener('resize', () => this.resize());
    this.renderer.domElement.addEventListener('pointermove', (e) => this._pointerMove(e));
    this.renderer.domElement.addEventListener('pointerleave', () => {
      if (this.onProbe) this.onProbe(null);
    });
    this.resize();
    this._animate();
  }

  resize() {
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  _clearMesh() {
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      this.mesh.material.dispose();
      this.mesh = null;
    }
    this.surface = null;
    this.fieldName = null;
    this.range = null;
  }

  /**
   * Place a mesh in the scene: recenter the z-up CFD frame into the y-up
   * viewer frame, sit it on the grid, and frame the camera.
   */
  _install(geometry, material) {
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    const bb = geometry.boundingBox;
    const size = Math.max(
      bb.max.x - bb.min.x, bb.max.y - bb.min.y, bb.max.z - bb.min.z
    ) || 1;
    this.modelSize = size;

    const mesh = new THREE.Mesh(geometry, material);
    // z-up -> y-up: vertex (x, y, z) lands at (x, z, -y).
    mesh.rotation.x = -Math.PI / 2;
    const cx = (bb.min.x + bb.max.x) / 2;
    const cy = (bb.min.y + bb.max.y) / 2;
    mesh.position.set(-cx, -bb.min.z, cy);
    this.scene.add(mesh);
    this.mesh = mesh;

    const height = bb.max.z - bb.min.z;
    this.controls.target.set(0, height / 2, 0);
    this.camera.near = size / 1000;
    this.camera.far = size * 50;
    this.camera.updateProjectionMatrix();
    this.setCameraPreset('iso');

    const gridSize = Math.ceil(size * 2.5);
    this.scene.remove(this.grid);
    this.grid = new THREE.GridHelper(gridSize, gridSize, 0x2b3648, 0x1e2632);
    this.scene.add(this.grid);
  }

  /** Show an untextured preview of a local STL geometry (before prediction). */
  showPreview(geometry) {
    this._clearMesh();
    const material = new THREE.MeshStandardMaterial({
      color: 0x9fb2cc, metalness: 0.15, roughness: 0.65, side: THREE.DoubleSide,
    });
    this._install(geometry, material);
    this._probeActive = false;
  }

  /** Show a prediction surface payload from the API. */
  showSurface(surface) {
    this._clearMesh();
    this.surface = surface;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      'position', new THREE.BufferAttribute(new Float32Array(surface.positions), 3)
    );
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(surface.indices), 1));
    const material = new THREE.MeshStandardMaterial({
      vertexColors: true, metalness: 0.1, roughness: 0.75, side: THREE.DoubleSide,
    });
    this._install(geometry, material);
    this._probeActive = true;
  }

  /** Field names available on the current surface. */
  fieldNames() {
    return this.surface ? Object.keys(this.surface.fields) : [];
  }

  /** Auto range (from the payload) for a field. */
  autoRange(name) {
    return this.surface?.ranges?.[name] ?? [0, 1];
  }

  /**
   * Color the surface by a field.
   * @param {string} name field name
   * @param {[number, number] | null} range manual [min, max]; null for auto
   */
  showField(name, range = null) {
    if (!this.surface || !this.mesh) return null;
    const values = this.surface.fields[name];
    if (!values) return null;
    this.fieldName = name;
    const [min, max] = range ?? this.autoRange(name);
    this.range = [min, max];
    const span = max - min || 1;
    const colors = new Float32Array(values.length * 3);
    for (let i = 0; i < values.length; i++) {
      const [r, g, b] = sample(this.colormapName, (values[i] - min) / span);
      colors[3 * i] = r; colors[3 * i + 1] = g; colors[3 * i + 2] = b;
    }
    this.mesh.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    this.mesh.geometry.attributes.color.needsUpdate = true;
    return this.range;
  }

  setColormap(name) {
    this.colormapName = name;
    if (this.fieldName) this.showField(this.fieldName, this.range);
  }

  setWireframe(on) {
    if (this.mesh) this.mesh.material.wireframe = on;
  }

  /** Camera presets in the viewer (y-up) frame; car nose faces -x on screen. */
  setCameraPreset(preset) {
    const s = this.modelSize || 6;
    const t = this.controls.target;
    const pos = {
      iso: [s * 1.15, s * 0.72, s * 1.15],
      front: [-s * 1.7, s * 0.35, 0],
      rear: [s * 1.7, s * 0.35, 0],
      side: [0, s * 0.35, s * 1.7],
      top: [0.001, s * 2.0, 0.001],
    }[preset] ?? [s * 1.15, s * 0.72, s * 1.15];
    this.camera.position.set(t.x + pos[0], t.y + pos[1], t.z + pos[2]);
    this.camera.lookAt(t);
  }

  /** Download the current view as a PNG. */
  screenshot(filename = 'aero_studio.png') {
    const url = this.renderer.domElement.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
  }

  _pointerMove(event) {
    if (!this._probeActive || !this.onProbe || !this.mesh || !this.surface) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObject(this.mesh, false);
    if (!hits.length || !this.fieldName) {
      this.onProbe(null);
      return;
    }
    const hit = hits[0];
    const values = this.surface.fields[this.fieldName];
    // Nearest vertex of the hit face.
    const posAttr = this.mesh.geometry.attributes.position;
    const local = this.mesh.worldToLocal(hit.point.clone());
    let best = hit.face.a, bestD = Infinity;
    for (const vi of [hit.face.a, hit.face.b, hit.face.c]) {
      const dx = posAttr.getX(vi) - local.x;
      const dy = posAttr.getY(vi) - local.y;
      const dz = posAttr.getZ(vi) - local.z;
      const d = dx * dx + dy * dy + dz * dz;
      if (d < bestD) { bestD = d; best = vi; }
    }
    this.onProbe({
      value: values[best],
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    });
  }
}
