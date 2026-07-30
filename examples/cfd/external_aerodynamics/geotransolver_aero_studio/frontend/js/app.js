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

/** GeoTransolver Aero Studio - browser GUI logic. */

import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { AeroViewer } from './viewer.js';
import { COLORMAP_NAMES, paintColorbar } from './colormaps.js';

const $ = (id) => document.getElementById(id);
const api = (p) => `/api/v1${p}`;

const FIELD_META = {
  pressure: { label: 'Pressure', unit: 'Pa' },
  wallShearStressMagnitude: { label: 'Wall shear stress |τ|', unit: 'Pa' },
  pressureUncertainty: { label: 'Pressure uncertainty (σ)', unit: 'Pa' },
};

// ---------------------------------------------------------------------------
// Viewer + probe
// ---------------------------------------------------------------------------

const probeEl = $('probe');
const viewer = new AeroViewer($('viewer'), (info) => {
  if (!info) { probeEl.style.display = 'none'; return; }
  const unit = FIELD_META[viewer.fieldName]?.unit ?? '';
  probeEl.textContent = `${fmt(info.value)} ${unit}`;
  probeEl.style.left = `${info.x}px`;
  probeEl.style.top = `${info.y}px`;
  probeEl.style.display = 'block';
});

function fmt(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  const a = Math.abs(v);
  if (a >= 10000) return v.toExponential(2);
  if (a >= 100) return v.toFixed(1);
  return v.toPrecision(3);
}

function updateColorbar() {
  if (!viewer.fieldName || !viewer.range) return;
  paintColorbar($('cb-canvas'), viewer.colormapName);
  const unit = FIELD_META[viewer.fieldName]?.unit ?? '';
  $('cb-max').textContent = fmt(viewer.range[1]);
  $('cb-min').textContent = fmt(viewer.range[0]);
  $('cb-unit').textContent = unit;
  $('colorbar').style.display = 'flex';
}

// ---------------------------------------------------------------------------
// Toolbar
// ---------------------------------------------------------------------------

const fieldSelect = $('field-select');
const cmapSelect = $('cmap-select');
for (const name of COLORMAP_NAMES) {
  const opt = document.createElement('option');
  opt.value = name;
  opt.textContent = name;
  cmapSelect.appendChild(opt);
}

function applyField(name, manualRange = null) {
  const range = viewer.showField(name, manualRange);
  if (!range) return;
  $('range-min').value = range[0].toPrecision(4);
  $('range-max').value = range[1].toPrecision(4);
  updateColorbar();
}

fieldSelect.addEventListener('change', () => applyField(fieldSelect.value));
cmapSelect.addEventListener('change', () => {
  viewer.setColormap(cmapSelect.value);
  updateColorbar();
});
$('range-apply').addEventListener('click', () => {
  const lo = parseFloat($('range-min').value);
  const hi = parseFloat($('range-max').value);
  if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) {
    applyField(viewer.fieldName ?? fieldSelect.value, [lo, hi]);
  }
});
$('range-auto').addEventListener('click', () => applyField(viewer.fieldName ?? fieldSelect.value));
$('wireframe').addEventListener('click', (e) => {
  const on = !e.target.classList.contains('on');
  e.target.classList.toggle('on', on);
  viewer.setWireframe(on);
});
for (const preset of ['iso', 'front', 'rear', 'side', 'top']) {
  $(`cam-${preset}`).addEventListener('click', () => viewer.setCameraPreset(preset));
}
$('screenshot').addEventListener('click', () => {
  viewer.screenshot(`${(currentFile?.name ?? 'aero_studio').replace(/\.[^.]+$/, '')}_view.png`);
});

// ---------------------------------------------------------------------------
// Upload + geometry preview
// ---------------------------------------------------------------------------

let currentFile = null;
const dz = $('dropzone');
dz.addEventListener('click', () => $('file-input').click());
dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('drag'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
dz.addEventListener('drop', (e) => {
  e.preventDefault(); dz.classList.remove('drag');
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});
$('file-input').addEventListener('change', (e) => {
  if (e.target.files.length) setFile(e.target.files[0]);
});

async function setFile(file) {
  currentFile = file;
  $('drop-label').innerHTML =
    `<span class="file">${file.name}</span><br/>${(file.size / 1e6).toFixed(1)} MB`;
  $('run').disabled = false;
  $('geom-info').textContent = '';
  $('geom-warn').style.display = 'none';

  // Instant local preview for STL files.
  if (/\.stl$/i.test(file.name)) {
    try {
      const buffer = await file.arrayBuffer();
      const geometry = new STLLoader().parse(buffer);
      geometry.computeBoundingBox();
      const bb = geometry.boundingBox;
      const dims = [bb.max.x - bb.min.x, bb.max.y - bb.min.y, bb.max.z - bb.min.z];
      const nTri = geometry.attributes.position.count / 3;
      viewer.showPreview(geometry);
      $('placeholder').style.display = 'none';
      $('toolbar').classList.add('hidden');
      $('colorbar').style.display = 'none';
      $('viewer-note').textContent = 'geometry preview — run a prediction to see fields';
      $('viewer-note').style.display = 'block';
      $('geom-info').innerHTML =
        `<b>${nTri.toLocaleString()}</b> triangles &middot; ` +
        `extent <b>${dims.map((d) => d.toFixed(2)).join(' × ')}</b>`;
      const maxDim = Math.max(...dims);
      if (maxDim > 100 || maxDim < 0.5) {
        $('geom-warn').textContent =
          'Extent looks unusual for a vehicle in meters — the model expects ' +
          'meter-scaled geometry with flow along +x.';
        $('geom-warn').style.display = 'block';
      }
    } catch (err) {
      $('geom-info').textContent = `Could not preview STL: ${err.message}`;
    }
  }
}

// ---------------------------------------------------------------------------
// Run prediction + jobs
// ---------------------------------------------------------------------------

function setStatus(msg, err = false) {
  const el = $('status');
  el.textContent = msg;
  el.className = err ? 'error' : '';
}

$('run').addEventListener('click', async () => {
  if (!currentFile) return;
  $('run').disabled = true;
  setStatus('Uploading geometry…');
  const fd = new FormData();
  fd.append('file', currentFile);
  fd.append('stream_velocity', $('velocity').value);
  fd.append('air_density', $('density').value);
  fd.append('mc_samples', $('mc').value);
  try {
    const res = await fetch(api('/jobs'), { method: 'POST', body: fd });
    if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText);
    const job = await res.json();
    $('spinner').style.display = 'flex';
    $('spinner-text').textContent = 'Running GeoTransolver inference…';
    setStatus(`Job ${job.jobId} queued.`);
    pollJob(job.jobId);
  } catch (err) {
    setStatus(`Upload failed: ${err.message}`, true);
  } finally {
    $('run').disabled = false;
    refreshJobs();
  }
});

async function pollJob(jobId) {
  try {
    const res = await fetch(api(`/jobs/${jobId}`));
    const job = await res.json();
    if (job.status === 'done') {
      $('spinner').style.display = 'none';
      setStatus(`Finished in ${job.summary.inferenceSeconds.toFixed(2)} s.`);
      await loadJob(job);
      refreshJobs();
      return;
    }
    if (job.status === 'failed') {
      $('spinner').style.display = 'none';
      setStatus(`Job failed: ${job.error}`, true);
      refreshJobs();
      return;
    }
    setStatus(`Job ${jobId}: ${job.status}…`);
    setTimeout(() => pollJob(jobId), 1000);
  } catch (err) {
    $('spinner').style.display = 'none';
    setStatus(`Polling failed: ${err.message}`, true);
  }
}

async function loadJob(job) {
  showSummary(job);
  const res = await fetch(api(`/jobs/${job.jobId}/surface`));
  if (!res.ok) return;
  viewer.showSurface(await res.json());
  $('placeholder').style.display = 'none';
  $('toolbar').classList.remove('hidden');
  $('viewer-note').style.display = 'none';

  fieldSelect.innerHTML = '';
  for (const name of viewer.fieldNames()) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = FIELD_META[name]?.label ?? name;
    fieldSelect.appendChild(opt);
  }
  applyField(fieldSelect.value);
}

function showSummary(job) {
  const s = job.summary;
  $('results-card').style.display = 'block';
  $('cd').textContent = s.dragCoefficient.toFixed(4);
  $('cl').textContent = s.liftCoefficient.toFixed(4);
  $('fd').textContent = s.dragForceN.toFixed(1);
  $('fl').textContent = s.liftForceN.toFixed(1);
  $('af').textContent = s.frontalAreaM2.toFixed(3);
  $('tm').textContent = s.inferenceSeconds.toFixed(2);
  const pPct = s.dragCoefficient
    ? (100 * s.dragPressureComponent / s.dragCoefficient).toFixed(0)
    : '–';
  $('drag-split').innerHTML =
    `Drag split: <b>${pPct}%</b> pressure / <b>${100 - pPct}%</b> friction` +
    (s.meanUncertaintyPa != null
      ? ` &middot; mean σ <b>${fmt(s.meanUncertaintyPa)} Pa</b> (${s.mcSamples} MC samples)`
      : '');
  const dl = $('download');
  dl.style.display = 'block';
  dl.onclick = () => { window.location.href = api(`/jobs/${job.jobId}/result.vtp`); };
}

// ---------------------------------------------------------------------------
// Job list + compare dashboard
// ---------------------------------------------------------------------------

let jobsCache = [];
let sortKey = 'createdAt';
let sortAsc = false;

async function refreshJobs() {
  try {
    jobsCache = await (await fetch(api('/jobs'))).json();
  } catch {
    return; // server unreachable; keep last known state
  }
  renderJobList();
  renderCompare();
}

function renderJobList() {
  const el = $('joblist');
  if (!jobsCache.length) { el.textContent = 'No jobs yet.'; return; }
  el.innerHTML = '';
  for (const j of jobsCache.slice(0, 12)) {
    const row = document.createElement('div');
    row.className = 'jobrow' + (j.status === 'done' ? ' clickable' : '');
    const cd = j.summary ? j.summary.dragCoefficient.toFixed(4) : '';
    row.innerHTML =
      `<span class="nm" title="${j.filename}">${j.filename}</span>` +
      `<span class="cd">${cd}</span>` +
      `<span class="st-${j.status}">${j.status}</span>`;
    if (j.status === 'done') row.onclick = () => { switchTab('studio'); loadJob(j); };
    el.appendChild(row);
  }
}

const COMPARE_COLUMNS = [
  ['filename', 'Geometry'],
  ['streamVelocity', 'U (m/s)'],
  ['dragCoefficient', 'Cd'],
  ['liftCoefficient', 'Cl'],
  ['dragForceN', 'Drag (N)'],
  ['liftForceN', 'Lift (N)'],
  ['frontalAreaM2', 'A front (m²)'],
  ['inferenceSeconds', 'Time (s)'],
];

function jobValue(job, key) {
  if (key === 'filename') return job.filename;
  if (key === 'streamVelocity') return job.streamVelocity;
  return job.summary?.[key];
}

function renderCompare() {
  const done = jobsCache.filter((j) => j.status === 'done' && j.summary);
  $('compare-empty').style.display = done.length ? 'none' : 'block';
  $('compare-content').style.display = done.length ? 'block' : 'none';
  if (!done.length) return;

  const sorted = [...done].sort((a, b) => {
    const va = jobValue(a, sortKey), vb = jobValue(b, sortKey);
    const c = typeof va === 'string' ? va.localeCompare(vb) : (va ?? 0) - (vb ?? 0);
    return sortAsc ? c : -c;
  });

  const minCd = Math.min(...done.map((j) => j.summary.dragCoefficient));
  const thead = COMPARE_COLUMNS
    .map(([key, label]) =>
      `<th data-key="${key}">${label}${sortKey === key ? (sortAsc ? ' ▲' : ' ▼') : ''}</th>`)
    .join('');
  const rows = sorted.map((j) => {
    const cells = COMPARE_COLUMNS.map(([key]) => {
      const v = jobValue(j, key);
      const shown = typeof v === 'number' ? (key === 'dragCoefficient' || key === 'liftCoefficient' ? v.toFixed(4) : fmt(v)) : v;
      const best = key === 'dragCoefficient' && v === minCd && done.length > 1;
      return `<td${best ? ' class="best"' : ''}>${shown}</td>`;
    }).join('');
    return `<tr class="selectable" data-job="${j.jobId}">${cells}</tr>`;
  }).join('');
  $('compare-table').innerHTML =
    `<thead><tr>${thead}</tr></thead><tbody>${rows}</tbody>`;

  for (const th of $('compare-table').querySelectorAll('th')) {
    th.onclick = () => {
      const key = th.dataset.key;
      if (sortKey === key) sortAsc = !sortAsc;
      else { sortKey = key; sortAsc = key === 'filename'; }
      renderCompare();
    };
  }
  for (const tr of $('compare-table').querySelectorAll('tr.selectable')) {
    tr.onclick = () => {
      const job = done.find((j) => j.jobId === tr.dataset.job);
      if (job) { switchTab('studio'); loadJob(job); }
    };
  }

  drawBarChart($('chart-cd'), sorted, 'dragCoefficient', '#76b900');
  drawBarChart($('chart-cl'), sorted, 'liftCoefficient', '#4fc3f7');
}

function drawBarChart(canvas, jobs, key, color) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const values = jobs.map((j) => j.summary[key]);
  const labels = jobs.map((j) => j.filename.replace(/\.[^.]+$/, ''));
  const maxAbs = Math.max(...values.map(Math.abs), 1e-9);
  const labelW = 120, pad = 10, valueW = 62, axisX = labelW + 6;
  const plotW = w - axisX - pad - valueW;
  const zeroX = axisX + plotW * (Math.min(...values, 0) < 0 ? 0.5 : 0);
  const scale = (plotW - (zeroX - axisX)) / maxAbs;
  const rowH = Math.min(34, (h - 2 * pad) / values.length);

  ctx.font = '11px "Segoe UI", sans-serif';
  ctx.textBaseline = 'middle';
  values.forEach((v, i) => {
    const y = pad + i * rowH + rowH / 2;
    ctx.fillStyle = '#8fa0b8';
    ctx.textAlign = 'right';
    ctx.fillText(labels[i].slice(0, 18), labelW, y);
    const barW = Math.abs(v) * scale;
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.85;
    ctx.fillRect(v >= 0 ? zeroX : zeroX - barW, y - rowH * 0.32, Math.max(barW, 1), rowH * 0.64);
    ctx.globalAlpha = 1;
    ctx.textAlign = 'left';
    ctx.fillStyle = '#e7edf6';
    ctx.fillText(v.toFixed(4), (v >= 0 ? zeroX + barW : zeroX) + 6, y);
  });
  ctx.strokeStyle = '#2b3648';
  ctx.beginPath();
  ctx.moveTo(zeroX, pad / 2);
  ctx.lineTo(zeroX, h - pad / 2);
  ctx.stroke();
}

// ---------------------------------------------------------------------------
// Tabs + boot
// ---------------------------------------------------------------------------

function switchTab(name) {
  $('tab-studio').classList.toggle('active', name === 'studio');
  $('tab-compare').classList.toggle('active', name === 'compare');
  $('studio-view').classList.toggle('hidden', name !== 'studio');
  $('compare-view').classList.toggle('hidden', name !== 'compare');
  if (name === 'studio') viewer.resize();
  if (name === 'compare') renderCompare();
}
$('tab-studio').addEventListener('click', () => switchTab('studio'));
$('tab-compare').addEventListener('click', () => switchTab('compare'));

(async function init() {
  try {
    const h = await (await fetch(api('/health'))).json();
    const badge = $('model-badge');
    badge.textContent = h.trained
      ? `${h.model} · ${h.device}`
      : `${h.model} · UNTRAINED (demo)`;
    badge.classList.toggle('untrained', !h.trained);
    $('velocity').value = h.defaults.streamVelocity;
    $('density').value = h.defaults.airDensity;
    $('mc').value = h.defaults.mcSamples;
  } catch {
    $('model-badge').textContent = 'server unreachable';
  }
  await refreshJobs();
  // Restore the most recent finished job, if any.
  const latest = jobsCache.find((j) => j.status === 'done');
  if (latest) loadJob(latest);
})();
