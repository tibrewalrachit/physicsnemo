<!-- markdownlint-disable -->
# GeoTransolver Aero Studio

An interactive, end-to-end **AI external-aerodynamics prediction tool** built on
the PhysicsNeMo [GeoTransolver](../../../../physicsnemo/experimental/models/geotransolver)
model — in the spirit of recent "upload a CAD file, get aerodynamics in seconds"
products (e.g. physics-transformer web services for vehicle aero).

Upload a watertight vehicle STL and, in seconds, get back:

- **Surface pressure** and **wall shear stress** fields on every cell
- **Integrated quantities**: drag / lift coefficients (Cd, Cl), pressure vs.
  friction drag split, forces in Newtons, and estimated frontal area
- **Per-point uncertainty** via MC-Dropout (for checkpoints trained with
  Concrete Dropout)
- An **interactive 3D viewer** with field selection and colormaps
- **VTP export** of all predicted fields for ParaView / downstream tooling

Everything is also available programmatically through a **REST API** and a
**batch CLI** for design-space sweeps over thousands of geometry variants.

```
   STL upload ──► TransolverDataPipe ──► GeoTransolver ──► unscale ──► fields
      (web UI,      (same preprocessing     (surface        │
       REST API,     as training: normals,   pressure +     ├──► Cd / Cl / forces
       CLI)          translation & scale     wall shear     ├──► 3D viewer payload
                     invariance, geometry    stress)        └──► VTP export
                     encoding)
```

## Relationship to the training recipe

This tool is the *serving* counterpart of the training recipe in
[`../transformer_models`](../transformer_models). It reuses:

- `physicsnemo.experimental.models.geotransolver.GeoTransolver` — the model
- `physicsnemo.datapipes.cae.transolver_datapipe.TransolverDataPipe` — the exact
  preprocessing used during training (embedding construction, translational /
  scale invariance, geometry encoding)
- The same normalization statistics (`surface_fields_normalization.npz`) and
  force-coefficient conventions (flow along +x, lift along +z — the
  [DrivAerML](https://caemldatasets.org/drivaerml/) convention)

Any surface checkpoint trained with `transformer_models`
(`--config-name geotransolver_surface`) can be dropped in unchanged.

## Quick start

### 1. Install

```bash
pip install nvidia-physicsnemo  # or an editable install of this repo
pip install -r requirements.txt
```

### 2. Get a trained model

Train a surface GeoTransolver on DrivAerML with the
[`transformer_models` recipe](../transformer_models/README.md):

```bash
cd ../transformer_models/src
python train.py --config-name geotransolver_surface
```

This produces checkpoints under `runs/<run_id>/checkpoints`. For per-point
uncertainty, train with `model.concrete_dropout=true training.lambda_reg=1e-4`.

### 3. Configure

Edit `conf/config.yaml`:

```yaml
inference:
  checkpoint_dir: /path/to/runs/geotransolver/surface/bq/checkpoints
```

The `model:` and `data:` sections must match the training configuration —
the defaults mirror `geotransolver_surface.yaml`. The bundled
`conf/surface_fields_normalization.npz` matches the default DrivAerML recipe;
if you recomputed normalizations for your dataset, point
`data.normalization_file` at your file.

### 4. Launch the web studio

```bash
python serve.py --config conf/config.yaml
# then open http://localhost:8000
```

The browser GUI has two views:

**Studio** — drag an STL into the browser and get an *instant local 3D
preview* (triangle count, bounding box, and a warning if the extent doesn't
look like a meters-scaled vehicle) before any upload. Set velocity /
density / MC samples and press **Run prediction**. The viewer then renders
the predicted fields (pressure, wall shear stress magnitude, and — with
MC-Dropout — per-point uncertainty) with:

- colormap selection (turbo / coolwarm / viridis / greys) and a live colorbar
- manual or automatic color-range control
- camera presets (iso / front / rear / side / top), wireframe toggle
- a hover probe that reads out the field value under the cursor
- one-click PNG screenshots and full-resolution VTP download

The summary panel shows Cd, Cl, forces, frontal area, the
pressure-vs-friction drag split, and mean uncertainty.

**Compare designs** — every finished job in the session side by side: a
sortable results table (lowest-drag design highlighted) plus Cd / Cl bar
charts, for ranking geometry variants. Clicking a row reopens that design
in the Studio.

> **No GPU / no checkpoint?** `python serve.py --config conf/config_demo.yaml`
> runs a tiny **untrained** model on CPU so you can exercise the entire
> pipeline and UI. Predictions in this mode are meaningless, and the UI
> displays an `UNTRAINED (demo)` badge.

### 5. Optional: serverless GPU backend on Modal (A100 / H100)

If your workstation has no GPU, run inference on a serverless A100 or H100
via [Modal](https://modal.com) while the web studio / CLI stay local. Only
geometry bytes go up; predicted fields come back.

```bash
pip install modal "python-socks[asyncio]"
modal token set --token-id <your-id> --token-secret <your-secret>

# Deploy the GPU service (defaults: A100, conf/config_modal.yaml):
modal deploy modal_app.py
# ... or on an H100 / 80GB A100:
AERO_STUDIO_GPU=H100 modal deploy modal_app.py
AERO_STUDIO_GPU=A100-80GB modal deploy modal_app.py

# Upload your trained checkpoint to the persistent volume (once):
modal volume put aero-studio-checkpoints /path/to/checkpoints /geotransolver
```

Then flip the local config to the remote backend and launch as usual:

```yaml
inference:
  backend: modal   # was: local
```

```bash
python serve.py --config conf/config.yaml   # UI unchanged; inference on A100/H100
```

The remote container is configured by `conf/config_modal.yaml` (selected at
deploy time via `AERO_STUDIO_CONFIG`); its model/data sections must match
your checkpoint, exactly like the local config. The studio's health badge
shows the remote device (e.g. `modal/A100 (NVIDIA A100-SXM4-40GB)`).
A checkpoint-free demo deployment for pipeline testing:

```bash
AERO_STUDIO_MODAL_APP=geotransolver-aero-studio-demo \
AERO_STUDIO_CONFIG=conf/config_modal_demo.yaml modal deploy modal_app.py
```

with `modal.app_name: geotransolver-aero-studio-demo` in the local config.
You can also smoke-test a deployment directly:
`modal run modal_app.py --stl path/to/car.stl`. Containers stay warm for
5 minutes after a request (`scaledown_window`), so sweeps don't pay the
cold-start cost per design.

### 6. Batch sweeps from the CLI

```bash
python cli.py designs/*.stl \
    --config conf/config.yaml \
    --velocity 30.0 --density 1.205 \
    --csv sweep_results.csv --output-dir predictions/
```

Evaluates every geometry, prints Cd / Cl per design, writes a CSV summary for
ranking variants, and optionally exports per-geometry VTP files.

### 7. REST API

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/v1/health` | Model, device, trained flag, defaults |
| `POST` | `/api/v1/jobs` | Multipart upload (`file`, `stream_velocity`, `air_density`, `mc_samples`) → job id |
| `GET`  | `/api/v1/jobs` | List jobs |
| `GET`  | `/api/v1/jobs/{id}` | Status + scalar summary (Cd, Cl, forces, timing) |
| `GET`  | `/api/v1/jobs/{id}/surface` | Decimated mesh + per-vertex fields for visualization |
| `GET`  | `/api/v1/jobs/{id}/result.vtp` | Full-resolution predictions as VTP |

Example programmatic evaluation:

```python
import requests

with open("suv_variant_042.stl", "rb") as f:
    job = requests.post(
        "http://localhost:8000/api/v1/jobs",
        files={"file": f},
        data={"stream_velocity": 35.0, "air_density": 1.205},
    ).json()

# ... poll ...
result = requests.get(f"http://localhost:8000/api/v1/jobs/{job['jobId']}").json()
print(result["summary"]["dragCoefficient"])
```

## How the physics is computed

- **Fields.** The model predicts normalized `[p, τx, τy, τz]` per surface cell;
  predictions are un-standardized with the training statistics and scaled by
  `ρ U²` to physical units (the training targets are nondimensionalized by
  `ρ U²`).
- **Forces.** Using cell areas `A` and unit normals `n` (from the uploaded
  mesh), the force along direction `d` is
  `F = Σ (n·d) A p − Σ (τ·d) A`, matching `compute_force_coefficients` in the
  training recipe.
- **Coefficients.** `Cd = F_drag / (½ ρ U² A_frontal)` with the frontal area
  estimated as `½ Σ |n·x̂| A` (the closed-surface silhouette projection). The
  recipe-convention raw coefficients `F / (ρ U²)` are also reported for direct
  comparison with `inference_on_zarr.py` outputs.
- **Uncertainty.** With `mc_samples > 0` and a Concrete-Dropout checkpoint,
  the tool runs N stochastic passes; the mean is the prediction and the
  standard deviation is surfaced per point and in the summary.

## Notes and caveats

- Input meshes should be **watertight, single-solid, meters-scaled** vehicle
  surfaces oriented with the flow along +x and up along +z (the DrivAerML
  convention the model was trained under). Out-of-convention inputs will
  silently degrade accuracy.
- Like any data-driven surrogate, accuracy degrades away from the training
  distribution. Consider the UQ options (MC-Dropout here; the GP head and OOD
  guard in `transformer_models`) before trusting predictions on unusual
  designs.
- The web GUI is fully self-contained: `three.js` (MIT-licensed) is vendored
  under `frontend/vendor/`, so the studio works offline and behind
  restrictive proxies.
- The job queue is in-memory and single-worker (one GPU); restart clears
  history. For production serving, put the engine behind your own scheduler.

## Files

```
geotransolver_aero_studio/
├── aero_studio/
│   ├── engine.py      # AeroPredictor: STL -> fields -> coefficients (+ UQ)
│   ├── geometry.py    # STL ingestion, frontal area, viewer payloads
│   ├── server.py      # FastAPI app (REST + static frontend)
│   ├── jobs.py        # background job queue
│   └── config.py      # config loading/path resolution
├── aero_studio/remote.py  # Modal GPU backend client (backend: modal)
├── modal_app.py       # serverless A100/H100 deployment (modal deploy)
├── conf/
│   ├── config.yaml    # production config (point at your checkpoint)
│   ├── config_demo.yaml  # tiny untrained CPU demo
│   ├── config_modal.yaml # remote-side config for the Modal container
│   ├── config_modal_demo.yaml # ... demo variant (untrained fallback)
│   └── surface_fields_normalization.npz
├── frontend/          # browser GUI (self-contained, vendored three.js)
│   ├── index.html     #   layout: Studio + Compare views
│   ├── css/app.css    #   styling
│   ├── js/app.js      #   UI logic, jobs, compare charts
│   ├── js/viewer.js   #   three.js scene, fields, probe, camera presets
│   ├── js/colormaps.js
│   └── vendor/        #   three.js r160 (MIT)
├── serve.py           # web server entry point
├── cli.py             # batch evaluation CLI
└── requirements.txt
```

## References

- **Transolver:** [Wu et al., 2024](https://arxiv.org/abs/2402.02366)
- **GeoTransolver / GALE attention:** see
  `physicsnemo.experimental.models.geotransolver`
- **DrivAerML dataset:** [caemldatasets.org/drivaerml](https://caemldatasets.org/drivaerml/)
- **Concrete Dropout:** [Gal, Hron & Kendall, 2017](https://arxiv.org/abs/1705.07832)
