<!-- markdownlint-disable -->
# Chip Thermal Studio: SeaScape-Style AI Thermal Design Exploration

A self-contained replication of the workflow behind the
[Ansys / NVIDIA announcement](https://www.ansys.com/news-center/press-releases/11-19-24-ansys-integrates-nvidia-modulus-with-seascape)
that integrated **NVIDIA Modulus — now PhysicsNeMo, this repository —**
into the Ansys SeaScape cloud EDA platform: chip design teams train
customized AI surrogate models on **their own library of completed
high-fidelity thermal simulations** (RedHawk-SC style power/thermal
analyses), then use the trained "AI engine" for design exploration at
**100x+ the speed of the solver**, ranking floorplan candidates by
size/power/performance specifications.

This example reproduces that loop end to end with open components:

```
1. DESIGN LIBRARY                    2. TRAIN CUSTOM AI ENGINE
   random chip floorplans               GeoTransolver (structured 2D)
   + package parameters                 power map + [kt, h] -> ΔT field
   solved by a high-fidelity   ──►      trained on YOUR library
   FD thermal solver                            │
   (stand-in for RedHawk-SC)                    ▼
                                     3. DESIGN EXPLORATION
                                        sweep 1000s of candidates in seconds,
                                        rank by peak temperature, verify the
                                        winners against the solver
```

## The physics

Each "signed-off analysis" solves the steady compact thermal model of a
die attached to a package/heatsink stack:

```
-k_eff·t_die·∇²ΔT + h_eff·ΔT = q(x, y)
```

- `q(x, y)` — areal power density from the floorplan's power macros [W/m²]
- `k_eff·t_die` — effective lateral heat-spreading conductance [W/K]
- `h_eff` — effective through-package conductance to ambient [W/(m²K)]
- adiabatic lateral die edges; solved with conjugate gradients in PyTorch
  (`src/thermal.py`), exact on uniform-power sanity checks and
  energy-conserving to <0.1%.

The surrogate is a **GeoTransolver in structured 2D mode**
(`structured_shape=(g, g)`): local embedding = normalized power map +
(x, y) coordinates; global embedding = standardized `[k_eff·t_die, h_eff]`;
output = standardized temperature-rise field.

## Quick start

```bash
cd src

# 1. Build the design library (the "completed simulations")
python generate_dataset.py --n 2000 --out ../data/library.npz

# 2. Train the custom AI engine on the library
python train.py --config ../conf/config.yaml        # GPU
python train.py --config ../conf/config_demo.yaml   # small CPU demo

# 3. Explore the design space with the AI engine
python explore.py --checkpoint ../outputs/surrogate.pt \
    --candidates 2000 --top-k 5
```

`explore.py` sweeps floorplan candidates with the surrogate, ranks them
by predicted peak ΔT, re-solves the winners plus a random control set
with the high-fidelity solver, and reports:

- the measured **surrogate-vs-solver speedup** per design
- **peak-ΔT validation error** on the checked subset
- `../outputs/exploration.csv` — the full ranked candidate list
- `../outputs/exploration.png` — best floorplan, AI vs solver ΔT maps,
  error map, validation scatter, and the design-space histogram

Constraints are explorable through flags, e.g. a 150 W power budget:
`python explore.py --total-power 150 ...`.

## Mapping to the SeaScape workflow

| SeaScape / Modulus integration | This example |
|---|---|
| Library of completed RedHawk-SC thermal analyses | `generate_dataset.py` + FD solver (`thermal.py`) |
| "Train their AI models using their library of completed designs" | `train.py` (GeoTransolver, structured 2D) |
| "Customized … AI surrogate models" | model + normalization stats checkpointed per-library |
| "Use the newly created engine for more robust design exploration" | `explore.py` candidate sweep + ranking |
| "Identify optimal designs based on specifications (size, power, performance)" | spec-constrained candidate generation, peak-ΔT objective |
| "Over 100x speed-up for thermal simulations" | measured speedup printed by `explore.py` |

The same pattern extends to real data: replace `generate_dataset.py`
with an exporter from your thermal sign-off tool (power maps +
temperature fields on a regular grid) and retrain — nothing else
changes. For serving the trained engine interactively or on serverless
GPUs, see the companion
[`geotransolver_aero_studio`](../../cfd/external_aerodynamics/geotransolver_aero_studio)
example (FastAPI + browser GUI + Modal A100/H100 backend); its serving
pattern applies to this surrogate as-is.

## Files

```
chip_thermal_studio/
├── conf/
│   ├── config.yaml        # full training config (GPU)
│   └── config_demo.yaml   # small CPU demo config
├── src/
│   ├── thermal.py         # floorplan generator + FD thermal solver (CG)
│   ├── dataset.py         # library IO, normalization, input encoding
│   ├── generate_dataset.py
│   ├── train.py           # GeoTransolver structured-2D training
│   └── explore.py         # AI design-space sweep + solver validation
└── requirements.txt
```

## References

- Ansys press release: [Ansys Integrates NVIDIA Modulus with SeaScape](https://www.ansys.com/news-center/press-releases/11-19-24-ansys-integrates-nvidia-modulus-with-seascape)
- Transolver: [Wu et al., 2024](https://arxiv.org/abs/2402.02366); GeoTransolver: `physicsnemo.experimental.models.geotransolver`
- Compact thermal modeling of ICs: standard die/package RC reduction (e.g. HotSpot-style models)
