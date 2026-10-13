# RoadMC

[简体中文](README.md) | English

RoadMC is an experimental codebase for road-surface point-cloud research. It focuses on procedural road surfaces with point-wise labels, controlled inspection of synthetic geometry and material changes, and training entry points for point-cloud segmentation models. It also includes tools for a fixed real-domain evaluation protocol. Synthetic scenes support controlled experiments and implementation diagnostics; they do not replace real road data or establish real-road model performance.

## Capabilities and experimental scope

- NumPy/SciPy road-surface and distress scene generation for asphalt and concrete, with grid sampling and optional observation models.
- An optional three-layer representation (design, roughness, texture) and procedural texture generation with explicit parameters.
- Deterministic replay from a frozen `SurfaceTile` and explicit JSON plan, with controlled prototypes for cracks, potholes, raveling, and patches.
- PyTorch training entry points, real-point-cloud loading, road-level splitting, and tools to fit and freeze an evaluation protocol from real calibration samples.

The label table in [`labels.py`](roadmc/data/synthetic/labels.py) defines 38 classes including background; it does not mean every distress class has a complete, validated physical generator. Frozen replay currently supports only asphalt line cracks 5–8, potholes 9–10, raveling 11–12, and patches 20; and concrete slab cracks 23–24, pitting 34, and patches 37. It does not model asphalt alligator or block cracking, nor all taxonomy classes. New texture and replay parameters remain engineering hypotheses and have not been calibrated against qualifying millimeter-accuracy real surfaces. There are currently no qualifying real samples for domain calibration or independent acceptance, so the project has no reportable real-domain gap or claim of a “minimal domain gap.”

## Installation

Python 3.11 or later is required. The complete project dependencies, including PyTorch for training, are listed in [`pyproject.toml`](pyproject.toml). To run NumPy/SciPy synthesis and surface tools only, create a lightweight environment from the repository root in Windows PowerShell; the repository root can import `roadmc` directly, so the full project does not need to be installed:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install numpy scipy matplotlib
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
$env:ROADMC_GENERATOR_NO_TORCH = '1'
.venv\Scripts\python.exe -m roadmc.scripts.generate_synthetic --help
```

`ROADMC_GENERATOR_NO_TORCH=1` prevents the generator from importing PyTorch; it does not enable model training. For the full training environment, use `uv sync` (the PyTorch index configuration in `pyproject.toml` is used by uv); with pip, configure the PyTorch wheel source for the target platform before installing project dependencies. Optional dependencies for LAS/LAZ and PLY input are listed in the `real` extra; Open3D-related functions are in the `pcd` extra.

## Quick start: generate synthetic data

This PowerShell example generates a small asphalt dataset from the repository root and disables observation simulation. Grid spacing is specified in meters, but the general generation CLI defaults to `normalize=True`, so point coordinates in scene NPZ files are normalized to the unit ball by default and must not be treated directly as metric coordinates. Files include metadata such as `coordinates_normalized` to describe the transform; restore the recorded transform before metric evaluation. The dedicated `capture_surface_geometry` workflow fixes `normalize=False`, so its `SurfaceTile` and captured coordinates remain metric. Each scene is a compressed NPZ file, and the dataset root also contains `metadata.json`.

```powershell
$env:ROADMC_GENERATOR_NO_TORCH = '1'
python -m roadmc.scripts.generate_synthetic `
  --train-count 4 --val-count 2 `
  --output-dir output/readme-demo/synthetic `
  --seed 73 --surface-grid-spacing 0.02 `
  --sensor-output-points 2048 --width 3 --length 4 `
  --pavement asphalt --observation-model off
```

Scene NPZ files contain `points`, `labels`, `feats`, `normals`, and coordinate-transform and sampling-resolution fields; exact fields depend on the generator version and configuration. In `labels.py`, 0 is background, 1–20 are asphalt taxonomy labels, and 21–37 are concrete taxonomy labels. Taxonomy numbering does not guarantee that a class appears in a scene or that its synthetic shape has been validated against real data.

## Optional surface workflows

The commands below capture a purely synthetic three-layer surface, save a `SurfaceTile`, replace its texture layer using explicit settings, and run a two-material frozen replay. These values demonstrate the interface only; they are not measured ranges or recommended calibration values. Outputs are written to `output/readme-demo/`, and no real texture or distress patches are read or copied.

```powershell
$env:ROADMC_GENERATOR_NO_TORCH = '1'
python -m roadmc.scripts.capture_surface_geometry `
  --output-dir output/readme-demo/base --label 6 --seed 73 `
  --width 2 --length 1 --spacing 0.01 --surface-layers
```

The checked-in texture settings examples [`asphalt-texture-settings.json`](examples/realism/asphalt-texture-settings.json) and [`concrete-texture-settings.json`](examples/realism/concrete-texture-settings.json) specify every field; frequency is in cycles/m and RMS is in meters.

```powershell
python -m roadmc.scripts.generate_texture_surface `
  --base output/readme-demo/base/surface_tile.npz `
  --settings examples/realism/asphalt-texture-settings.json `
  --seed 73 --output output/readme-demo/asphalt-texture/surface_tile.npz

python -m roadmc.scripts.generate_texture_surface `
  --base output/readme-demo/base/surface_tile.npz `
  --settings examples/realism/concrete-texture-settings.json `
  --seed 73 --output output/readme-demo/concrete-texture/surface_tile.npz

python -m roadmc.scripts.replay_surface `
  --base output/readme-demo/asphalt-texture/surface_tile.npz `
  --plan examples/realism/asphalt-plan.json `
  --output output/readme-demo/replay/asphalt/replay.npz

python -m roadmc.scripts.replay_surface `
  --base output/readme-demo/concrete-texture/surface_tile.npz `
  --plan examples/realism/concrete-plan.json `
  --output output/readme-demo/replay/concrete/replay.npz
```

The two checked-in plans and texture settings are in [`examples/realism/`](examples/realism/); each plan uses labels valid for its material. The current asphalt and concrete texture settings differ only in the provenance `material_id`; their numeric values are equal and do not represent modeled or measured visual differences between the materials. Both workflows reuse the same design and roughness layers as a controlled common-base hypothesis. A plan must explicitly declare its material, road ID, events, seeds, parameters, labels, provenance, and hypotheses. Every event must include a `label` key; use `null` for unknown, which is saved as `-1` rather than background. Unknown base labels also remain `-1`. Plan parameters and exposed-surface textures are uncalibrated hypotheses.

Each replay NPZ stores `points`, `labels`, `normals`, `coordinates_normalized=False`, `metadata`, `plan_hash`, plan JSON, the final metric height field, material weights/texture, event footprints, and the base layers. Frozen replay currently omits the intensity channel. The quick generator writes `train/scene_*.npz`, `val/scene_*.npz`, and a root `metadata.json`; the surface example uses `base/`, `asphalt-texture/`, `concrete-texture/`, and `replay/<material>/` directories. This quick demo uses a 10 mm grid for an 8 mm asphalt crack event; that sampling cannot resolve millimeter-scale crack geometry and is not millimeter truth or accuracy validation.

The optional line-profiler observation supports only a single regular grid block with equal spacing in both axes. The small configuration example is [`examples/realism/line-profiler-config.json`](examples/realism/line-profiler-config.json); pass an explicit seed and configuration:

```powershell
python -m roadmc.scripts.replay_surface `
  --base output/readme-demo/asphalt-texture/surface_tile.npz `
  --plan examples/realism/asphalt-plan.json `
  --output output/readme-demo/replay/asphalt/line-profiler.npz `
  --observation line-profiler --observation-seed 11 `
  --observation-config examples/realism/line-profiler-config.json
```

Segmented sampling and cross-segment profile de-duplication are not implemented. `off` mode retains metric grid coordinates and does not require equal axis spacing.

## Fixed real-domain evaluation

The evaluation entry point is `roadmc.scripts.diagnose_domain_protocol`. It first splits by `road_segment_id`, fits and saves normalization and kernel bandwidth using only real calibration samples, then applies the frozen protocol to synthetic candidates and independent real roads. The sample manifest is a JSON object with a top-level `samples` array. Each entry provides a file `path`, `domain`, `role`, `road_segment_id`, and `material_id`. Real inputs also require verifiable metadata for coordinate units, quality, sampling spacing, measurement precision, and source. A road segment must not be split across groups.

For the synthetic output above, first inspect candidate coverage. The manifest example [`examples/realism/synthetic-candidate-manifest.json`](examples/realism/synthetic-candidate-manifest.json) resolves its path relative to that manifest and points to the asphalt replay NPZ generated above. It contains no verified real-data quality metadata and cannot be used for calibration. This diagnostic checks synthetic inputs and available descriptors only; it does not fit a real reference or perform acceptance:

```powershell
python -m roadmc.scripts.diagnose_domain_protocol inspect `
  --manifest examples/realism/synthetic-candidate-manifest.json `
  --output output/readme-demo/candidate-diagnostics.json
```

After real data are available and pass admission checks, use the following sequence to create a road-level split, fit a frozen protocol, and evaluate a manifest. `real-manifest.json`, the generated `split.json`, and `evaluation-manifest.json` must list actual real files with their correct roles; synthetic samples cannot stand in for real calibration, validation, or holdout data. The split requires at least 25 independent real roads to provide at least five roads each for calibration, real validation, and real holdout.

```powershell
python -m roadmc.scripts.diagnose_domain_protocol split `
  --manifest real-manifest.json --output split.json
python -m roadmc.scripts.diagnose_domain_protocol calibrate `
  --manifest split.json --output-protocol fixed-protocol.json
python -m roadmc.scripts.diagnose_domain_protocol evaluate `
  --manifest evaluation-manifest.json --protocol fixed-protocol.json `
  --output evaluation-report.json
```

There are currently no real scans that meet admission requirements. These calibration and evaluation commands therefore cannot produce a valid real-domain conclusion from the data currently available. The tools should report `not_calibrated` or `not_evaluated`; missing evidence is never treated as zero distance. The fixed protocol splits by road to prevent a segment from entering both calibration and independent acceptance. Data must first have verified units, sampling spacing, measurement precision, quality, and source. The minimum road counts here describe the tool's current input requirement; they do not substitute for evidence that data quality is acceptable.

## Model training

Training is separate from synthetic data generation and requires PyTorch and the training dependencies. `label_stage` offers `binary`, `four`, `eight`, and `full38` training mappings, but these options do not prove complete physical generation coverage for every label. The following short synthetic-data smoke run is not a model-quality evaluation:

```powershell
python roadmc/train.py baseline `
  --data_dir output/readme-demo/synthetic `
  --label_stage binary --max_epochs 1 --batch_size 2 `
  --max_points 2048 --num_workers 0
```

See `python roadmc/train.py --help` for training parameters, backbones, optimizers, checkpoints, and run-directory options. A small smoke run cannot establish generalization. Formal experiments need sufficient data, fixed splits, recorded configurations and random seeds, and evaluation on independent data.

## Repository layout and validation

- `roadmc/`: generator, data readers, models, training, and experiment scripts.
- `examples/realism/`: reusable explicit JSON plans, texture settings, observation configuration, and candidate manifest.
- `output/`: local experiment outputs, normally excluded from version control.

From the repository root, check script syntax and code style with:

```powershell
python -m py_compile roadmc/scripts/generate_synthetic.py roadmc/scripts/capture_surface_geometry.py roadmc/scripts/generate_texture_surface.py roadmc/scripts/replay_surface.py roadmc/scripts/diagnose_domain_protocol.py
python -m ruff check roadmc
```

Replay and evaluation support controlled procedural experiments and consistency checks only. They do not provide evidence of real-road naturalness, complete distress coverage, sensor calibration, or real-domain performance. Replay uses regular grid prototypes and does not implement complete crack topology, irregular pothole edges, multilayer pothole walls, or segmented sensor observation.

## License

This project is distributed under the [MIT License](LICENSE).
