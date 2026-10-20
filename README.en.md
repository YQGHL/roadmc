# RoadMC

[简体中文](README.md) | English

RoadMC is an experimental codebase for road-surface point-cloud research. It focuses on procedural road surfaces with point-wise labels, controlled inspection of synthetic geometry and material changes, and training entry points for point-cloud segmentation models. It also includes tools for a fixed real-domain evaluation protocol. Synthetic scenes support controlled experiments and implementation diagnostics; they do not replace real road data or establish real-road model performance.

## Capabilities and experimental scope

- NumPy/SciPy road-surface and distress scene generation for asphalt and concrete, with grid sampling and optional observation models.
- An optional three-layer representation (design, roughness, texture) and procedural texture generation with explicit parameters.
- Deterministic replay from a frozen `SurfaceTile` and explicit JSON plan, with controlled prototypes for cracks, potholes, raveling, and patches.
- PyTorch training entry points, real-point-cloud loading, road-level splitting, and tools to fit and freeze an evaluation protocol from real calibration samples.

The label table in [`labels.py`](roadmc/data/synthetic/labels.py) defines 38 classes including background; it does not mean every distress class has a complete, validated physical generator. Frozen replay v1 supports asphalt line cracks, potholes, raveling and patches, plus concrete slab cracks, pitting and patches. V2 is an asphalt prototype with explicit line/alligator/block crack graphs and ordered material replacement; it does not cover the full taxonomy. Public texture matrices and stereo point clouds support limited morphology comparisons, but unresolved units, instrument metadata or road independence prevent real-domain calibration and independent acceptance. The project has no reportable claim of a “minimal domain gap.”

## Installation

Python 3.11 or later is required. The complete project dependencies, including PyTorch for training, are listed in [`pyproject.toml`](pyproject.toml). To run NumPy/SciPy synthesis and surface tools only, create a lightweight environment from the repository root in Windows PowerShell; the repository root can import `roadmc` directly, so the full project does not need to be installed:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install numpy scipy matplotlib pillow
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
$env:PYTHONIOENCODING = 'utf-8'
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

The line-profiler currently combines height-field interpolation with engineering noise. Beam-footprint integration, optical triangulation and geometric self-occlusion are not implemented. Sampling spacing does not establish instrument accuracy; the incidence parameter affects a noise proxy only. The sampling audit records these limits. Production preserves labels at sampled XY and interpolated latent surface normals; intensity and curvature use nearest XY support so Z noise does not change surface ownership.

## Optional sampling and texture-shape experiments

These diagnostics keep production defaults unchanged. The resolution experiment saves one 0.5 mm master surface and compares source grids of 5, 2, 1, and 0.5 mm under two scan spacings. Coarse-grid aliasing is an intentional negative control; distress geometry is not silently smoothed.

```powershell
python -m roadmc.scripts.diagnose_sampling_resolution --output-dir output/resolution-check
python -m roadmc.scripts.calibrate_texture_shape `
  --archive path/to/rsta20220166_si_001.zip --output-dir output/texture-shape-check
```

Output directories must be empty or absent for the resolution experiment, and absent for texture calibration. The default 1 m surface has about four million source points, with additional diagnostic memory overhead. Zero height error at sampled coordinates does not prove alias-free reconstruction; inspect both directional frequencies and sampling bandwidths.

Texture experiments use the released Royal Society DAC/MS/OGFC/SMA asphalt matrices, freeze the training/file-holdout split, and exclude explored files and content duplicates. Spectrum selection and non-Gaussian target moments use training data only. Outputs include holdout PSD, autocorrelation, standardized W1, and achieved moments. XY spacing follows the paper's 0.5 mm grid; CSV Z units and processing stage remain unresolved. This is normalized morphology comparison, not absolute-height calibration or independent-road validation.

The [v2 material example](examples/surface_replay_v2_materials.json) adds optional elliptical potholes and `retain`, `iid`, or `procedural` textures for potholes, raveling, and patches. Use either `contour` or legacy `radius_m`, and either `texture` or legacy `exposed_rms_m`. Its base tile must cover x=[-0.6, 0.6] m and y=[-0.45, 0.45] m. Retaining the current texture is a control option. V2 remains asphalt-only and uncalibrated. Valid legacy v1 geometry and RNG behavior are preserved; corrected line-profiler interpolation changes observed results.

Explicit texture settings can replace `hurst` with a `spectrum_profile` containing `frequency_cycles_m` and `relative_psd` arrays. This generates fresh random surfaces from statistical spectra, without measured phases or pasted patches. Non-Gaussian settings optionally accept `spectral_matching={"iterations":50,"tolerance":1e-5}` with actual iterations and marginal residuals recorded. These are numerical settings. The final projection prioritizes generated Fourier amplitudes; exact marginal matching is not guaranteed. The current file benchmark improves spectral/autocorrelation errors but does not beat the prior non-Gaussian candidate in W1, so defaults remain unchanged.

V2 potholes optionally accept `profile={"kind":"tilted_power_bowl","edge_power":1.0,"tilt_major":0.5,"tilt_minor":0.0}` to vary the wall profile and deepest-point position within the same footprint. These example numbers are hypotheses. Depth is normalized by the continuous analytic peak, not by a grid-dependent sampled maximum. V2 metadata also integrates area, volume and final ownership of the latent material-loss field. Patches reset earlier loss; texture relief and hidden layers are excluded. These integrals are simulator quantities, not real measured loss.

The following commands require explicit input paths and a nonexistent output directory. The texture comparison reuses a frozen split; previously examined test files are a seen-file benchmark. The pothole comparison tests spatial-block interpolation within each view, not independent-pothole generalization.

```powershell
python -m roadmc.scripts.benchmark_texture_envelopes `
  --archive path/to/rsta20220166_si_001.zip `
  --split-manifest output/texture-shape-check/split_manifest.json `
  --prior-config output/texture-shape-check/fitted_configurations.json `
  --output-dir output/texture-envelope-check

python -m roadmc.scripts.diagnose_pothole_profiles `
  --dataset-root path/to/dataset3 --output-dir output/pothole-profile-check
```

## Reproducible realism studies

The unified runner verifies admitted data roles, units, exposure and file hashes, then runs training-internal texture selection, a frozen benchmark, geometry/material controls and native-unit pothole diagnostics. It saves executing source, seeds, stage logs and artifact hashes. See the [workflow](examples/realism/study-workflow.md) and [numerical protocol](examples/realism/study-protocol.json). Keep raw data and large experiment outputs outside the repository.

```powershell
python -m roadmc.scripts.run_realism_study `
  --manifest path/to/verified-manifest.json `
  --protocol examples/realism/study-protocol.json `
  --output-dir path/to/new-study
```

Append `--resume` to the same command to verify and reuse completed stages. Changed inputs, source, dependency stages or artifacts invalidate reuse. `--stages admission` checks the manifest alone; see `--help` for other stages. New texture candidates are recommended only when all three main internal-CV metrics are no worse than the previous candidate; otherwise selection falls back. Previously seen TEST files are a frozen benchmark. Recommendations do not automatically change production defaults.

A v2 elliptical `contour` may include `radial_harmonics={"cos":[0.08,-0.04,0.03],"sin":[-0.03,0.02,0.01]}` for irregular edges using orders 3, 4 and 5, while preserving continuous ellipse area. Loss, footprint and material updates use the same contour. These coefficients are engineering hypotheses, not measured distributions; absent or all-zero coefficients retain the previous contour.

Additional explicit options include the frozen-result `pothole_outline_fairness` stage, v2 wall/floor `material_partition`, and native study imports preserving unknown units and auxiliary columns. See the [engineering extension guide (Chinese)](examples/realism/engineering-extensions.md) for commands and combination checks. Native import bundles grant neither metric training input nor independent real-road acceptance.

Dedicated readers and bounded comparisons are available for KIT normal concrete, SUD native LAS fields, and SHREC video; see the [new-data study guide](examples/realism/new-data-study.md). ELA profile units are explicit, while inferred PLY scale, unknown SUD class semantics, and compressed SHREC disparity retain separate admission limits. Optional concrete statistical-spectrum candidates do not replace production defaults or constitute independent-road validation.

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

Replay and evaluation support controlled procedural experiments, explicit crack graphs and consistency checks. Limited public-data morphology comparisons do not establish independent real-domain acceptance. V2 now has an optional irregular-edge pothole prototype. The height field cannot represent multilayer walls, overhangs or volumetric crack interiors; segmented sensor observation, material-state calibration and sensor calibration remain incomplete.

## Development data sources and attribution

We acknowledge the authors and maintainers of the public datasets below. Development used limited subsets for texture descriptions, native-field inspection, contours, or observation-support diagnostics. This does not imply completed training, field calibration, or independent real-domain evaluation.

| Source | Actual development role and limitation |
|---|---|
| [Pan et al., asphalt texture supplementary matrices](https://doi.org/10.1098/rsta.2022.0166) | Normalized texture/spectrum comparisons; released Z units/processing unresolved; previously inspected TEST files are not blind holdouts |
| [Fan et al., stereo pothole datasets](https://github.com/ruirangerfan/stereo_pothole_datasets) | Limited dataset3 views for boundaries, missing XYZ and coordinate diagnostics; not independent laser ground truth |
| [KIT concrete road measurements](https://doi.org/10.35097/ku8qs1jt347zxyeq) | Normal-surface ELA/PLY and repeat-scan comparisons; not distress-label ground truth |
| [SUT-Crack](https://doi.org/10.17632/gsbmknrhkv.1) | Local 130 masks and 3 original images for 2D exploration; not 130 complete segmentation pairs |
| [SUD-ROAD](https://github.com/msqiu/SUD-Road) | Native LAS codes/local geometry; local byte-code mapping and physical units remain unresolved |
| [Pothole Mix / SHREC 2022 V2](https://doi.org/10.17632/kfth5g2xk3.2) | Limited RGB/disparity/automatic video-mask contours; not metric 3D manual ground truth |
| [Laboratory texture/friction](https://doi.org/10.17632/kkcztnxzph.1), [field asphalt scans](https://doi.org/10.5281/zenodo.20606458), [paired MLS scans](https://doi.org/10.5281/zenodo.19267189) | Selected-sample descriptions, format and observation-support diagnostics; not full-dataset use or calibrated damage/material evolution |

Additional selected surfaces, the M2S-RoAD subsample, and incompletely traced sources are documented in [DATASETS.md](docs/DATASETS.md); importable citations are in [datasets.bib](docs/datasets.bib). Cite the original data records and author-requested papers when using data or reporting derived results, and retain version, license, processing and split records. This project's MIT license covers its code, not third-party datasets. This notice does not redistribute raw data.

## Code and research status

Production generation/replay and research prototypes are maintained separately. This branch contains the distress and native-data diagnostic code; check the actual merge status of PR [#15](https://github.com/YQGHL/roadmc/pull/15) before assuming it is available on the default branch. H1 material history, occupancy cavities and A/B/C system comparisons remain isolated research components outside this repository, not native default-generator or replay-schema functionality. See [research status and integration boundaries](docs/RESEARCH_STATUS.md).

## License

This project is distributed under the [MIT License](LICENSE).
