# synthetic/ — Physics-Based Data Generation

## OVERVIEW

11 physical primitives → configurable .npz scenes following JTG 5210-2018 (38 disease labels).

## STRUCTURE

```
synthetic/
├── config.py      # GeneratorConfig + 12 sub-configs (RoadSurface, Crack, Pothole, Rutting...)
├── primitives.py  # 11 functions: generate_road_surface, add_crack, add_pothole, add_rutting...
└── generator.py   # SyntheticRoadDataset — 10-step pipeline orchestrator
```

## WHERE TO LOOK

| Task | File | Detail |
|------|------|--------|
| Add disease type | `config.py` → `LABEL_MAP` + `primitives.py` → new function + `generator.py` → register |
| Change road surface | `primitives.py` → `generate_road_surface()` | ISO 8608 PSD + inverse FFT |
| Modify LiDAR noise | `config.py` → `LidarNoiseConfig` + `primitives.py` → `simulate_lidar_noise()` |
| Adjust grid resolution | `config.py` → `RoadSurfaceConfig.grid_res` | Default 0.01m (10mm) |

## CONVENTIONS

- All primitives take `(points, labels, ...)` and return modified copies
- `numpy` only (no torch in this module)
- `rng: Optional[np.random.Generator]` for seed control
- Disease severity: `"light"` or `"severe"` strings
- Label priority: background(0) < disease types, later diseases override earlier
- `.npz` keys: `points`, `labels`, `feats`, `normals`, `pavement_type`

## ANTI-PATTERNS

- **DO NOT reorder disease application** — Priority system (P1-3) controls override order
- **DO NOT skip NaN filtering** — P0-3 fix: remove NaN before KDTree label transfer
- **DO NOT skip 3σ distance threshold** — P0-4 fix: KDTree bounded transfer
