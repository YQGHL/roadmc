"""Capture paired pre/post-disease metric surface geometry for diagnosis.

For small diagnostic scenes: captures retain full float64 arrays, even with
segmented generation, and are additional to the normal surface memory budget.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np


def main() -> None:
    from roadmc.data.synthetic.config import AmEnvelopeConfig, GeneratorConfig, RoadSurfaceConfig
    from roadmc.data.synthetic.generator import SyntheticRoadDataset
    from roadmc.data.synthetic.surface_tile import SurfaceTile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", type=int, required=True, help="force this JTG target label")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=float, default=3.0, help="road width in meters (default: 3)")
    parser.add_argument("--length", type=float, default=4.0, help="road length in meters (default: 4)")
    parser.add_argument("--spacing", type=float, default=0.005, help="surface grid spacing in meters")
    parser.add_argument("--max-surface-points", type=int, default=None,
                        help="per-tile point budget; may force segmentation, but capture remains in RAM")
    parser.add_argument("--render", action="store_true", help="render diagnostics after capture")
    parser.add_argument("--max-3d-points", type=int, default=100_000)
    parser.add_argument("--surface-layers", action="store_true",
                        help="enable the opt-in design/roughness/texture representation")
    parser.add_argument("--am-enable", action="store_true",
                        help="enable AM; in layered mode only the texture layer is transformed")
    parser.add_argument("--crossfall", type=float, default=0.02)
    parser.add_argument("--longitudinal-grade", type=float, default=0.0)
    args = parser.parse_args()

    road = RoadSurfaceConfig(width=args.width, length=args.length, grid_res=args.spacing,
                             crossfall=args.crossfall,
                             longitudinal_grade=args.longitudinal_grade)
    config = GeneratorConfig(
        road=road,
        seed=args.seed,
        observation_model="off",
        normalize=False,
        retain_all_surface_points=True,
        target_label_min_output_points=0,
        surface_layers=args.surface_layers,
        am_envelope=AmEnvelopeConfig(enable=args.am_enable),
        **({"max_surface_points": args.max_surface_points}
           if args.max_surface_points is not None else {}),
    )
    dataset = SyntheticRoadDataset(config, dataset_size=1)
    # capture_geometry is a diagnostic-only return path; geometry and disease
    # parameters continue to come from the project config and generator.
    scene = dataset.generate_scene(0, target_label=args.label, capture_geometry=True)
    if "geometry_diagnostics" not in scene:
        raise RuntimeError("generate_scene(capture_geometry=True) returned no geometry_diagnostics")
    geometry = scene["geometry_diagnostics"]
    required = ("base_points", "surface_points", "surface_labels", "surface_normals")
    absent = [key for key in required if key not in geometry]
    if absent:
        raise RuntimeError(f"geometry_diagnostics is missing: {', '.join(absent)}")

    metadata = {
        "config": asdict(config),
        "scene_index": 0,
        "target_label": args.label,
        "resolution_metadata": scene.get("resolution_metadata"),
        "geometry_metadata": geometry.get("metadata"),
        "capture": {"units": "meters", "road_only": True, "observation_model": "off",
                    "normalized": False},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / "surface_geometry_capture.npz"
    npz_fields = {key: np.asarray(geometry[key]) for key in required}
    tile = geometry.get("surface_tile")
    if tile is not None:
        npz_fields.update({
            "tile_x_m": tile["x"], "tile_y_m": tile["y"],
            "tile_design_m": tile["design_m"],
            "tile_roughness_m": tile["roughness_m"],
            "tile_texture_m": tile["texture_m"],
        })
        metadata["surface_tile"] = tile["metadata"]
        SurfaceTile(
            tile["x"], tile["y"], tile["design_m"], tile["roughness_m"],
            tile["texture_m"], tile["metadata"],
        ).save(args.output_dir / "surface_tile.npz")
    np.savez_compressed(
        npz_path, **npz_fields,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, default=str)),
    )
    print(f"Saved geometry capture: {npz_path}")
    if args.render:
        from roadmc.scripts.diagnose_surface_geometry import render

        summary = render(npz_path, args.output_dir, label=args.label,
                         max_3d_points=args.max_3d_points, seed=args.seed)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
