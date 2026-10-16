"""Replay a frozen SurfaceTile from an explicit JSON plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roadmc.data.synthetic.config import LineProfilerConfig
from roadmc.data.synthetic.surface_replay import replay_surface, save_replay
from roadmc.data.synthetic.surface_tile import SurfaceTile


def _demo_plans() -> dict[str, dict]:
    common = {"schema_version": 1, "road_id": "B1-frozen-demo", "provenance": {
        "source": "B1 saved procedural SurfaceTile; synthetic only", "calibration": "none"},
        "hypothesis": {"status": "engineering hypothesis; uncalibrated", "material_model":
        "procedural exposed texture; no copied real patches"}}
    asphalt = {**common, "material": "asphalt", "events": [
        {"instance_id": "crack-a", "type": "crack", "seed": 173,
         "label": 6, "parameters": {"center_x_m": .8, "center_y_m": .45,
         "start_x_m": .55, "start_y_m": .25, "end_x_m": 1.2, "end_y_m": .78,
         "width_m": .008, "depth_m": .004, "exposed_rms_m": .0002, "severity": "severe"}},
        {"instance_id": "ravel-b", "type": "raveling", "seed": 239,
         "label": 11, "parameters": {"center_x_m": 1.25, "center_y_m": .62,
         "radius_x_m": .28, "radius_y_m": .2, "corrlen_m": .025,
         "depth_m": .002, "threshold": .3, "exposed_rms_m": .0003}},
        {"instance_id": "patch-c", "type": "patch", "seed": 307,
         "label": 20, "parameters": {"center_x_m": .8, "center_y_m": .45,
         "radius_x_m": .15, "radius_y_m": .12, "thickness_m": .001,
         "exposed_rms_m": .00015}}]}
    concrete = {**common, "material": "concrete", "events": [
        {"instance_id": "slab-crack-a", "type": "crack", "seed": 419,
         "label": 23, "parameters": {"center_x_m": .9, "center_y_m": .5,
         "start_x_m": .9, "start_y_m": .18, "end_x_m": .9, "end_y_m": .82,
         "width_m": .006, "depth_m": .015, "exposed_rms_m": .0002, "severity": "light"}},
        {"instance_id": "pitting-b", "type": "pothole", "seed": 523,
         "label": 34, "parameters": {"center_x_m": 1.3, "center_y_m": .55,
         "radius_m": .12, "depth_m": .008, "exposed_rms_m": .0004}},
        {"instance_id": "patch-c", "type": "patch", "seed": 631,
         "label": 37, "parameters": {"center_x_m": .9, "center_y_m": .5,
         "radius_x_m": .13, "radius_y_m": .11, "thickness_m": .001,
         "exposed_rms_m": .00015}}]}
    return {"asphalt": asphalt, "concrete": concrete}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, help="frozen SurfaceTile NPZ")
    parser.add_argument("--plan", type=Path, help="explicit replay plan JSON")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--observation", choices=("off", "line-profiler"), default="off")
    parser.add_argument("--observation-noise", choices=("on", "off"), default="on",
                        help="v2 line-profiler noise model; source-grid holes remain missing")
    parser.add_argument("--observation-seed", type=int)
    parser.add_argument("--observation-config", type=Path,
                        help="JSON object of LineProfilerConfig overrides")
    parser.add_argument("--demo", action="store_true", help="write and replay asphalt/concrete synthetic B1 plans")
    args = parser.parse_args()
    if args.observation_config and args.observation != "line-profiler":
        parser.error("--observation-config requires --observation line-profiler")
    cfg = None
    if args.observation_config:
        cfg_data = json.loads(args.observation_config.read_text(encoding="utf-8"))
        if not isinstance(cfg_data, dict):
            parser.error("observation config must be a JSON object")
        cfg = LineProfilerConfig(**cfg_data)
    root = Path(__file__).resolve().parents[2]
    if args.demo:
        base = args.base or root / "output/realism_batch_b1/layers_grade/surface_tile.npz"
        target = args.output or root / "output/realism_replay"
        tile = SurfaceTile.load(base)
        for material, plan in _demo_plans().items():
            folder = target / material
            folder.mkdir(parents=True, exist_ok=True)
            plan_path = folder / "plan.json"
            plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
            result = replay_surface(tile, plan, observation=args.observation,
                                    observation_seed=args.observation_seed,
                                    line_profiler_config=cfg,
                                    observation_noise_enabled=args.observation_noise == "on")
            save_replay(folder / "replay.npz", tile, plan, result)
            print(f"{material}: plan_hash={result.metadata['plan_hash']} output={folder}")
        return
    if not args.base or not args.plan or not args.output:
        parser.error("--base, --plan and --output are required unless --demo is used")
    tile = SurfaceTile.load(args.base)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema_version") == 1 and args.observation_noise == "off":
        parser.error("--observation-noise off is only supported for schema_version=2 plans")
    result = replay_surface(tile, plan, observation=args.observation,
                            observation_seed=args.observation_seed,
                            line_profiler_config=cfg,
                            observation_noise_enabled=args.observation_noise == "on")
    save_replay(args.output, tile, plan, result)
    print(f"plan_hash={result.metadata['plan_hash']} output={args.output}")


if __name__ == "__main__":
    main()
