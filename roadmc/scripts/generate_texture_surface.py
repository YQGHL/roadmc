"""Replace the texture layer of a saved SurfaceTile using explicit JSON settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from roadmc.data.synthetic.procedural_texture import generate_texture, load_settings_json
from roadmc.data.synthetic.surface_tile import SurfaceTile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="input SurfaceTile NPZ")
    parser.add_argument("--settings", type=Path, required=True, help="strict JSON settings")
    parser.add_argument("--seed", type=int, required=True, help="non-negative random seed")
    parser.add_argument("--output", type=Path, required=True, help="output SurfaceTile NPZ")
    args = parser.parse_args()

    try:
        settings: dict[str, Any] = load_settings_json(args.settings)
        tile = SurfaceTile.load(args.base)
        texture, metadata = generate_texture(tile.x, tile.y, settings, args.seed)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    tile.texture = texture
    tile.metadata["procedural_texture"] = metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tile.save(args.output)
    print(json.dumps({"output": str(args.output), "metadata": metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
