"""Small, explicit layered representation of a generated road surface."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SurfaceTile:
    """Three co-registered height layers in metres, stored x-major."""

    x: np.ndarray
    y: np.ndarray
    design: np.ndarray
    roughness: np.ndarray
    texture: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)
        expected = (len(self.x), len(self.y))
        for name in ("design", "roughness", "texture"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite heights")
            setattr(self, name, value)
        if not np.isfinite(self.x).all() or not np.isfinite(self.y).all():
            raise ValueError("surface coordinates must be finite")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")

    @property
    def height(self) -> np.ndarray:
        return self.design + self.roughness + self.texture

    def reconstruct(self) -> np.ndarray:
        """Return reconstructed z values in x-major (x, y) layout."""
        return self.height.copy()

    def frozen_copy(self) -> SurfaceTile:
        """Copy the actual sampled layers without drawing another random field."""
        return SurfaceTile(
            self.x.copy(), self.y.copy(), self.design.copy(), self.roughness.copy(),
            self.texture.copy(), deepcopy(self.metadata),
        )

    def save(self, path: str | Path) -> None:
        """Persist this realized tile and its provenance as a compressed NPZ."""
        import json

        def json_default(value):
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(f"cannot encode metadata value of type {type(value).__name__}")

        np.savez_compressed(
            path,
            x=self.x,
            y=self.y,
            design_m=self.design,
            roughness_m=self.roughness,
            texture_m=self.texture,
            metadata_json=np.asarray(
                json.dumps(self.metadata, ensure_ascii=False, default=json_default)
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> SurfaceTile:
        import json

        with np.load(path, allow_pickle=False) as data:
            return cls(
                data["x"].copy(), data["y"].copy(), data["design_m"].copy(),
                data["roughness_m"].copy(), data["texture_m"].copy(),
                json.loads(str(data["metadata_json"].item())),
            )
