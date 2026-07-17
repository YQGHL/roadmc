"""Metadata contract for real road point-cloud scenes.

Each ``scene.ply`` / ``scene.pcd`` / ``scene.las`` / ``scene.laz`` may have a sibling
``scene.json``. The sidecar prevents silent unit and intensity-scale mismatches
when comparing a real scan with a physics-generated RoadMC scene.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np


UNIT_TO_METERS = {"m": 1.0, "cm": 0.01, "mm": 0.001}
INTENSITY_SCALES = {"normalized_0_1", "uint8", "uint16", "raw_range"}


@dataclass(frozen=True)
class RoadPointCloudMetadata:
    """Portable provenance and calibration information for one road scene."""

    schema_version: str = "roadmc.real.v1"
    sensor: str = "unknown"
    coordinate_units: str = "m"
    intensity_scale: str = "normalized_0_1"
    road_segment_id: str | None = None
    capture_date: str | None = None
    source_dataset: str | None = None
    coordinate_reference_system: str | None = None
    label_source: str | None = None
    notes: str | None = None
    intensity_min: float | None = None
    intensity_max: float | None = None

    def __post_init__(self) -> None:
        if self.coordinate_units not in UNIT_TO_METERS:
            raise ValueError(f"coordinate_units must be one of {tuple(UNIT_TO_METERS)}, got {self.coordinate_units!r}")
        if self.intensity_scale not in INTENSITY_SCALES:
            raise ValueError(f"intensity_scale must be one of {tuple(INTENSITY_SCALES)}, got {self.intensity_scale!r}")
        if self.intensity_scale == "raw_range":
            if self.intensity_min is None or self.intensity_max is None:
                raise ValueError("raw_range intensity requires intensity_min and intensity_max")
            if self.intensity_max <= self.intensity_min:
                raise ValueError("intensity_max must be greater than intensity_min")
        if self.capture_date is not None:
            date.fromisoformat(self.capture_date)

    @property
    def coordinate_scale_to_meters(self) -> float:
        return UNIT_TO_METERS[self.coordinate_units]

    def normalize_intensity(self, values: np.ndarray | None) -> np.ndarray | None:
        """Convert source intensity values to a stable ``[0, 1]`` convention."""
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float32)
        if self.intensity_scale == "normalized_0_1":
            normalized = array
        elif self.intensity_scale == "uint8":
            normalized = array / 255.0
        elif self.intensity_scale == "uint16":
            normalized = array / 65535.0
        else:
            assert self.intensity_min is not None and self.intensity_max is not None
            normalized = (array - self.intensity_min) / (self.intensity_max - self.intensity_min)
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoadPointCloudMetadata":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items() if key in allowed})


def metadata_sidecar_path(point_cloud_path: str | Path) -> Path:
    """Return the JSON sidecar path for a point-cloud file."""
    return Path(point_cloud_path).with_suffix(".json")


def load_scene_metadata(
    point_cloud_path: str | Path,
    *,
    fallback: RoadPointCloudMetadata | None = None,
    require_sidecar: bool = False,
) -> RoadPointCloudMetadata:
    """Load and validate a scene sidecar, or return a documented fallback."""
    sidecar = metadata_sidecar_path(point_cloud_path)
    if not sidecar.exists():
        if require_sidecar:
            raise FileNotFoundError(f"Missing required RoadMC metadata sidecar: {sidecar}")
        return fallback or RoadPointCloudMetadata()
    with sidecar.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Metadata sidecar must contain a JSON object: {sidecar}")
    return RoadPointCloudMetadata.from_dict(payload)


def write_scene_metadata(point_cloud_path: str | Path, metadata: RoadPointCloudMetadata) -> Path:
    """Write a validated metadata sidecar and return its path."""
    sidecar = metadata_sidecar_path(point_cloud_path)
    with sidecar.open("w", encoding="utf-8") as handle:
        json.dump(metadata.as_dict(), handle, ensure_ascii=False, indent=2)
    return sidecar
