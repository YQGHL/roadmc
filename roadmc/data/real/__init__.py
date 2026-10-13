"""Real point-cloud loading and metadata utilities.

Raw NumPy/SciPy evaluation imports must not load the tensor Dataset or PyTorch.
"""

from .metadata import RoadPointCloudMetadata, load_scene_metadata, write_scene_metadata


def __getattr__(name: str):
    if name == "RealRoadDataset":
        from .dataset import RealRoadDataset

        return RealRoadDataset
    raise AttributeError(name)

__all__ = [
    "RealRoadDataset",
    "RoadPointCloudMetadata",
    "load_scene_metadata",
    "write_scene_metadata",
]
