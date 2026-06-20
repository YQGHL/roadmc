"""RoadMC model package."""

__all__ = [
    "Swin3D",
    "PointMambaBackbone",
    "MHCConnection",
    "WindowAttention3D",
    "ShiftedWindowTransformerBlock",
    "DeformableWindowAttention3D",
    "RoadMCSegModel",
]


def __getattr__(name: str):
    if name == "Swin3D":
        from roadmc.models.backbone.swin3d import Swin3D
        return Swin3D
    if name == "PointMambaBackbone":
        from roadmc.models.backbone.pointmamba import PointMambaBackbone
        return PointMambaBackbone
    if name == "MHCConnection":
        from roadmc.models.mhc.mhc import MHCConnection
        return MHCConnection
    if name in {"WindowAttention3D", "ShiftedWindowTransformerBlock", "DeformableWindowAttention3D"}:
        from roadmc.models.attention.window_attention import (
            DeformableWindowAttention3D,
            ShiftedWindowTransformerBlock,
            WindowAttention3D,
        )
        return {
            "WindowAttention3D": WindowAttention3D,
            "ShiftedWindowTransformerBlock": ShiftedWindowTransformerBlock,
            "DeformableWindowAttention3D": DeformableWindowAttention3D,
        }[name]
    if name == "RoadMCSegModel":
        from roadmc.models.model_pl import RoadMCSegModel
        return RoadMCSegModel
    raise AttributeError(name)
