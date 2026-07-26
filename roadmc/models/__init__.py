"""RoadMC model package."""

# DeformableWindowAttention3D 不在导出面：默认训练路径从未使用它，
# 且其 argmin 采样不可微、cdist 距离阵在 8GB 卡上不可用（审计 F0.4）。
# 类保留在 window_attention.py 中作为实验性代码。
__all__ = [
    "Swin3D",
    "PointMambaBackbone",
    "MHCConnection",
    "WindowAttention3D",
    "ShiftedWindowTransformerBlock",
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
    if name in {"WindowAttention3D", "ShiftedWindowTransformerBlock"}:
        from roadmc.models.attention.window_attention import (
            ShiftedWindowTransformerBlock,
            WindowAttention3D,
        )
        return {
            "WindowAttention3D": WindowAttention3D,
            "ShiftedWindowTransformerBlock": ShiftedWindowTransformerBlock,
        }[name]
    if name == "RoadMCSegModel":
        from roadmc.models.model_pl import RoadMCSegModel
        return RoadMCSegModel
    raise AttributeError(name)
