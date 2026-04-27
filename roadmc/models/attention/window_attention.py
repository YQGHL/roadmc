"""3D window attention for point cloud transformers.

Implements:
- WindowAttention3D with MLP-learned relative position bias
- DeformableWindowAttention3D (optional, M1)
- ShiftedWindowTransformerBlock matching 项目规划.md spec:
    x = LN → WindowAttention → +x → LN → FFN → MHCConnection(x, residual)
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure roadmc is importable when running as __main__
_HERE = Path(__file__).resolve().parents[3]  # project root (4 dirs up from this file)
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from roadmc.models.mhc.mhc import MHCConnection


def _window_partition(
    coords: torch.Tensor,
    window_size: int,
    shift: bool = False,
) -> Tuple[torch.Tensor, int]:
    """Assign points to 3D windows based on spatial coordinates.

    Divides the 3D space into a regular grid and assigns each point to
    a grid cell (window). The grid resolution is chosen so that each
    window contains approximately ``window_size`` points.

    Args:
        coords: (B, N, 3) point coordinates in original space.
        window_size: target number of points per window.
        shift: if True, apply cyclic shift to coordinates before
            partitioning (shifted-window attention).

    Returns:
        window_indices: (B, N) integer window ID for each point.
        num_windows: total number of possible windows (grid_res^3).
    """
    B, N, _ = coords.shape

    # Normalize coords to [0, 1] per batch
    coords_min = coords.amin(dim=1, keepdim=True)
    coords_max = coords.amax(dim=1, keepdim=True)
    coords_range = coords_max - coords_min
    # Guard against degenerate (zero-range) dimensions
    coords_range = torch.where(
        coords_range < 1e-6, torch.ones_like(coords_range), coords_range
    )
    coords_norm = (coords - coords_min) / coords_range  # (B, N, 3) in [0, 1]

    # Grid resolution: aim for ~window_size points per cell
    # total_cells = grid_res^3 ≈ N / window_size
    grid_res = max(1, round((N / window_size) ** (1.0 / 3.0)))

    # Apply cyclic shift for shifted-window attention
    if shift:
        shift_amount = 0.5 / grid_res
        coords_norm = (coords_norm + shift_amount) % 1.0

    # Bin coordinates into grid cells
    bin_idx = (coords_norm * grid_res).long().clamp(0, grid_res - 1)  # (B, N, 3)

    # Hash (bx, by, bz) → integer window ID
    window_id = (
        bin_idx[..., 0] * (grid_res * grid_res)
        + bin_idx[..., 1] * grid_res
        + bin_idx[..., 2]
    )  # (B, N)

    num_windows = grid_res ** 3

    return window_id, num_windows


class WindowAttention3D(nn.Module):
    """3D window attention with MLP-learned relative position bias.

    Input: (B, N, C) features + (B, N, 3) coordinates
    Output: (B, N, C) attended features

    Procedure:
    1. Project Q, K, V from input: QKV = Linear(C, 3*C), split into
       Q, K, V each (B, N, C)
    2. Reshape to multi-head: Q (B, num_heads, N, C//num_heads)
    3. Partition points into windows based on (x, y, z) coordinates
    4. Compute attention scores = QK^T/sqrt(d) + relative_pos_bias
    5. Mask out cross-window attention
    6. Softmax → output = AV → reshape back
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 3,
        window_size: int = 32,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, (
            f"dim {dim} must be divisible by num_heads {num_heads}"
        )

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

        # 3D relative position bias via MLP:
        # Input: 3D offset vector (dx, dy, dz) between query and key
        # Output: per-head bias value
        pos_hidden = int(dim // mlp_ratio)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_hidden),
            nn.GELU(),
            nn.Linear(pos_hidden, num_heads),
        )

    def forward(
        self,
        coords: torch.Tensor,
        x: torch.Tensor,
        shift: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            coords: (B, N, 3) point coordinates.
            x: (B, N, C) input features.
            shift: if True, apply cyclic shift to window partition.

        Returns:
            (B, N, C) attended features.
        """
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        # 1. Project Q, K, V
        qkv = self.qkv(x)  # (B, N, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, N, C)

        # 2. Reshape to multi-head: (B, H, N, D)
        q = q.view(B, N, H, D).transpose(1, 2)
        k = k.view(B, N, H, D).transpose(1, 2)
        v = v.view(B, N, H, D).transpose(1, 2)

        # 3. Partition into windows
        window_id, _num_windows = _window_partition(
            coords, self.window_size, shift=shift
        )

        # 4. Compute attention scores
        scale = D ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, H, N, N)

        # 5. Compute 3D relative position bias
        # Pairwise coordinate offsets: (B, N, N, 3)
        offsets = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, N, N, 3)
        rel_pos_bias = self.pos_mlp(offsets)  # (B, N, N, H)
        rel_pos_bias = rel_pos_bias.permute(0, 3, 1, 2)  # (B, H, N, N)
        attn = attn + rel_pos_bias

        # 6. Mask out cross-window attention
        # window_id: (B, N) → same_window: (B, 1, N, N)
        same_window = window_id.unsqueeze(2) == window_id.unsqueeze(1)  # (B, N, N)
        same_window = same_window.unsqueeze(1)  # (B, 1, N, N)
        attn = attn.masked_fill(~same_window, -1e4)

        # 7. Softmax and attend
        attn = self.softmax(attn)

        # 8. Weighted sum
        out = attn @ v  # (B, H, N, D)
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        # 9. Project output
        out = self.proj(out)

        return out


class DeformableWindowAttention3D(nn.Module):
    """M1: Deformable 3D window attention — each query predicts 3D offset.

    Instead of fixed window partitioning, each query learns to predict
    a 3D offset vector. The offset is used to sample key/value features
    from a deformed grid, enabling adaptive receptive fields.

    Reference: Deformable DETR (Zhu et al., 2021) adapted for 3D.

    Structure:
        offset_net: Linear(C, 3) → predicts (dx, dy, dz) for each query
        sampled keys/values from deformed positions
        standard attention on sampled set
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 3,
        num_sample_points: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        offset_init_scale: float = 10.0,  # H1 fix: learnable offset multiplier
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_sample_points = num_sample_points
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, f"dim {dim} not divisible by {num_heads}"

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

        # Offset network: each query → 3D offset per sample point
        self.offset_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 3 * num_sample_points),
        )

        # H1 fix: offset scale multiplier (plain float, not nn.Parameter)
        # The nearest-neighbor argmin is non-differentiable, so having this as
        # nn.Parameter would decay via weight_decay without gradient, undermining the fix.
        self.offset_scale = offset_init_scale

        # 3D relative position bias (per-head, not same across heads)
        pos_hidden = int(dim // mlp_ratio)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_hidden),
            nn.GELU(),
            nn.Linear(pos_hidden, num_heads),
        )

    def forward(self, coords: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with real deformable attention.

        Each query predicts K=num_sample_points 3D offsets.
        Keys/values are sampled at deformed positions via nearest neighbor,
        then attention is computed over K sampled keys (not all N keys).
        """
        B, N, C = x.shape
        H, D, K = self.num_heads, self.head_dim, self.num_sample_points

        # QKV projection
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
        k = k.view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
        v = v.view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)

        # Predict 3D offsets for each query
        offsets = self.offset_net(x)  # (B, N, 3*K)
        offsets = offsets.view(B, N, K, 3)  # (B, N, K, 3)

        # Reference sampling points: query_xyz + offset_scale * predicted_offset
        ref_points = coords.unsqueeze(2)  # (B, N, 1, 3)
        sample_points = ref_points + offsets * self.offset_scale  # (B, N, K, 3)

        # For each sample point, find nearest neighbor in the key set
        # pair_dist[p, s] = ‖sample_point[p, s] - key_coords[all]‖²
        # Use per-batch cdist: (N*K, 3) with (N, 3) → (N*K, N)
        sampled_k = []
        sampled_v = []
        for b in range(B):
            sp = sample_points[b]  # (N, K, 3)
            kc = coords[b]         # (N, 3)
            # Flatten to (N*K, 3) vs (N, 3) → (N*K, N)
            dist = torch.cdist(sp.reshape(-1, 3), kc.unsqueeze(0))  # (N*K, N)
            _, nn_idx = dist.min(dim=-1)  # (N*K,) nearest key index
            nn_idx = nn_idx.view(N, K)    # (N, K) nearest keys per query
            # Gather k and v at sampled positions
            # k: (H, N, D), nn_idx: (N, K) → gather over dim=1
            k_b = k[b]    # (H, N, D)
            v_b = v[b]    # (H, N, D)
            # Index into N dim for each head
            k_sampled = k_b[:, nn_idx]  # (H, N, K, D)
            v_sampled = v_b[:, nn_idx]  # (H, N, K, D)
            sampled_k.append(k_sampled)
            sampled_v.append(v_sampled)

        sampled_k = torch.stack(sampled_k)  # (B, H, N, K, D)
        sampled_v = torch.stack(sampled_v)  # (B, H, N, K, D)

        # Attention: q attends over K sampled keys
        # q: (B, H, N, D) → unsqueeze for K dim
        # sampled_k: (B, H, N, K, D)
        scale = D ** -0.5
        attn = (q.unsqueeze(3) * sampled_k).sum(dim=-1) * scale  # (B, H, N, K)

        # 3D relative position bias between query and sampled positions
        # offsets are the predicted offset from query to sample
        # pos_mlp produces per-head biases (one per head, not shared)
        sampled_offsets = offsets  # (B, N, K, 3)
        rel_pos_bias = self.pos_mlp(sampled_offsets)  # (B, N, K, H)
        rel_pos_bias = rel_pos_bias.permute(0, 3, 1, 2)  # (B, H, N, K)
        attn = attn + rel_pos_bias

        # Softmax over K sampled positions
        attn = self.softmax(attn)  # (B, H, N, K)

        # Weighted sum over sampled values
        # attn: (B, H, N, K), sampled_v: (B, H, N, K, D)
        out = (attn.unsqueeze(-1) * sampled_v).sum(dim=3)  # (B, H, N, D)
        out = out.transpose(1, 2).reshape(B, N, C)

        # Project output
        out = self.proj(out)
        return out


class ShiftedWindowTransformerBlock(nn.Module):
    """Transformer block with window attention and MHC connection.

    Architecture (matching 项目规划.md lines 275-278)::

        x_ln = LN(x)
        x_attn = WindowAttention(x_ln) + x          # pre-LN residual
        x_ln2 = LN(x_attn)
        x_ffn = FFN(x_ln2) + x_attn                # pre-LN residual
        x_out = MHCConnection(x_ffn, x_attn)       # learned channel mixing
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 3,
        window_size: int = 32,
        mlp_ratio: float = 4.0,
        shift: bool = False,
        use_mhc: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.shift = shift
        self.use_mhc = use_mhc

        # LN → WindowAttention → residual
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, num_heads, window_size, mlp_ratio)

        # LN → FFN → residual
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

        # MHC connection (channel-space residual mixing)
        if use_mhc:
            self.mhc = MHCConnection(dim)
        else:
            self.mhc = None

    def forward(self, coords: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            coords: (B, N, 3) point coordinates.
            x: (B, N, C) input features.

        Returns:
            (B, N, C) output features.
        """
        B, N, C = x.shape

        # 1. LN → Window Attention → residual (pre-LN: +x, not +x_norm)
        # spec: x = LN(x) → WindowAttention → +x
        x_norm = self.norm1(x)
        x_attn = self.attn(coords, x_norm, shift=self.shift) + x

        # 2. LN → FFN → residual (pre-LN: +x_attn, not +x_ffn_norm)
        x_ffn = self.norm2(x_attn)
        x_ffn = self.ffn(x_ffn) + x_attn

        # 3. MHC: learned channel mixing on residual
        if self.mhc is not None and self.use_mhc:
            # Flatten (B, N, C) → (B*N, C) for MHC
            x_flat = x_ffn.reshape(-1, C)
            r_flat = x_attn.reshape(-1, C)
            x_out = self.mhc(x_flat, r_flat).reshape(B, N, C)
        else:
            # Fallback: simple residual add (no MHC)
            x_out = x_ffn

        return x_out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    torch.manual_seed(42)
    B, N, C = 2, 512, 96

    # Generate random point cloud
    coords = torch.rand(B, N, 3) * 100
    feats = torch.rand(B, N, C)

    # Test WindowAttention3D
    attn = WindowAttention3D(dim=C, num_heads=4, window_size=50)
    out = attn(coords, feats)
    assert out.shape == (B, N, C), f"WindowAttention shape: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in attention output"
    print(f"[PASS] WindowAttention3D: output={out.shape}")

    # Test DeformableWindowAttention3D
    def_attn = DeformableWindowAttention3D(dim=C, num_heads=4, num_sample_points=8)
    out_def = def_attn(coords, feats)
    assert out_def.shape == (B, N, C), f"DeformableAttn shape: {out_def.shape}"
    assert not torch.isnan(out_def).any(), "NaN in deformable attention"
    print(f"[PASS] DeformableWindowAttention3D: output={out_def.shape}")

    # Test ShiftedWindowTransformerBlock (with MHC integration)
    block = ShiftedWindowTransformerBlock(dim=C, num_heads=4, window_size=50, use_mhc=True)
    out2 = block(coords, feats)
    assert out2.shape == (B, N, C), f"TransformerBlock shape: {out2.shape}"
    assert not torch.isnan(out2).any(), "NaN in transformer block"
    # Verify MHC is being used
    assert hasattr(block, 'mhc') and block.mhc is not None, "MHC not integrated"
    print(f"[PASS] ShiftedWindowTransformerBlock (with MHC): output={out2.shape}")

    # Test without MHC (fallback mode)
    block2 = ShiftedWindowTransformerBlock(dim=C, num_heads=4, window_size=50, use_mhc=False)
    out3 = block2(coords, feats)
    assert out3.shape == (B, N, C), f"Block (no MHC) shape: {out3.shape}"
    print(f"[PASS] ShiftedWindowTransformerBlock (no MHC): output={out3.shape}")