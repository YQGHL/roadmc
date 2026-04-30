"""Style transfer generator for RoadMC point cloud domain adaptation.

Maps synthetic point clouds (with normals) to style-adjusted outputs
via a DGCNN-style encoder-decoder with EdgeConv operations.

Architecture:
    Encoder:
        EdgeConv(6 -> 64, k=16)    # input: cat(points, normals) = 6-dim
        EdgeConv(64 -> 128, k=16)
        EdgeConv(128 -> 256, k=16)
    Decoder:
        Linear(256 -> 128) -> ReLU
        Linear(128 -> 64)  -> ReLU
        Linear(64 -> 6)             # output: [dx,dy,dz, dnx,dny,dnz]

All operations use pure PyTorch (no external graph libraries).
k-NN is computed via torch.cdist in coordinate space.
"""

import torch
from torch import nn as nn


class EdgeConv(nn.Module):
    """Edge convolution: aggregate features from k-NN neighbors.

    For each point i, we find its k nearest neighbors (in coordinate space),
    construct edge features as ``cat([x_i, x_j - x_i])``, apply an MLP,
    and max-pool over the k neighbors.

    **Memory warning**: ``torch.cdist`` computes O(N²) pairwise distances.
    At N=65536 (default config), this requires ~17 GB per sample.
    For production use, consider random downsampling before the GAN,
    or limit input N to ≤ 8192.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
        k: Number of nearest neighbors.
    """

    def __init__(self, in_channels: int, out_channels: int, k: int = 16) -> None:
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features, shape ``(B, C_in, N)``.
            coords: Coordinate space for k-NN, shape ``(B, 3, N)``.
                Typically ``syn_points.transpose(1, 2)``.

        Returns:
            Aggregated features, shape ``(B, C_out, N)``.
        """
        B, C, N = x.shape
        k = min(self.k, max(N - 1, 1))  # handle tiny point clouds

        # ------------------------------------------------------------------
        # 1. k-NN in coordinate space
        # ------------------------------------------------------------------
        coords_t = coords.transpose(1, 2)                     # (B, N, 3)
        dist = torch.cdist(coords_t, coords_t)                # (B, N, N)

        # k+1 because the closest is self (distance 0)
        _, idx = dist.topk(k=k + 1, dim=-1, largest=False)    # (B, N, k+1)
        idx = idx[:, :, 1:]                                   # (B, N, k), exclude self

        # ------------------------------------------------------------------
        # 2. Gather neighbor features
        # ------------------------------------------------------------------
        # idx_expanded: (B, C, N, k)  —  index into dim=2 (N)
        idx_expanded = idx.unsqueeze(1).expand(-1, C, -1, -1)
        # x_expanded:   (B, C, N, k)  —  repeat features k times for gather
        x_expanded = x.unsqueeze(-1).expand(-1, -1, -1, k)
        neighbor_feats = torch.gather(x_expanded, dim=2, index=idx_expanded)

        # ------------------------------------------------------------------
        # 3. Build edge features: cat([x_i, x_j - x_i])
        # ------------------------------------------------------------------
        x_center = x.unsqueeze(-1).expand(-1, -1, -1, k)     # (B, C, N, k)
        edge_diff = neighbor_feats - x_center                 # (B, C, N, k)
        edge_feat = torch.cat([x_center, edge_diff], dim=1)  # (B, 2*C, N, k)

        # ------------------------------------------------------------------
        # 4. Apply MLP and max-pool over neighbors
        # ------------------------------------------------------------------
        edge_feat_flat = edge_feat.permute(0, 2, 3, 1)       # (B, N, k, 2*C)
        edge_feat_flat = edge_feat_flat.reshape(-1, 2 * C)   # (B*N*k, 2*C)

        out = self.mlp(edge_feat_flat)                       # (B*N*k, C_out)
        out = out.reshape(B, N, k, -1)                       # (B,   N, k, C_out)
        out = out.max(dim=2)[0]                               # (B,   N,    C_out)
        out = out.permute(0, 2, 1)                            # (B, C_out, N)

        return out


class StyleTransferGen(nn.Module):
    """Point cloud style transfer generator (DGCNN-style).

    Takes synthetic point clouds + normals and predicts residuals
    to adjust them toward a real-world style distribution.

    Input:
        syn_points  (B, N, 3)  — synthetic point coordinates
        syn_normals (B, N, 3)  — synthetic point normals

    Output:
        (B, N, 6) — coordinate residual ``[dx, dy, dz]``
                    and normal adjustment ``[dnx, dny, dnz]``
    """

    def __init__(self, k: int = 16) -> None:
        super().__init__()
        self.enc1 = EdgeConv(6, 64, k=k)
        self.enc2 = EdgeConv(64, 128, k=k)
        self.enc3 = EdgeConv(128, 256, k=k)

        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6),
        )

    def forward(
        self,
        syn_points: torch.Tensor,
        syn_normals: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            syn_points:  (B, N, 3) point coordinates.
            syn_normals: (B, N, 3) unit normals.

        Returns:
            (B, N, 6) residuals.
        """
        x = torch.cat([syn_points, syn_normals], dim=-1)     # (B, N, 6)
        x = x.transpose(1, 2)                                 # (B, 6, N)

        # Coordinate space for k-NN
        coords = syn_points.transpose(1, 2)                   # (B, 3, N)

        x = self.enc1(x, coords)                              # (B,  64, N)
        x = self.enc2(x, coords)                              # (B, 128, N)
        x = self.enc3(x, coords)                              # (B, 256, N)

        x = x.transpose(1, 2)                                 # (B, N, 256)
        x = self.decoder(x)                                   # (B, N,   6)

        return x


# ====================================================================
# Self-test
# ====================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    torch.manual_seed(42)
    B, N = 2, 1024
    points = torch.randn(B, N, 3)
    normals = torch.nn.functional.normalize(torch.randn(B, N, 3), dim=-1)

    gen = StyleTransferGen()
    out = gen(points, normals)

    assert out.shape == (B, N, 6), f"Expected ({B},{N},6), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"

    # Verify gradient flow
    loss = out.sum()
    loss.backward()
    params_no_grad = [
        n
        for n, p in gen.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not params_no_grad, f"Params without gradient: {params_no_grad}"

    print(
        f"[PASS] StyleTransferGen: "
        f"output={out.shape}, "
        f"params={sum(p.numel() for p in gen.parameters())}"
    )
