import torch
import torch.nn as nn


class WGANDiscriminator(nn.Module):
    """WGAN-GP discriminator for point cloud style detection.
    
    PointNet-style: per-point MLP → max pooling → global MLP → logit
    
    Input: (B, N, C_in) — cat(points, normals) or stylized output
    Output: (B, 1) — logits (no sigmoid, as required by WGAN-GP)
    """

    def __init__(self, in_channels=6):
        super().__init__()
        # Per-point MLP (shared across all points)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
        )
        # Global MLP (after max pooling)
        self.global_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (B, N, C_in)
        # Per-point features: (B, N, 256)
        x = self.mlp(x)
        # Max pooling over points: (B, 256)
        x = x.max(dim=1)[0]
        # Global output: (B, 1)
        x = self.global_mlp(x)
        return x


if __name__ == '__main__':
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    torch.manual_seed(42)
    B, N = 2, 1024
    x = torch.randn(B, N, 6)

    disc = WGANDiscriminator(in_channels=6)
    out = disc(x)

    assert out.shape == (B, 1), f"Expected ({B}, 1), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"

    loss = out.sum()
    loss.backward()
    params_no_grad = [
        n for n, p in disc.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert not params_no_grad, f"Params without gradient: {params_no_grad}"

    print(
        f"[PASS] WGANDiscriminator: output={out.shape}, params={sum(p.numel() for p in disc.parameters())}"
    )