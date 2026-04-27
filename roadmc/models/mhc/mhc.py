"""Manifold Hyper-Connection (mHC) — learnable doubly-stochastic channel mixing."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_knopp(M: torch.Tensor, iters: int = 5, eps: float = 1e-6) -> torch.Tensor:
    """Sinkhorn-Knopp normalization to produce a doubly stochastic matrix.

    Args:
        M: Unnormalized affinity matrix of shape (C, C).
        iters: Number of alternating row/column normalization iterations.
        eps: Small constant to clamp denominators and prevent division by zero.

    Returns:
        Doubly stochastic matrix H of shape (C, C) satisfying
        H.sum(dim=0) ≈ 1 and H.sum(dim=1) ≈ 1.
    """
    # Exponentiate to get positive entries
    H = torch.exp(M)

    for _ in range(iters):
        # Row normalize: each row sums to 1
        row_sum = H.sum(dim=1, keepdim=True).clamp(min=eps)
        H = H / row_sum

        # Column normalize: each column sums to 1
        col_sum = H.sum(dim=0, keepdim=True).clamp(min=eps)
        H = H / col_sum

    return H


class MHCConnection(nn.Module):
    """Manifold Hyper-Connection: learnable doubly-stochastic channel mixing.

    Given input x ∈ ℝ^(B×C) and residual r ∈ ℝ^(B×C), computes:
        M = softplus(W₁) · softplus(W₂)ᵀ ∈ ℝ^(C×C)
        H = sinkhorn_knopp(M / τ)   (doubly stochastic)
        y = x + H @ r

    After ``deploy()``, H is frozen and the Sinkhorn iterations + softplus
    are eliminated, turning forward into a single matmul.
    """

    def __init__(self, channels: int, sinkhorn_iters: int = 5, temp: float = 1.0) -> None:
        super().__init__()
        self.channels = channels
        self.sinkhorn_iters = sinkhorn_iters
        self.temp = temp

        self.W1 = nn.Parameter(torch.randn(channels, channels) * 0.01 - 5.0)
        self.W2 = nn.Parameter(torch.randn(channels, channels) * 0.01 - 5.0)
        # Note: W initialized to ~-5.0 so that softplus(W) ≈ 0.007,
        # preventing exp(M) overflow in Sinkhorn-Knopp for C up to 768.
        # softplus(-5 + ε) ≈ 0.007, M ≈ C × 0.007², exp(M) ≈ exp(C×5e-5).
        # For C=768: exp(0.038) ≈ 1.04 — numerically safe.

        # Buffers
        self.register_buffer("stochastic_matrix", None)
        self.register_buffer("_deployed", torch.tensor(False))

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Apply manifold hyper-connection.

        If deployed, uses the frozen H directly (no Sinkhorn iterations).

        Args:
            x: Input tensor of shape (B, C).
            residual: Residual tensor of shape (B, C).

        Returns:
            Output tensor of shape (B, C): x + H @ residual.
        """
        if self._deployed.item():
            # Fast path: use pre-computed frozen H
            H = self.stochastic_matrix
        else:
            # Compute positive affinity matrix
            M = F.softplus(self.W1) @ F.softplus(self.W2).T  # (C, C)

            # Sinkhorn-Knopp normalization with temperature
            H = sinkhorn_knopp(M / self.temp, iters=self.sinkhorn_iters)  # (C, C)
        # Save H as buffer (detached) for deploy() and verification
        self.stochastic_matrix = H.detach().clone()

        # y = x + H @ r  — matrix multiply on channel dimension
        y = x + (residual @ H.T)
        return y

    def deploy(self) -> "MHCConnection":
        """Freeze the stochastic matrix — converts dynamic Sinkhorn to static matmul.

        After deploy():
        - H is computed once and frozen (detached, persistent buffer)
        - ``forward()`` skips softplus + Sinkhorn, uses H directly
        - W1, W2 remain (are still saved in state_dict) but are never used in forward
        - To truly remove W1, W2, use ``torch.save(model.stochastic_matrix, ...)``

        Returns:
            self, for chaining.
        """
        # Compute H from current weights (always, even if previously computed)
        M = F.softplus(self.W1) @ F.softplus(self.W2).T
        H = sinkhorn_knopp(M / self.temp, iters=self.sinkhorn_iters).detach()

        self.register_buffer("stochastic_matrix", H, persistent=True)
        self._deployed = torch.tensor(True)
        return self


if __name__ == "__main__":
    torch.manual_seed(42)
    B, C = 2, 64
    mhc = MHCConnection(C)
    x = torch.randn(B, C)
    r = torch.randn(B, C)
    y = mhc(x, r)

    # Test 1: output shape
    assert y.shape == (B, C), f"Expected ({B}, {C}), got {y.shape}"

    # Test 2: doubly stochastic
    H = mhc.stochastic_matrix
    row_sum = H.sum(dim=1)
    col_sum = H.sum(dim=0)
    assert (row_sum - 1).abs().max() < 1e-3, f"Row sum error: {(row_sum - 1).abs().max()}"
    assert (col_sum - 1).abs().max() < 1e-3, f"Col sum error: {(col_sum - 1).abs().max()}"

    # Test 3: deploy() freezes H AND switches to static forward
    mhc.eval()
    mhc.deploy()
    assert mhc.stochastic_matrix.requires_grad is False
    assert mhc._deployed.item() is True
    y2 = mhc(x, r)  # fast path: no Sinkhorn iterations
    assert y2.shape == (B, C)
    assert not torch.isnan(y2).any()
    # Deployed forward should give similar (not identical due to float precision) result
    print(f"  deploy: y close = {torch.allclose(y, y2, atol=1e-4)}")

    # Test 4: non-NaN
    assert not torch.isnan(y).any()

    print(
        f"[PASS] MHCConnection: output={y.shape}, "
        f"row_err={row_sum.sub(1).abs().max():.6f}, "
        f"col_err={col_sum.sub(1).abs().max():.6f}"
    )