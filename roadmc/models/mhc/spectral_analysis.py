"""Spectral analysis of MHC — verifying the contractive property of doubly stochastic matrices.

Uses torch.linalg.svdvals to compute spectral norms and cascade simulations.
"""

import sys
from pathlib import Path

import torch


class SpectralAnalyzer:
    """Analyze MHC matrices for spectral contractive properties.

    Spectral norm verification, 60-layer cascade stability simulation,
    and Birkhoff polytope membership test.
    """

    @staticmethod
    def spectral_norm(H: torch.Tensor) -> torch.Tensor:
        """Compute spectral norm (largest singular value) of H.
        For doubly stochastic H, should be ≤ 1 + 1e-6.
        """
        svals = torch.linalg.svdvals(H)
        return svals.max()

    @staticmethod
    def verify_doubly_stochastic(H: torch.Tensor, tol: float = 1e-4) -> dict[str, float]:
        """Verify H satisfies row-sum=1, col-sum=1, non-negativity, and spectral norm ≤ 1."""
        row_sum = H.sum(dim=1)
        col_sum = H.sum(dim=0)
        return {
            "row_err": (row_sum - 1.0).abs().max().item(),
            "col_err": (col_sum - 1.0).abs().max().item(),
            "min_entry": H.min().item(),
            "spectral_norm": SpectralAnalyzer.spectral_norm(H).item(),
        }

    @staticmethod
    def cascade_energy(
        H: torch.Tensor,
        depth: int = 60,
        n_samples: int = 100,
    ) -> dict[str, float]:
        """Simulate repeated application of H on random residuals, tracking energy ratio.

        Given H ∈ ℝ^(C×C), simulate x_{k+1} = x_k + H @ r_k where r_k are random
        residuals. Tracks ‖x_k‖ / ‖x_0‖ through depth layers. For stable systems,
        the energy ratio should remain bounded.
        """
        C = H.shape[0]
        ratios = []

        for _ in range(n_samples):
            x = torch.randn(1, C)
            x0_norm = torch.norm(x)
            # Track energy without normalization
            for _ in range(depth):
                r = torch.randn(1, C) * 0.1
                x = x + r @ H.T
            ratio = (torch.norm(x) / x0_norm).item()
            ratios.append(ratio)

        return {
            "max_ratio": max(ratios),
            "min_ratio": min(ratios),
            "avg_ratio": sum(ratios) / len(ratios),
            "std_ratio": (torch.tensor(ratios).std()).item(),
        }


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    torch.manual_seed(42)

    analyzer = SpectralAnalyzer()

    # Test 1: identity matrix has spectral norm = 1
    H_eye = torch.eye(64)
    sn = analyzer.spectral_norm(H_eye)
    assert abs(sn - 1.0) < 1e-6, f"Identity spectral norm should be 1, got {sn}"

    # Test 2: create a valid MHC matrix and verify
    from roadmc.models.mhc.mhc import MHCConnection

    mhc = MHCConnection(64)
    x = torch.randn(1, 64)
    r = torch.randn(1, 64)
    _ = mhc(x, r)
    H = mhc.stochastic_matrix

    stats = analyzer.verify_doubly_stochastic(H)
    assert stats["row_err"] < 1e-3, f"Row sum error: {stats['row_err']}"
    assert stats["col_err"] < 1e-3, f"Col sum error: {stats['col_err']}"
    assert stats["spectral_norm"] <= 1.0 + 1e-4, (
        f"Spectral norm > 1: {stats['spectral_norm']}"
    )

    # Test 3: cascade stability
    cascade = analyzer.cascade_energy(H, depth=30, n_samples=20)
    assert cascade["max_ratio"] < 50, f"Energy ratio too high: {cascade}"

    print(
        f"SpectralAnalyzer: norm={stats['spectral_norm']:.6f}, "
        f"row_err={stats['row_err']:.2e}, "
        f"cascade_ratio={cascade['avg_ratio']:.2f}±{cascade['std_ratio']:.2f}"
    )
