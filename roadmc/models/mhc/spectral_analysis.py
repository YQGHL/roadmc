"""Spectral analysis of DSCM — non-expansiveness of doubly-stochastic mixing.

数学事实（Birkhoff–von Neumann + 范数凸性）：双随机矩阵 H 是置换
矩阵的凸组合，故 σ_max(H) ≤ 1；又 H·1 = 1 给出 σ_max(H) = 1——
**非扩张 (non-expansive)，不是收缩 (contractive)**，界是紧的。

该性质保证的是**乘法/凸组合接线**的稳定性：x_{l+1} = H_l x_l + f_l，
因为双随机矩阵之积仍双随机，∏H_l 的谱范数恒为 1。加法接线
y = x + H x 的雅可比为 (I+H)，沿全 1 向量特征值为 2，L 层复合范数
~2^L——旧版本的级联测试用与 x 无关的随机残差模拟（一个无论 H 为何
都"稳定"的随机游走），验证的是稻草人；本版本模拟真实的耦合递推。
"""

import sys
from pathlib import Path

import torch


class SpectralAnalyzer:
    """Analyze DSCM mixing matrices: spectral norm and cascade stability."""

    @staticmethod
    def spectral_norm(H: torch.Tensor) -> torch.Tensor:
        """Largest singular value; = 1 (tight) for doubly stochastic H."""
        svals = torch.linalg.svdvals(H)
        return svals.max()

    @staticmethod
    def verify_doubly_stochastic(H: torch.Tensor, tol: float = 1e-4) -> dict[str, float]:
        """Row/col sums, non-negativity, and spectral norm of H."""
        del tol
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
        wiring: str = "multiplicative",
    ) -> dict[str, float]:
        """Simulate the **actual** coupled recursion through ``depth`` layers.

        wiring:
            - ``multiplicative``（当前接线）: x ← H x + 0.1·randn。
              ‖x_L‖ ≤ ‖x_0‖ + Σ‖f_l‖，随深度线性有界。
            - ``additive_legacy``（旧接线，保留作回归证据）:
              x ← x + H x。含 (I+H)，沿全 1 方向按 2^L 指数增长。
        """
        C = H.shape[0]
        ratios = []

        for _ in range(n_samples):
            x = torch.randn(1, C)
            x0_norm = torch.norm(x)
            for _ in range(depth):
                if wiring == "multiplicative":
                    x = x @ H.T + torch.randn(1, C) * 0.1
                elif wiring == "additive_legacy":
                    x = x + x @ H.T
                else:
                    raise ValueError(f"unknown wiring: {wiring}")
            ratios.append((torch.norm(x) / x0_norm).item())

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

    H_eye = torch.eye(64)
    sn = analyzer.spectral_norm(H_eye)
    assert abs(sn - 1.0) < 1e-6, f"Identity spectral norm should be 1, got {sn}"

    from roadmc.models.mhc.mhc import MHCConnection, sinkhorn_log

    mhc = MHCConnection(64)
    with torch.no_grad():
        # 扰动出一个远离恒等/均匀的非退化 H，避免只在退化点验证。
        # 尖锐核的 Sinkhorn 收敛率下降 (Franklin & Lorenz 1989)，此处
        # 用足量迭代验证投影本身；训练中的收敛残差由
        # MHCConnection.diagnostics() 的 row/col 误差监控。
        mhc.log_kernel.add_(torch.randn(64, 64) * 2.0)
        H = sinkhorn_log(mhc.log_kernel / mhc.temp, iters=300)

    stats = analyzer.verify_doubly_stochastic(H)
    assert stats["row_err"] < 1e-3, f"Row sum error: {stats['row_err']}"
    assert stats["col_err"] < 1e-3, f"Col sum error: {stats['col_err']}"
    assert stats["spectral_norm"] <= 1.0 + 1e-4, (
        f"Spectral norm > 1: {stats['spectral_norm']}"
    )

    # 真实接线：depth=60 有界（线性于噪声注入，而非指数）。
    ok = analyzer.cascade_energy(H, depth=60, n_samples=20, wiring="multiplicative")
    assert ok["max_ratio"] < 10, f"multiplicative wiring unstable: {ok}"

    # 旧接线：同一个 H 下指数爆炸——锁死审计发现，防止接线回归。
    bad = analyzer.cascade_energy(H, depth=60, n_samples=5, wiring="additive_legacy")
    assert bad["avg_ratio"] > 1e6, f"legacy additive wiring should explode: {bad}"

    print(
        f"SpectralAnalyzer: norm={stats['spectral_norm']:.6f}, "
        f"multiplicative depth-60 ratio={ok['avg_ratio']:.2f} (bounded), "
        f"legacy additive ratio={bad['avg_ratio']:.2e} (exponential, regression-locked)"
    )
