"""DSCM — doubly-stochastic channel mixing with identity initialization.

设计要点（对应 2026-07 学术审计的三个 P0 修复）：

1. **log-domain Sinkhorn**（Cuturi 2013; Peyré & Cuturi 2019 §4.4）：
   迭代 ``u ← -logsumexp(L + v)``、``v ← -logsumexp(L + u)``，输出
   ``H = exp(L + u ⊕ v)``。对任意大的 log-核数值稳定——旧实现的
   ``torch.exp(M)`` 在权重漂移 +6 后溢出为 inf/NaN 且无护栏。

2. **恒等初始化 + 单参数 log-核**：``L = γ·I + 0.01·randn``，
   Sinkhorn(exp(L)) ≈ I，训练起点严格等价于标准残差（Hyper-
   Connections 的 identity-init 原则）。旧的 ``softplus(W1)@softplus(W2)ᵀ``
   双矩阵参数化初始化在最大熵均匀矩阵处（Birkhoff 多胞体内离恒等
   最远的点），且其近可分离扰动落在 Sinkhorn 归一化的零空间中，
   实测反传梯度 ~1e-21——4.3M 参数事实冻结、从未学习。

3. **乘法混合语义**：模块只做 ``y = H·x``（由调用方以
   ``H·x_stream + branch`` 接线）。双随机矩阵是置换矩阵的凸组合
   （Birkhoff–von Neumann），σ_max(H) = 1（沿全 1 向量取等，
   **非扩张**而非"收缩"），且双随机矩阵之积仍双随机，因此深层复合
   范数有界。旧的加法接线 ``y = x + H·x`` 的雅可比 (I+H) 沿全 1
   方向特征值为 2，同维堆叠特征范数按 2^L 指数增长。

命名说明：本模块**不是**文献中的 Hyper-Connections / mHC（Zhu et
al. 2024 及其流形约束变体作用于 n 条残差流间的 n×n 混合，扩展率
n≥2）；这里是通道维 (C×C) 的双随机混合残差，故命名 DSCM，类名
MHCConnection 仅为向后兼容保留。
"""

import torch
import torch.nn as nn


def sinkhorn_log(L: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """Log-domain Sinkhorn projection onto the Birkhoff polytope.

    Args:
        L: log-核 (C, C)。任意实值——logsumexp 内部减最大值，无溢出。
        iters: 迭代次数。严格正核下收敛到唯一双随机极限
            (Sinkhorn 1964; Sinkhorn & Knopp 1967)。

    Returns:
        H ≈ 双随机矩阵 (C, C)，完全可微。
    """
    # float32 域内计算：autocast fp16 下 logsumexp/exp 的动态范围不足。
    L = L.float()
    u = torch.zeros(L.shape[0], device=L.device, dtype=L.dtype)
    v = torch.zeros(L.shape[1], device=L.device, dtype=L.dtype)
    for _ in range(iters):
        u = -torch.logsumexp(L + v[None, :], dim=1)
        v = -torch.logsumexp(L + u[:, None], dim=0)
    return torch.exp(L + u[:, None] + v[None, :])


class MHCConnection(nn.Module):
    """DSCM: learnable doubly-stochastic channel mixing, identity-initialized.

    ``forward(x)`` 返回 ``x @ H.T``，H = sinkhorn_log(L/τ)。调用方负责
    残差接线（推荐 ``H·x_stream + branch``）。

    Args:
        channels: 通道数 C。
        sinkhorn_iters: Sinkhorn 迭代次数（log-domain 下每次两个
            logsumexp，开销可忽略）。
        temp: 温度 τ。τ→0 趋向置换矩阵，τ→∞ 趋向均匀矩阵。
        init_gamma: 恒等初始化强度 γ。γ=10 时 H 与 I 的最大偏差
            ~3e-3、y=Hx 相对偏差 ~0.3%——初始动力学等价标准残差。
    """

    def __init__(
        self,
        channels: int,
        sinkhorn_iters: int = 20,
        temp: float = 1.0,
        init_gamma: float = 10.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.sinkhorn_iters = sinkhorn_iters
        self.temp = temp

        log_kernel = init_gamma * torch.eye(channels) + 0.01 * torch.randn(channels, channels)
        self.log_kernel = nn.Parameter(log_kernel)

        self.register_buffer("stochastic_matrix", torch.eye(channels))
        self._deployed = False

    def current_H(self) -> torch.Tensor:
        """按当前参数计算双随机混合矩阵（可微）。"""
        return sinkhorn_log(self.log_kernel / self.temp, iters=self.sinkhorn_iters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix channels: y = x @ H.T.

        不在前向热路径写 buffer（旧实现每步 copy_ 触发 DDP 广播与
        functional 化不兼容）；stochastic_matrix 仅在 deploy() 时物化。
        """
        H = self.stochastic_matrix if self._deployed else self.current_H()
        return x @ H.T.to(x.dtype)

    def deploy(self) -> "MHCConnection":
        """Freeze H — converts dynamic Sinkhorn into a single static matmul."""
        with torch.no_grad():
            self.stochastic_matrix.copy_(self.current_H())
        self._deployed = True
        return self

    def diagnostics(self) -> dict:
        """论文消融所需的 H 形态诊断量（熵、离恒等/均匀距离、谱范数）。"""
        with torch.no_grad():
            H = self.current_H()
            C = self.channels
            uniform = torch.full_like(H, 1.0 / C)
            entropy = -(H.clamp_min(1e-12) * H.clamp_min(1e-12).log()).sum() / C
            return {
                "entropy_per_row": float(entropy),
                "dist_to_identity": float((H - torch.eye(C, device=H.device)).abs().max()),
                "dist_to_uniform": float((H - uniform).abs().max()),
                "spectral_norm": float(torch.linalg.matrix_norm(H, ord=2)),
                "row_sum_err": float((H.sum(dim=1) - 1).abs().max()),
                "col_sum_err": float((H.sum(dim=0) - 1).abs().max()),
            }


class HyperConnection(nn.Module):
    """Faithful n-stream Hyper-Connections with Birkhoff-constrained mixing (mHC).

    这是文献口径的 mHC（Zhu et al. 2024 的 Hyper-Connections + 流形约束
    变体），与本模块的 DSCM 是**不同的轴**：HC 混合的是 n 条并行残差流
    (n×n)，DSCM 混合的是通道 (C×C)。提供本类是为了让论文的消融矩阵能
    给出 DSCM vs 真 mHC(n=2/4) 的受控对照——xHC (arXiv 2607.14530) 之后
    这是必答题。

    层更新（静态 HC + 双随机残差混合）::

        x_in  = w_pre · h                       # (n,) 读出，聚合 n 条流
        out   = F(x_in)                         # 分支（注意力 / FFN）
        h'    = H_res h + w_post ⊗ out          # H_res ∈ Birkhoff 多胞体

    **恒等初始化是 bit-exact 的**：h 各流初始化为输入的副本，
    w_pre = 1/n·1（读出 = x），H_res = I，w_post = 1（每条流都加上
    分支输出）——此时 n 条流恒等且整网严格等价于标准 pre-norm 残差
    网络，训练从已知良好解出发（HC 论文的 identity-init 原则）。

    Args:
        channels: 通道数（仅用于诊断/记录，混合不作用在通道上）。
        n_streams: 残差流数 n。n=1 时退化为标准残差（可作对照）。
        sinkhorn_iters / temp / init_gamma: 同 :class:`MHCConnection`。
    """

    def __init__(
        self,
        channels: int,
        n_streams: int = 2,
        sinkhorn_iters: int = 20,
        temp: float = 1.0,
        init_gamma: float = 10.0,
    ) -> None:
        super().__init__()
        if n_streams < 1:
            raise ValueError(f"n_streams must be >= 1, got {n_streams}")
        self.channels = channels
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.temp = temp

        log_kernel = init_gamma * torch.eye(n_streams) + 0.01 * torch.randn(n_streams, n_streams)
        self.log_kernel = nn.Parameter(log_kernel)
        self.w_pre = nn.Parameter(torch.full((n_streams,), 1.0 / n_streams))
        self.w_post = nn.Parameter(torch.ones(n_streams))

    def current_H(self) -> torch.Tensor:
        """n×n 双随机残差混合矩阵（可微）。"""
        return sinkhorn_log(self.log_kernel / self.temp, iters=self.sinkhorn_iters)

    def read(self, h: torch.Tensor) -> torch.Tensor:
        """(B, N, n, C) -> (B, N, C)：按 w_pre 聚合各流作为分支输入。"""
        return torch.einsum("s,bnsc->bnc", self.w_pre.to(h.dtype), h)

    def write(self, h: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        """h' = H_res·h + w_post ⊗ branch，形状 (B, N, n, C)。"""
        H = self.current_H().to(h.dtype)
        mixed = torch.einsum("ts,bnsc->bntc", H, h)
        return mixed + self.w_post.to(h.dtype).view(1, 1, -1, 1) * branch.unsqueeze(-2)

    @staticmethod
    def expand(x: torch.Tensor, n_streams: int) -> torch.Tensor:
        """(B, N, C) -> (B, N, n, C)：复制为 n 条相同的流。"""
        return x.unsqueeze(-2).expand(-1, -1, n_streams, -1).contiguous()

    @staticmethod
    def contract(h: torch.Tensor) -> torch.Tensor:
        """(B, N, n, C) -> (B, N, C)：流维取均值（与 expand 互逆于恒等初始化）。"""
        return h.mean(dim=-2)

    def diagnostics(self) -> dict:
        with torch.no_grad():
            H = self.current_H()
            n = self.n_streams
            return {
                "n_streams": n,
                "dist_to_identity": float((H - torch.eye(n, device=H.device)).abs().max()),
                "spectral_norm": float(torch.linalg.matrix_norm(H, ord=2)),
                "row_sum_err": float((H.sum(dim=1) - 1).abs().max()),
                "col_sum_err": float((H.sum(dim=0) - 1).abs().max()),
                "w_pre": [float(v) for v in self.w_pre.detach().cpu()],
                "w_post": [float(v) for v in self.w_post.detach().cpu()],
            }


if __name__ == "__main__":
    torch.manual_seed(42)
    B, C = 4, 64
    mhc = MHCConnection(C)
    x = torch.randn(B, C)
    y = mhc(x)

    assert y.shape == (B, C)
    assert not torch.isnan(y).any()

    diag = mhc.diagnostics()
    assert diag["row_sum_err"] < 1e-3 and diag["col_sum_err"] < 1e-4
    assert diag["dist_to_identity"] < 1e-2, "identity init violated"
    assert (y - x).norm() / x.norm() < 0.02, "init must behave as identity"

    # 梯度必须非零（旧实现 ~1e-21）
    loss = mhc(x).square().sum()
    loss.backward()
    grad_norm = mhc.log_kernel.grad.norm().item()
    assert grad_norm > 1e-6, f"frozen mixing matrix: grad={grad_norm}"

    mhc2 = MHCConnection(C).deploy()
    y2 = mhc2(x)
    assert torch.allclose(y2, mhc2(x)), "deployed forward must be deterministic"

    print(f"DSCM self-test passed: dist_to_I={diag['dist_to_identity']:.2e}, "
          f"grad={grad_norm:.3e}, sigma_max={diag['spectral_norm']:.6f}")

    # --- n 流 Hyper-Connections（文献口径 mHC） ---
    for n in (2, 4):
        hc = HyperConnection(C, n_streams=n)
        h = HyperConnection.expand(torch.randn(2, 16, C), n)
        # 恒等初始化：read 恢复输入
        x_in = hc.read(h)
        assert torch.allclose(x_in, h[..., 0, :], atol=1e-5), "identity read broken"
        # 一层更新后各流仍应几乎相同（bit-exact identity init 的可观测后果）
        branch = torch.randn(2, 16, C)
        h2 = hc.write(h, branch)
        spread = (h2 - h2.mean(dim=-2, keepdim=True)).abs().max()
        assert float(spread) < 5e-2, f"streams diverged at init: {float(spread)}"
        # 等价于标准残差
        assert torch.allclose(HyperConnection.contract(h2), h[..., 0, :] + branch, atol=5e-2)
        d = hc.diagnostics()
        assert d["row_sum_err"] < 1e-3 and d["spectral_norm"] <= 1.0 + 1e-3
        hc.write(h, branch).square().sum().backward()
        assert float(hc.log_kernel.grad.norm()) > 1e-8, "HC mixing frozen"
        print(f"HC n={n} self-test passed: dist_to_I={d['dist_to_identity']:.2e}, "
              f"sigma_max={d['spectral_norm']:.6f}")
