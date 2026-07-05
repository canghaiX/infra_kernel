"""
练习项目 1：PyTorch 实现 RMSNorm、Softmax、RoPE。

目标：
1. 写出简洁、正确、可读的 PyTorch reference。
2. 对每个算子做 correctness test。
3. 重点观察 shape 和数值稳定性。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    RMSNorm reference。

    x shape: [..., hidden]
    weight shape: [hidden]

    RMSNorm 不减均值，只用 root mean square 做归一化：
        y = x / sqrt(mean(x^2) + eps) * weight
    """

    mean_square = (x * x).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_square + eps)
    return x * inv_rms * weight


def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    数值稳定 softmax。

    直接 exp(x) 可能溢出，所以先减去最大值：
        softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    """

    x_max = x.max(dim=dim, keepdim=True).values
    numerator = torch.exp(x - x_max)
    denominator = numerator.sum(dim=dim, keepdim=True)
    return numerator / denominator


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """
    RoPE reference。

    x shape: [batch, heads, seq_len, head_dim]
    要求 head_dim 是偶数，因为 RoPE 会把最后一维两两配对旋转。
    """

    batch, heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0

    positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device=x.device, dtype=x.dtype) / head_dim)
    )
    angles = torch.outer(positions, inv_freq)[None, None, :, :]
    cos = angles.cos()
    sin = angles.sin()

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


def check_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-6) -> None:
    """统一打印 correctness test 结果。"""

    max_error = (actual - expected).abs().max().item()
    ok = torch.allclose(actual, expected, atol=atol, rtol=1e-5)
    status = "PASS" if ok else "FAIL"
    print(f"{name:<12} {status} max_error={max_error:.6e}, shape={tuple(actual.shape)}")
    if not ok:
        raise AssertionError(f"{name} correctness check failed")


def test_rmsnorm() -> None:
    """和 torch.nn.functional.rms_norm 对比。"""

    x = torch.randn(2, 4, 16)
    weight = torch.randn(16)
    actual = rmsnorm(x, weight)
    expected = F.rms_norm(x, normalized_shape=(16,), weight=weight, eps=1e-6)
    check_close("RMSNorm", actual, expected)


def test_softmax() -> None:
    """和 torch.softmax 对比，同时验证每行概率和为 1。"""

    x = torch.randn(4, 9) * 5
    actual = stable_softmax(x, dim=-1)
    expected = torch.softmax(x, dim=-1)
    check_close("Softmax", actual, expected)
    print("softmax row sums:", actual.sum(dim=-1))


def test_rope() -> None:
    """
    RoPE 的一个简单性质：
    position=0 时 angle=0，所以 cos=1、sin=0，旋转前后应该完全相同。
    """

    x = torch.randn(2, 3, 8, 16)
    y = apply_rope(x)
    check_close("RoPE pos0", y[:, :, 0, :], x[:, :, 0, :])
    print("RoPE output shape:", tuple(y.shape))


def main() -> None:
    torch.manual_seed(0)
    test_rmsnorm()
    test_softmax()
    test_rope()


if __name__ == "__main__":
    main()
