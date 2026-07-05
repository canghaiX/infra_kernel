"""
Triton 算子入门。

这份代码用 PyTorch 做 reference，用 Triton 写 kernel，并检查两者误差。

重点不是追求极致性能，而是先看懂这些 GPU kernel 的共同套路：

1. program_id: 当前 Triton program 负责哪一块数据。
2. block / tile: 一次处理多少元素或矩阵块。
3. mask: 处理越界，尤其是长度不能整除 block size 时。
4. tl.load / tl.store: 显存读写。
5. parallel reduction: 在一个 block 内并行求 max、sum、平方和。
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


def require_cuda() -> None:
    """Triton kernel 需要 NVIDIA GPU。"""

    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到 CUDA GPU，无法运行 Triton 示例。")


def next_power_of_2(x: int) -> int:
    """Triton reduction 常把 block size 取成 2 的幂，便于并行规约。"""

    return 1 << (x - 1).bit_length()


def report_error(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    """打印最大误差，帮助确认 Triton kernel 算得对。"""

    max_error = (actual - expected).abs().max().item()
    print(f"{name:<16} max_error={max_error:.6e}, shape={tuple(actual.shape)}")


@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    最简单的 Triton kernel：向量加法。

    一个 program 处理 BLOCK_SIZE 个元素。
    如果 n_elements 不能整除 BLOCK_SIZE，最后一个 program 会越界，所以必须用 mask。
    """

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Python wrapper：负责分配输出、配置 grid、启动 Triton kernel。"""

    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape

    out = torch.empty_like(x)
    n_elements = x.numel()
    block_size = 1024

    # grid 表示启动多少个 Triton program。
    grid = (triton.cdiv(n_elements, block_size),)
    vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size)
    return out


@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    行级 softmax。

    一个 program 处理矩阵的一行：
    1. load 一整行。
    2. 减去行最大值，避免 exp 溢出。
    3. exp 后求和。
    4. 除以 sum 并写回。
    """

    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(x_ptr + row * stride + offsets, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    out = numerator / denominator

    tl.store(out_ptr + row * stride + offsets, out, mask=mask)


def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    """对二维 tensor 的最后一维做 softmax。"""

    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()

    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block_size = next_power_of_2(n_cols)

    softmax_kernel[(n_rows,)](
        x,
        out,
        n_cols,
        x.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out


@triton.jit
def rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    stride: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    行级 RMSNorm。

    输入通常可以看成 [tokens, hidden]：
    - 每个 program 处理一个 token 的 hidden 向量。
    - 先求 mean(x^2)，再乘 rsqrt。
    - 最后乘可学习的 weight。
    """

    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(x_ptr + row * stride + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)

    mean_square = tl.sum(x * x, axis=0) / n_cols
    inv_rms = tl.rsqrt(mean_square + eps)
    out = x * inv_rms * weight

    tl.store(out_ptr + row * stride + offsets, out, mask=mask)


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Triton RMSNorm wrapper。"""

    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    assert weight.is_cuda and weight.dim() == 1 and weight.numel() == x.shape[1]

    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block_size = next_power_of_2(n_cols)

    rmsnorm_kernel[(n_rows,)](
        x,
        weight,
        out,
        n_cols,
        x.stride(0),
        eps,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out


@triton.jit
def rope_kernel(
    x_ptr,
    out_ptr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    """
    RoPE kernel。

    输入布局: [tokens, heads, head_dim]，连续内存。
    一个 program 处理一个 token 的一个 head。

    head_dim 两两分组：
    even = 0, 2, 4, ...
    odd  = 1, 3, 5, ...
    然后做二维旋转。
    """

    token_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)

    half_offsets = tl.arange(0, BLOCK_HALF)
    mask = half_offsets < (head_dim // 2)

    base = (token_id * n_heads + head_id) * head_dim
    even_offsets = base + half_offsets * 2
    odd_offsets = even_offsets + 1

    x_even = tl.load(x_ptr + even_offsets, mask=mask, other=0.0)
    x_odd = tl.load(x_ptr + odd_offsets, mask=mask, other=0.0)

    # inv_freq = 1 / 10000 ** (2i / head_dim)
    inv_freq = tl.exp(-tl.log(10000.0) * ((half_offsets * 2).to(tl.float32) / head_dim))
    angle = token_id * inv_freq
    cos = tl.cos(angle)
    sin = tl.sin(angle)

    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    tl.store(out_ptr + even_offsets, out_even, mask=mask)
    tl.store(out_ptr + odd_offsets, out_odd, mask=mask)


def rope_triton(x: torch.Tensor) -> torch.Tensor:
    """对 [tokens, heads, head_dim] 应用 RoPE。"""

    assert x.is_cuda and x.dim() == 3 and x.is_contiguous()
    _, n_heads, head_dim = x.shape
    assert head_dim % 2 == 0

    out = torch.empty_like(x)
    block_half = next_power_of_2(head_dim // 2)

    rope_kernel[(x.shape[0], n_heads)](
        x,
        out,
        n_heads,
        head_dim,
        BLOCK_HALF=block_half,
        num_warps=4,
    )
    return out


def rope_reference(x: torch.Tensor) -> torch.Tensor:
    """PyTorch 版 RoPE，用来校验 Triton kernel。"""

    tokens, _, head_dim = x.shape
    half_dim = head_dim // 2
    positions = torch.arange(tokens, device=x.device, dtype=x.dtype)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=x.device, dtype=x.dtype) / head_dim))
    angle = torch.outer(positions, inv_freq)[:, None, :]
    cos = angle.cos()
    sin = angle.sin()

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    assert out.shape[-1] == half_dim * 2
    return out


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    tiled matmul: C = A @ B。

    A shape: [M, K]
    B shape: [K, N]
    C shape: [M, N]

    一个 program 计算 C 的一个 [BLOCK_M, BLOCK_N] tile。
    K 维按 BLOCK_K 分块累加。
    """

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + offs_k

        a = tl.load(
            a_ptr + offs_m[:, None] * K + k_offsets[None, :],
            mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + k_offsets[:, None] * N + offs_n[None, :],
            mask=(k_offsets[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        # input_precision="ieee" 更偏向精确校验。
        # 真实性能优化时常会使用 TF32 / FP16 / BF16 来换取吞吐。
        acc += tl.dot(a, b, input_precision="ieee")

    tl.store(
        c_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def matmul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Triton matmul wrapper。"""

    assert a.is_cuda and b.is_cuda
    assert a.dim() == 2 and b.dim() == 2
    assert a.shape[1] == b.shape[0]
    assert a.is_contiguous() and b.is_contiguous()

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    block_m = 16
    block_n = 16
    block_k = 32
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return c


def main() -> None:
    require_cuda()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False

    print("device:", torch.cuda.get_device_name(0))
    print("\n=== Triton correctness check ===")

    x = torch.randn(10000, device="cuda")
    y = torch.randn(10000, device="cuda")
    report_error("vector_add", vector_add(x, y), x + y)

    logits = torch.randn(128, 257, device="cuda")
    report_error("softmax", softmax_triton(logits), torch.softmax(logits, dim=-1))

    hidden = torch.randn(128, 256, device="cuda")
    weight = torch.randn(256, device="cuda")
    expected_rms = hidden * torch.rsqrt((hidden * hidden).mean(dim=-1, keepdim=True) + 1e-6) * weight
    report_error("rmsnorm", rmsnorm_triton(hidden, weight), expected_rms)

    rope_input = torch.randn(32, 4, 64, device="cuda")
    report_error("rope", rope_triton(rope_input), rope_reference(rope_input))

    a = torch.randn(65, 96, device="cuda")
    b = torch.randn(96, 80, device="cuda")
    report_error("matmul", matmul_triton(a, b), a @ b)

    print("\n所有 Triton kernel 已完成。第一次运行可能包含 JIT 编译时间。")


if __name__ == "__main__":
    main()
