"""
练习项目 2：Triton 实现 RMSNorm，并和 torch 对比。

包含：
1. correctness test
2. 不同 hidden_size benchmark
3. 和 torch reference 对比 latency
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    hidden_size: tl.constexpr,
    stride: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    一个 Triton program 处理一行 [hidden_size]。

    输入可理解为 x shape=[tokens, hidden_size]。
    每行做：
        out = x * rsqrt(mean(x^2) + eps) * weight
    """

    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size

    x = tl.load(x_ptr + row * stride + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)

    mean_square = tl.sum(x * x, axis=0) / hidden_size
    inv_rms = tl.rsqrt(mean_square + eps)
    out = x * inv_rms * weight

    tl.store(out_ptr + row * stride + offsets, out, mask=mask)


def next_power_of_2(x: int) -> int:
    """Triton block size 通常取 2 的幂，方便 reduction。"""

    return 1 << (x - 1).bit_length()


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Triton RMSNorm Python wrapper。"""

    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    assert weight.is_cuda and weight.dim() == 1

    tokens, hidden_size = x.shape
    out = torch.empty_like(x)
    block_size = next_power_of_2(hidden_size)

    rmsnorm_kernel[(tokens,)](
        x,
        weight,
        out,
        hidden_size,
        x.stride(0),
        eps,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return out


def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """torch reference，作为正确性和性能对照。"""

    return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps) * weight


def benchmark_ms(fn, warmup: int = 20, repeat: int = 100) -> float:
    """用 CUDA event 计时，返回平均 latency ms。"""

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def run_case(tokens: int, hidden_size: int) -> None:
    """跑一个 hidden_size 配置。"""

    x = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.float32)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float32)

    triton_out = rmsnorm_triton(x, weight)
    torch_out = rmsnorm_torch(x, weight)
    max_error = (triton_out - torch_out).abs().max().item()

    triton_ms = benchmark_ms(lambda: rmsnorm_triton(x, weight))
    torch_ms = benchmark_ms(lambda: rmsnorm_torch(x, weight))
    speedup = torch_ms / triton_ms if triton_ms > 0 else float("inf")

    print(
        f"tokens={tokens:<5} hidden={hidden_size:<5} "
        f"error={max_error:.3e} triton={triton_ms:.4f}ms "
        f"torch={torch_ms:.4f}ms speedup={speedup:.2f}x"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 才能运行 Triton benchmark。")

    torch.manual_seed(0)
    print("device:", torch.cuda.get_device_name(0))
    print("\n=== Triton RMSNorm correctness + hidden_size benchmark ===")

    for hidden_size in [128, 256, 512, 1024, 2048, 4096]:
        run_case(tokens=1024, hidden_size=hidden_size)


if __name__ == "__main__":
    main()
