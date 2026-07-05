"""
练习项目 3：Triton 实现 Softmax，并分析不同 seq_len 的 latency。

包含：
1. correctness test
2. 不同 seq_len benchmark
3. 和 torch.softmax 对比
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    一个 Triton program 处理一行 logits。

    x shape 可以理解为 [rows, seq_len]。
    每行做稳定 softmax：
        exp(x - max(x)) / sum(exp(x - max(x)))
    """

    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len

    x = tl.load(x_ptr + row * stride + offsets, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    out = numerator / denominator

    tl.store(out_ptr + row * stride + offsets, out, mask=mask)


def next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    """Triton softmax wrapper。"""

    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    rows, seq_len = x.shape
    out = torch.empty_like(x)
    block_size = next_power_of_2(seq_len)

    softmax_kernel[(rows,)](
        x,
        out,
        seq_len,
        x.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return out


def benchmark_ms(fn, warmup: int = 20, repeat: int = 100) -> float:
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


def run_case(rows: int, seq_len: int) -> None:
    x = torch.randn(rows, seq_len, device="cuda", dtype=torch.float32)

    triton_out = softmax_triton(x)
    torch_out = torch.softmax(x, dim=-1)
    max_error = (triton_out - torch_out).abs().max().item()

    triton_ms = benchmark_ms(lambda: softmax_triton(x))
    torch_ms = benchmark_ms(lambda: torch.softmax(x, dim=-1))
    speedup = torch_ms / triton_ms if triton_ms > 0 else float("inf")

    print(
        f"rows={rows:<5} seq_len={seq_len:<5} "
        f"error={max_error:.3e} triton={triton_ms:.4f}ms "
        f"torch={torch_ms:.4f}ms speedup={speedup:.2f}x"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 才能运行 Triton benchmark。")

    torch.manual_seed(0)
    print("device:", torch.cuda.get_device_name(0))
    print("\n=== Triton Softmax seq_len benchmark ===")

    for seq_len in [64, 128, 256, 512, 1024, 2048, 4096]:
        run_case(rows=1024, seq_len=seq_len)

    print("\n观察:")
    print("- seq_len 越大，每行 reduction 的工作越多，latency 通常会上升。")
    print("- 当 seq_len 不是很大时，kernel launch overhead 也会占明显比例。")


if __name__ == "__main__":
    main()
