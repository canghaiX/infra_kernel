"""
PyTorch 和 tensor 计算入门。

这份代码不是为了训练一个模型，而是把大模型里最常见的几个基础算子拆开：

1. matmul
2. softmax
3. attention
4. RMSNorm
5. RoPE
6. SiLU / SwiGLU

阅读重点：
- 每一步的 tensor shape 是什么。
- 哪些维度代表 batch、序列长度、hidden size、head 数量。
- 为什么这些操作会成为 CUDA / 推理优化的重点。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def show(name: str, x: torch.Tensor) -> None:
    """打印 tensor 名称、shape 和 dtype，方便观察 shape 怎么流动。"""

    print(f"{name:<24} shape={tuple(x.shape)}, dtype={x.dtype}")


def demo_matmul() -> None:
    """演示 matmul：线性层和 attention 都离不开矩阵乘法。"""

    print("\n=== 1. matmul ===")

    # 一个小 batch：
    # batch_size = 2，seq_len = 4，hidden_size = 8。
    # 在大模型中，hidden_size 可能是 4096、8192 甚至更大。
    x = torch.randn(2, 4, 8)

    # 线性层权重：把 hidden_size=8 映射到 out_features=16。
    # PyTorch 里也可以用 nn.Linear，这里手写 matmul 方便看 shape。
    weight = torch.randn(8, 16)

    # 对最后一维做矩阵乘法：
    # [2, 4, 8] @ [8, 16] -> [2, 4, 16]
    y = x @ weight

    show("x", x)
    show("weight", weight)
    show("y = x @ weight", y)


def demo_softmax() -> None:
    """演示 softmax：把任意 logits 归一化为概率。"""

    print("\n=== 2. softmax ===")

    logits = torch.tensor([[1.0, 2.0, 3.0], [10.0, 0.0, -10.0]])

    # dim=-1 表示在最后一个维度上归一化。
    # 每一行 softmax 之后的和都是 1。
    probs = F.softmax(logits, dim=-1)

    show("logits", logits)
    show("probs", probs)
    print("每行概率和:", probs.sum(dim=-1))


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    把 [batch, seq_len, hidden] 拆成 [batch, heads, seq_len, head_dim]。

    attention 通常按 head 并行计算。
    hidden_size 必须能被 num_heads 整除：
    head_dim = hidden_size // num_heads。
    """

    batch, seq_len, hidden = x.shape
    assert hidden % num_heads == 0
    head_dim = hidden // num_heads

    # view 先拆出 heads 维度：
    # [batch, seq_len, hidden] -> [batch, seq_len, heads, head_dim]
    x = x.view(batch, seq_len, num_heads, head_dim)

    # transpose 把 heads 放到 seq_len 前面：
    # [batch, seq_len, heads, head_dim] -> [batch, heads, seq_len, head_dim]
    return x.transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """把 [batch, heads, seq_len, head_dim] 合回 [batch, seq_len, hidden]。"""

    batch, heads, seq_len, head_dim = x.shape

    # transpose 后 tensor 可能不是连续内存，因此 contiguous 后再 view。
    x = x.transpose(1, 2).contiguous()
    return x.view(batch, seq_len, heads * head_dim)


def demo_attention() -> None:
    """演示标准 causal self-attention。"""

    print("\n=== 3. attention ===")

    batch = 2
    seq_len = 4
    hidden = 8
    num_heads = 2
    head_dim = hidden // num_heads

    x = torch.randn(batch, seq_len, hidden)

    # 大模型里通常会有三个线性层，把 x 投影成 Q/K/V。
    wq = torch.randn(hidden, hidden)
    wk = torch.randn(hidden, hidden)
    wv = torch.randn(hidden, hidden)

    q = split_heads(x @ wq, num_heads)
    k = split_heads(x @ wk, num_heads)
    v = split_heads(x @ wv, num_heads)

    # QK^T:
    # q: [batch, heads, query_len, head_dim]
    # k.transpose(-2, -1): [batch, heads, head_dim, key_len]
    # scores: [batch, heads, query_len, key_len]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)

    # causal mask 用来禁止当前位置看到未来 token。
    # 上三角为 True 的位置会被填成 -inf，softmax 后概率接近 0。
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))

    # 对 key_len 维度做 softmax，得到每个 query 对历史 token 的注意力权重。
    attn_weights = F.softmax(scores, dim=-1)

    # attention 输出：
    # [batch, heads, query_len, key_len] @ [batch, heads, key_len, head_dim]
    # -> [batch, heads, query_len, head_dim]
    context = attn_weights @ v
    output = merge_heads(context)

    show("x", x)
    show("q", q)
    show("k", k)
    show("v", v)
    show("scores = q @ k^T", scores)
    show("attn_weights", attn_weights)
    show("context", context)
    show("output", output)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    RMSNorm。

    LayerNorm 会减均值再除以标准差；RMSNorm 只除以均方根：
    rms = sqrt(mean(x^2))

    这在很多 LLM 中很常见，例如 LLaMA 系列使用 RMSNorm。
    """

    # mean(dim=-1, keepdim=True) 表示每个 token 单独算自己的 RMS。
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    normalized = x / rms

    # weight 是可学习缩放参数，shape=[hidden]，会广播到 batch 和 seq_len。
    return normalized * weight


def demo_rmsnorm() -> None:
    """演示 RMSNorm 的输入输出 shape 不变。"""

    print("\n=== 4. rmsnorm ===")

    x = torch.randn(2, 4, 8)
    weight = torch.ones(8)
    y = rmsnorm(x, weight)

    show("x", x)
    show("weight", weight)
    show("y = rmsnorm(x)", y)


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """
    给 Q 或 K 应用 RoPE 旋转位置编码。

    输入 shape: [batch, heads, seq_len, head_dim]
    输出 shape: [batch, heads, seq_len, head_dim]

    RoPE 的直觉：
    - 把 head_dim 两两分组。
    - 每一组看成二维平面上的一个点。
    - 根据 token position 旋转这个点，从而注入位置信息。
    """

    batch, heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "RoPE 要求 head_dim 是偶数，方便两两旋转"

    device = x.device

    # inv_freq 控制不同维度的旋转频率。
    # 低维旋转更快，高维旋转更慢，这是 Transformer 位置编码的常见设计。
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    # positions: [seq_len]
    positions = torch.arange(seq_len, device=device).float()

    # freqs: [seq_len, head_dim / 2]
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos()[None, None, :, :]
    sin = freqs.sin()[None, None, :, :]

    # 偶数维和奇数维两两配对。
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # 二维旋转公式：
    # [a, b] -> [a*cos - b*sin, a*sin + b*cos]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    # 把偶数维和奇数维交错放回原来的 head_dim。
    rotated = torch.empty_like(x)
    rotated[..., 0::2] = rotated_even
    rotated[..., 1::2] = rotated_odd
    return rotated


def demo_rope() -> None:
    """演示 RoPE：shape 不变，但 Q/K 的数值带上了位置信息。"""

    print("\n=== 5. rope ===")

    x = torch.randn(2, 2, 4, 4)
    y = apply_rope(x)

    show("x", x)
    show("y = apply_rope(x)", y)
    print("第 0 个位置旋转前后接近相同:", torch.allclose(x[:, :, 0, :], y[:, :, 0, :]))


def demo_silu_swiglu() -> None:
    """演示 SiLU 和 SwiGLU：现代 LLM 的 MLP 常用门控结构。"""

    print("\n=== 6. silu / swiglu ===")

    batch = 2
    seq_len = 4
    hidden = 8
    intermediate = 16

    x = torch.randn(batch, seq_len, hidden)

    # SiLU(x) = x * sigmoid(x)，也叫 Swish。
    silu_x = F.silu(x)

    # SwiGLU 的常见形式：
    # down_proj( silu(gate_proj(x)) * up_proj(x) )
    # 这里手写权重，只展示 shape。
    w_gate = torch.randn(hidden, intermediate)
    w_up = torch.randn(hidden, intermediate)
    w_down = torch.randn(intermediate, hidden)

    gate = F.silu(x @ w_gate)
    up = x @ w_up
    hidden_states = gate * up
    y = hidden_states @ w_down

    show("x", x)
    show("silu(x)", silu_x)
    show("gate", gate)
    show("up", up)
    show("gate * up", hidden_states)
    show("swiglu output", y)


def main() -> None:
    # 固定随机种子，让每次运行的输出更容易对照。
    torch.manual_seed(0)

    demo_matmul()
    demo_softmax()
    demo_attention()
    demo_rmsnorm()
    demo_rope()
    demo_silu_swiglu()


if __name__ == "__main__":
    main()
