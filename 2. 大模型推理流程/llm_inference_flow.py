"""
大模型推理流程入门。

这份代码用一个极简 Transformer block 演示：

1. prefill
2. decode
3. KV cache
4. batching
5. TTFT
6. TPOT
7. tokens/s
8. 显存占用估算

注意：
- 这里的模型是随机初始化的玩具模型，不会生成有意义文本。
- 目标是学习推理系统的执行路径，而不是模型效果。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KVCache:
    """
    保存单层 Transformer 的 K/V cache。

    k shape: [batch, heads, cached_seq_len, head_dim]
    v shape: [batch, heads, cached_seq_len, head_dim]

    在真实 LLM 中，每一层都有自己的 KV cache：
    cache[layer_id].k / cache[layer_id].v
    """

    k: torch.Tensor
    v: torch.Tensor


@dataclass
class InferenceStats:
    """记录推理指标，方便理解 TTFT / TPOT / tokens/s。"""

    ttft_s: float
    decode_s: float
    decode_tokens: int

    @property
    def tpot_s(self) -> float:
        """TPOT: decode 阶段平均每个 token step 的耗时。"""

        if self.decode_tokens == 0:
            return 0.0
        return self.decode_s / self.decode_tokens

    @property
    def tokens_per_s(self) -> float:
        """tokens/s: decode 阶段吞吐。"""

        if self.decode_s == 0:
            return float("inf")
        return self.decode_tokens / self.decode_s


class TinyLLMBlock(nn.Module):
    """
    一个最小可读的 Transformer decoder block。

    真实大模型会堆叠很多层，这里只保留一层，方便观察：
    - RMSNorm
    - self-attention
    - SwiGLU MLP
    """

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # q/k/v/o 是 attention 的四个线性投影。
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # MLP 使用 SwiGLU 结构。
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.attn_norm = nn.RMSNorm(hidden_size)
        self.mlp_norm = nn.RMSNorm(hidden_size)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, seq_len, hidden] -> [batch, heads, seq_len, head_dim]。"""

        batch, seq_len, hidden = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, heads, seq_len, head_dim] -> [batch, seq_len, hidden]。"""

        batch, heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, heads * head_dim)

    def attention(
        self,
        x: torch.Tensor,
        cache: KVCache | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """
        执行 self-attention，并可选地读写 KV cache。

        prefill:
        - x 是整段 prompt，seq_len > 1。
        - 计算整段 prompt 的 K/V，并写入 cache。

        decode:
        - x 通常只有 1 个新 token，seq_len = 1。
        - 新 token 的 K/V 追加到历史 cache 后面。
        - Q 只来自新 token，但 K/V 来自完整历史。
        """

        batch, query_len, _ = x.shape

        q = self.split_heads(self.q_proj(x))
        new_k = self.split_heads(self.k_proj(x))
        new_v = self.split_heads(self.v_proj(x))

        if cache is None:
            # prefill 第一次进入时，没有历史 cache。
            k = new_k
            v = new_v
        else:
            # decode 时把新 token 的 K/V 拼到历史后面。
            # 真实推理框架通常会预分配 cache，然后原地写入，避免频繁 cat。
            k = torch.cat([cache.k, new_k], dim=2)
            v = torch.cat([cache.v, new_v], dim=2)

        # 保存更新后的 cache，下一轮 decode 继续复用。
        next_cache = KVCache(k=k, v=v) if use_cache else None

        key_len = k.shape[2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # causal mask:
        # prefill 阶段 query_len > 1，需要屏蔽未来 token。
        # decode 阶段 query_len = 1，query 在序列末尾，天然只能看到历史和自己。
        if query_len > 1:
            causal_mask = torch.triu(
                torch.ones(query_len, key_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        context = weights @ v
        output = self.o_proj(self.merge_heads(context))
        return output, next_cache

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """Transformer block 前向：attention 残差 + MLP 残差。"""

        attn_input = self.attn_norm(x)
        attn_output, next_cache = self.attention(attn_input, cache, use_cache)
        x = x + attn_output

        mlp_input = self.mlp_norm(x)
        mlp_output = self.down_proj(F.silu(self.gate_proj(mlp_input)) * self.up_proj(mlp_input))
        x = x + mlp_output

        return x, next_cache


class TinyLLM(nn.Module):
    """
    一个极简 decoder-only LLM。

    输入 token ids:
    [batch, seq_len]

    输出 logits:
    [batch, seq_len, vocab_size]
    """

    def __init__(
        self,
        vocab_size: int = 128,
        hidden_size: int = 64,
        num_heads: int = 4,
        intermediate_size: int = 128,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.block = TinyLLMBlock(hidden_size, num_heads, intermediate_size)
        self.final_norm = nn.RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: KVCache | None = None,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, KVCache | None]:
        x = self.embed_tokens(input_ids)
        x, next_cache = self.block(x, cache=cache, use_cache=use_cache)
        logits = self.lm_head(self.final_norm(x))
        return logits, next_cache


def estimate_kv_cache_bytes(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    bytes_per_element: int,
) -> int:
    """
    估算 KV cache 显存。

    每层缓存 K 和 V 两份 tensor：
    batch * heads * seq_len * head_dim * 2(K,V) * bytes_per_element

    真实模型还会有权重、激活、workspace、碎片等额外开销。
    """

    return batch_size * num_layers * 2 * num_heads * seq_len * head_dim * bytes_per_element


def format_mib(num_bytes: int) -> str:
    """把字节格式化成 MiB，便于读显存规模。"""

    return f"{num_bytes / 1024 / 1024:.2f} MiB"


@torch.inference_mode()
def generate(
    model: TinyLLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
) -> tuple[torch.Tensor, KVCache, InferenceStats]:
    """
    演示一次完整生成。

    关键流程：
    1. prefill: 输入完整 prompt，建立 KV cache，拿到第一个 next token。
    2. decode: 每次只输入上一个 token，复用并扩展 KV cache。
    """

    generated_tokens: list[torch.Tensor] = []

    # ---------- prefill ----------
    # prefill 的输入是完整 prompt，shape=[batch, prompt_len]。
    # 这一步通常计算量大，但可以并行处理整段 prompt。
    prefill_start = time.perf_counter()
    logits, cache = model(prompt_ids, cache=None, use_cache=True)

    # 只取最后一个位置的 logits 来预测第一个新 token。
    # logits[:, -1, :] shape=[batch, vocab_size]
    #
    # 注意：这个 token 已经可以返回给用户，所以 TTFT 到这里结束。
    # 但它还没有作为模型输入跑过一轮，因此此刻 KV cache 只包含 prompt。
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    generated_tokens.append(next_token)
    ttft_s = time.perf_counter() - prefill_start

    # ---------- decode ----------
    # decode 每轮只输入 1 个 token。
    # 这一步串行依赖强，因此 TPOT 对用户体验非常重要。
    decode_start = time.perf_counter()
    decode_tokens = max_new_tokens - 1
    for _ in range(decode_tokens):
        logits, cache = model(next_token, cache=cache, use_cache=True)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens.append(next_token)

    decode_s = time.perf_counter() - decode_start

    output_ids = torch.cat(generated_tokens, dim=1)
    stats = InferenceStats(
        ttft_s=ttft_s,
        decode_s=decode_s,
        decode_tokens=decode_tokens,
    )
    assert cache is not None
    return output_ids, cache, stats


def print_run_summary(
    prompt_ids: torch.Tensor,
    output_ids: torch.Tensor,
    cache: KVCache,
    stats: InferenceStats,
    model: TinyLLM,
) -> None:
    """打印一次推理的 shape、指标和显存估算。"""

    batch_size, prompt_len = prompt_ids.shape
    output_len = output_ids.shape[1]
    cached_seq_len = cache.k.shape[2]

    print("\n=== 推理结果 ===")
    print(f"prompt_ids shape: {tuple(prompt_ids.shape)}")
    print(f"output_ids shape: {tuple(output_ids.shape)}")
    print(f"KV cache K shape: {tuple(cache.k.shape)}")
    print(f"KV cache V shape: {tuple(cache.v.shape)}")
    print(f"cache 中已处理 token 数: {cached_seq_len}")
    print("说明: 最后一个已生成 token 还未作为下一轮输入，因此不在当前 cache 中。")

    print("\n=== 性能指标 ===")
    print(f"TTFT: {stats.ttft_s * 1000:.2f} ms")
    print(f"Decode total: {stats.decode_s * 1000:.2f} ms")
    print(f"TPOT: {stats.tpot_s * 1000:.2f} ms/token")
    print(f"tokens/s: {stats.tokens_per_s:.2f}")

    bytes_per_element = 4  # float32。真实推理常用 fp16/bf16，即 2 bytes。
    kv_bytes = estimate_kv_cache_bytes(
        batch_size=batch_size,
        seq_len=cached_seq_len,
        num_layers=1,
        num_heads=model.block.num_heads,
        head_dim=model.block.head_dim,
        bytes_per_element=bytes_per_element,
    )

    print("\n=== 显存占用估算 ===")
    print(f"当前示例 KV cache 估算: {format_mib(kv_bytes)}")
    print("公式: batch * layers * 2(K,V) * heads * seq_len * head_dim * bytes")


def demo_single_request() -> None:
    """演示单请求：最容易看清 prefill 和 decode。"""

    print("=== 单请求推理：prefill + decode + KV cache ===")

    model = TinyLLM().eval()

    # 一个请求，prompt 长度为 8。
    prompt_ids = torch.randint(low=0, high=128, size=(1, 8))
    output_ids, cache, stats = generate(model, prompt_ids, max_new_tokens=6)
    print_run_summary(prompt_ids, output_ids, cache, stats, model)


def demo_batching() -> None:
    """演示 batching：多个请求合成一个 batch 一起推理。"""

    print("\n=== batching 推理 ===")

    model = TinyLLM().eval()

    # 这里为了简单，batch 内 prompt 长度相同。
    # 真实服务中请求长度经常不同，需要 padding、packing 或 paged KV cache。
    batch_size = 4
    prompt_len = 8
    prompt_ids = torch.randint(low=0, high=128, size=(batch_size, prompt_len))

    output_ids, cache, stats = generate(model, prompt_ids, max_new_tokens=6)
    print_run_summary(prompt_ids, output_ids, cache, stats, model)

    print("\n观察点:")
    print("- batch 变大后，单轮计算更满，吞吐通常更好。")
    print("- 但等待凑 batch 可能增加排队时间，影响 TTFT。")
    print("- KV cache 显存随 batch_size 和 seq_len 线性增长。")


def demo_realistic_kv_memory() -> None:
    """用更接近真实模型的参数估算 KV cache，帮助建立显存直觉。"""

    print("\n=== 真实规模 KV cache 估算 ===")

    # 近似一个 7B 量级模型的配置，不绑定具体模型。
    batch_size = 8
    seq_len = 4096
    num_layers = 32
    num_heads = 32
    head_dim = 128
    bytes_per_element = 2  # fp16 / bf16

    kv_bytes = estimate_kv_cache_bytes(
        batch_size=batch_size,
        seq_len=seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        bytes_per_element=bytes_per_element,
    )

    print(f"batch_size={batch_size}, seq_len={seq_len}, layers={num_layers}")
    print(f"heads={num_heads}, head_dim={head_dim}, dtype=fp16/bf16")
    print(f"KV cache 约占用: {format_mib(kv_bytes)}")
    print("这解释了为什么长上下文、大 batch、并发请求会很快吃掉显存。")


def main() -> None:
    # 固定随机种子，方便多次运行时对照输出。
    torch.manual_seed(0)

    demo_single_request()
    demo_batching()
    demo_realistic_kv_memory()


if __name__ == "__main__":
    main()
