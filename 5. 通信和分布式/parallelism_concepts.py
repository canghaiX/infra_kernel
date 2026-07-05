"""
TP / PP / DP / EP 的 shape 入门。

这份脚本不启动多进程，也不真的跨 GPU 通信。
它只用 tensor shape 演示：不同并行策略到底在切什么。

如果你刚开始学分布式，先看 shape 往往比先看框架源码更有效。
"""

from __future__ import annotations

import torch


def show(name: str, tensors: list[torch.Tensor] | torch.Tensor) -> None:
    """打印一个 tensor 或一组 tensor 的 shape。"""

    if isinstance(tensors, torch.Tensor):
        print(f"{name:<28} shape={tuple(tensors.shape)}")
    else:
        shapes = [tuple(t.shape) for t in tensors]
        print(f"{name:<28} shards={shapes}")


def demo_data_parallel() -> None:
    """
    DP: Data Parallel。

    每张卡都有完整模型，但处理不同 batch。
    训练时每张卡得到一份梯度，然后用 AllReduce 求平均。
    """

    print("\n=== DP: Data Parallel ===")

    batch = torch.randn(8, 16)
    shards = list(torch.chunk(batch, chunks=2, dim=0))

    show("global batch", batch)
    show("rank batches", shards)
    print("通信直觉: backward 后对每个参数梯度做 AllReduce。")


def demo_tensor_parallel_column() -> None:
    """
    TP: Tensor Parallel 的列切分。

    线性层 y = x @ W。
    如果 W 按输出维度切成两片，每张卡计算一部分输出，最后 AllGather 拼起来。
    """

    print("\n=== TP: column parallel linear ===")

    x = torch.randn(4, 8)
    weight = torch.randn(8, 12)
    weight_shards = list(torch.chunk(weight, chunks=2, dim=1))
    y_shards = [x @ shard for shard in weight_shards]
    y = torch.cat(y_shards, dim=1)

    show("x", x)
    show("weight shards by out_dim", weight_shards)
    show("partial y shards", y_shards)
    show("AllGather y", y)
    print("通信直觉: 输出维度被切开，需要 AllGather 得到完整 hidden。")


def demo_tensor_parallel_row() -> None:
    """
    TP: Tensor Parallel 的行切分。

    W 按输入维度切开，x 也按 hidden 切开。
    每张卡算局部 matmul，最后 AllReduce 求和。
    """

    print("\n=== TP: row parallel linear ===")

    x = torch.randn(4, 8)
    weight = torch.randn(8, 12)
    x_shards = list(torch.chunk(x, chunks=2, dim=1))
    weight_shards = list(torch.chunk(weight, chunks=2, dim=0))
    partial = [xs @ ws for xs, ws in zip(x_shards, weight_shards)]
    y = partial[0] + partial[1]

    show("x shards by hidden", x_shards)
    show("weight shards by in_dim", weight_shards)
    show("partial outputs", partial)
    show("AllReduce SUM y", y)
    print("通信直觉: 每张卡只有部分乘积，需要 AllReduce 求和。")


def demo_pipeline_parallel() -> None:
    """
    PP: Pipeline Parallel。

    不同层放在不同 rank。
    rank0 跑前几层，把 activation 发给 rank1，rank1 跑后几层。
    """

    print("\n=== PP: Pipeline Parallel ===")

    micro_batches = [torch.randn(2, 16) for _ in range(4)]
    show("micro batches", micro_batches)
    print("rank0: layers 0-11，处理 micro_batch_i 后发送 activation。")
    print("rank1: layers 12-23，接收 activation 后继续前向。")
    print("通信直觉: 相邻 pipeline stage 之间 send/recv activation 和 gradient。")


def demo_expert_parallel() -> None:
    """
    EP: Expert Parallel。

    MoE 模型里，不同 expert 可以放到不同 rank。
    每个 token 先经过 router，决定发给哪个 expert。
    这通常需要 All-to-All。
    """

    print("\n=== EP: Expert Parallel / MoE ===")

    tokens = torch.randn(8, 16)

    # 假设有 2 个 rank，每个 rank 放一组 expert。
    # router_assignments[i] 表示第 i 个 token 要去哪个 rank 的 expert。
    router_assignments = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1])
    rank0_tokens = tokens[router_assignments == 0]
    rank1_tokens = tokens[router_assignments == 1]

    show("tokens", tokens)
    print("router assignments:", router_assignments.tolist())
    show("tokens to rank0 experts", rank0_tokens)
    show("tokens to rank1 experts", rank1_tokens)
    print("通信直觉: token 按 expert 路由，常用 All-to-All dispatch / combine。")


def main() -> None:
    torch.manual_seed(0)
    demo_data_parallel()
    demo_tensor_parallel_column()
    demo_tensor_parallel_row()
    demo_pipeline_parallel()
    demo_expert_parallel()


if __name__ == "__main__":
    main()
