"""
通信和分布式 collective 入门。

运行方式：

单机 2 卡:
    torchrun --standalone --nproc_per_node=2 "5. 通信和分布式/distributed_collectives_intro.py"

CPU/gloo:
    DIST_BACKEND=gloo torchrun --standalone --nproc_per_node=2 "5. 通信和分布式/distributed_collectives_intro.py"

这份代码演示：
1. AllReduce
2. AllGather
3. ReduceScatter
4. All-to-All

注意：
- NCCL 只能用于 CUDA tensor。
- gloo 可以用于 CPU tensor，适合没有多 GPU 时学习语义。
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def setup_distributed() -> tuple[int, int, str, torch.device]:
    """初始化 torch.distributed，并根据 backend 选择设备。"""

    backend = os.environ.get("DIST_BACKEND")
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    if backend == "nccl":
        # torchrun 会为每个进程设置 LOCAL_RANK。
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend=backend, device_id=device)
    else:
        device = torch.device("cpu")
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    return rank, world_size, backend, device


def log(rank: int, message: str) -> None:
    """带 rank 前缀打印，避免多进程输出混在一起看不清。"""

    print(f"[rank {rank}] {message}", flush=True)


def demo_all_reduce(rank: int, world_size: int, device: torch.device) -> None:
    """
    AllReduce:
    每个 rank 先有自己的 tensor，通信后每个 rank 都得到 reduce 后的完整 tensor。

    典型用途：
    - Data Parallel 训练中同步梯度。
    """

    x = torch.full((4,), float(rank + 1), device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    expected_value = sum(range(1, world_size + 1))
    log(rank, f"AllReduce SUM -> {x.tolist()}，期望每个元素都是 {expected_value}")


def demo_all_gather(rank: int, world_size: int, device: torch.device) -> None:
    """
    AllGather:
    每个 rank 持有一片 tensor，通信后每个 rank 都拿到所有 rank 的片。

    典型用途：
    - Tensor Parallel 中把被切开的 hidden / logits gather 回来。
    """

    local = torch.tensor([rank * 10 + 0, rank * 10 + 1], device=device, dtype=torch.float32)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)

    full = torch.cat(gathered)
    log(rank, f"AllGather local={local.tolist()} -> full={full.tolist()}")


def demo_reduce_scatter(rank: int, world_size: int, backend: str, device: torch.device) -> None:
    """
    ReduceScatter:
    先对所有 rank 的 input list 做 reduce，再把第 i 片结果发给 rank i。

    典型用途：
    - Tensor Parallel 中把 AllReduce 拆成 ReduceScatter + AllGather，减少中间显存。
    """

    # 每个 rank 准备 world_size 个 chunk。
    # 第 chunk_id 个 chunk 会在 reduce 后发给 rank=chunk_id。
    chunks = [
        torch.full((2,), float(rank + chunk_id), device=device)
        for chunk_id in range(world_size)
    ]
    out = torch.empty(2, device=device)

    if backend == "nccl":
        # NCCL 支持 reduce_scatter。
        dist.reduce_scatter(out, chunks, op=dist.ReduceOp.SUM)
    else:
        # 一些 gloo 版本不支持 reduce_scatter_tensor/list。
        # 这里用 all_reduce + 本地切片模拟语义，方便 CPU 环境学习。
        stacked = torch.cat(chunks)
        dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
        out.copy_(stacked[rank * 2 : (rank + 1) * 2])

    log(rank, f"ReduceScatter -> rank {rank} got {out.tolist()}")


def demo_all_to_all(rank: int, world_size: int, backend: str, device: torch.device) -> None:
    """
    All-to-All:
    每个 rank 给每个 rank 发不同切片。

    典型用途：
    - MoE / Expert Parallel 中，把 token 按 expert 路由到对应 rank。
    """

    # input_chunks[dst_rank] 表示当前 rank 要发给 dst_rank 的数据。
    input_chunks = [
        torch.tensor([rank * 100 + dst], dtype=torch.float32, device=device)
        for dst in range(world_size)
    ]
    output_chunks = [torch.empty(1, dtype=torch.float32, device=device) for _ in range(world_size)]

    if backend == "nccl":
        dist.all_to_all(output_chunks, input_chunks)
    else:
        # CPU/gloo 环境下用 all_gather 模拟 all-to-all 的语义。
        flat_send = torch.cat(input_chunks)
        gathered = [torch.empty_like(flat_send) for _ in range(world_size)]
        dist.all_gather(gathered, flat_send)
        for src in range(world_size):
            output_chunks[src].copy_(gathered[src][rank : rank + 1])

    received = torch.cat(output_chunks)
    log(rank, f"All-to-All received={received.tolist()}")


def main() -> None:
    rank, world_size, backend, device = setup_distributed()
    log(rank, f"backend={backend}, world_size={world_size}, device={device}")

    # barrier 让所有 rank 同步开始，输出更整齐。
    dist.barrier()
    demo_all_reduce(rank, world_size, device)

    dist.barrier()
    demo_all_gather(rank, world_size, device)

    dist.barrier()
    demo_reduce_scatter(rank, world_size, backend, device)

    dist.barrier()
    demo_all_to_all(rank, world_size, backend, device)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
